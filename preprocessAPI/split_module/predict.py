from transformers import AutoTokenizer, DebertaV2Model
import torch
import torch.nn.functional as F
from kiwipiepy import Kiwi
from tqdm import tqdm
from .model_setting import Config, Variables, TaggingModel, LabelData
from typing import Literal
import numpy as np
import json
from dataclasses import dataclass
import os
from safetensors.torch import load_file

@dataclass
class FileNames():
    clause_model_pt : str = "models/causal_split/causal_model_earth.safetensors"
    # splited_json : str    = 'splited.json'
    # embedding_np : str    = 'clause_embedding.npy'
    # significant_json: str = 'significant.jsonl'
    # saved_temp_dir : str  = './saved_temp'

@torch.no_grad()
def prediction(model, tokenizer, sentence, label_map, device='cuda', max_length=128, return_cls=False, return_lhs = False):
    """
    주어진 문장에 대해 토큰 단위 분류 예측을 수행하는 함수
    - 모델을 평가 모드로 설정하고 입력 토큰을 구성
    - 예측된 클래스 ID를 라벨로 변환하여 confidence와 함께 반환

    Args:
        model: 학습된 tagging model
        tokenizer: 사전 학습된 토크나이저
        sentence: 입력 문장
        label_map: 예측 ID → 라벨 매핑 dict
        device: 연산 장치 (기본값 'cuda')
        max_length: 최대 시퀀스 길이
        return_cls: [CLS] 임베딩 벡터 반환 여부

    Returns:
        List of (token, label, confidence) 또는 (위 결과, cls_vector)
    """
    # gpu에 모델이 없다면 올리기
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()

    encoding = tokenizer(
        sentence,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_attention_mask=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0]  # (L, 2)

    outputs, cls_vector = model({'input_ids': input_ids, 'attention_mask': attention_mask}, return_cls=True)
    confidences = [float(int(float(max(m)) * 10000) / 10000) for m in outputs[0]]  # 각 토큰의 confidence 정규화
    preds = torch.argmax(outputs, dim=-1)[0].cpu().tolist()  # 가장 높은 점수의 클래스 예측

    valid_len = attention_mask.sum().item()  # 실제 토큰 수
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    be_clause = []
    results = []
    for token, pred, confidence, offset in zip(tokens, preds, confidences, offset_mapping):
        if offset[0].item() == 0 and offset[1].item() == 0:
            continue  # [PAD] 토큰 제외
        be_clause.append((token, label_map[pred], confidence))
    if return_cls:
        results.append(be_clause)
        results.append(cls_vector.detach().cpu())
    if return_lhs:
        results.append(masked_lhs.detach().cpu())

    return tuple(results) if return_cls else be_clause

def recover_wordpieces(tokens: list) -> str:
    """
    WordPiece 토큰들을 원래 단어로 병합하는 함수
    - '##'로 시작하는 토큰은 이전 토큰과 연결하여 단어 복원

    Args:
        tokens: WordPiece 토큰 리스트

    Returns:
        공백 기준으로 병합된 문자열
    """
    words = []
    current_word = ''
    for token in tokens:
        if token.startswith('##'):
            current_word += token[2:]
        else:
            if current_word:
                words.append(current_word)
            current_word = token
    if current_word:
        words.append(current_word)
    return ' '.join(words)

def highlight(sentences: list[list[str]], highlight_words: list[list[list[str]]]) -> str:
    """
    강조 단어가 포함된 구문을 ANSI 색상으로 출력하는 함수 (콘솔 기반 시각화)
    - 각 절 단위로 하이라이트 단어가 포함된 경우 색상 적용

    Args:
        sentences: 절 리스트 (ex: [['구문1', '구문2'], [...], ...])
        highlight_words: 강조할 단어 리스트 (절별 단어 리스트)

    Returns:
        강조된 문장 문자열 (줄바꿈으로 이어짐)
    """
    color = ('\033[95m', '\033[0m')  # ANSI 콘솔 마젠타
    highlighted_sentences = []

    def split_by_keyword(text: str, keyword: str):
        idx = text.find(keyword)
        return [text[:idx], keyword, text[idx + len(keyword):]] if idx != -1 else []

    for clause_list, clause_keywords in zip(sentences, highlight_words):
        highlighted_clauses = []
        for clause, keywords in zip(clause_list, clause_keywords):
            result = []
            for word in clause.split():
                q = sum([split_by_keyword(word, term) for term in keywords], [])
                if q:
                    result.append(f"{q[0]}{color[0]}{q[1]}{color[1]}{q[2]}")
                else:
                    result.append(word)
            highlighted_clauses.append(' '.join(result))
        highlighted_sentence = ' / '.join(highlighted_clauses)
        highlighted_sentences.append(highlighted_sentence)

    return '\n'.join(highlighted_sentences)

def highlight_jsonl(jsonl_path: str):
    """
    JSONL 파일을 읽어 색상 강조된 문장을 반환하는 함수
    - 각 라인은 절 리스트와 강조 단어 리스트를 포함해야 함

    Args:
        jsonl_path: JSONL 파일 경로

    Returns:
        강조된 문장 출력 문자열
    """
    sentences, highlight_words = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sentences.append(item["clause"])
            highlight_words.append(item["highlight"])
    return highlight(sentences, highlight_words)

class ClauseSpliting:
    """
    입력 문장을 토큰 분류 모델로 절 단위로 분할하고, 각 절에 대해 임베딩 및 핵심 단어를 추출하는 클래스

    주요 기능:
    - 문장 내 구문 분리 (E/E2/E3 태그 기반)
    - 각 구문별 DeBERTa 기반 [CLS] 임베딩 추출
    - 구문 임베딩과 토큰 임베딩 유사도를 통해 의미 강조 단어 탐색
    """
    def __init__(self, sentences, config = Config(), filenames =FileNames(), e_option: Literal['all', 'E3', 'E2', 'E'] = 'E3', threshold=True):
        """
        초기화 메서드
        - tokenizer, tagging model, embedding model, 설정 값 등을 로드함
        - 절 분리 수행 및 결과를 JSON 파일로 저장
        - 각 절에 대해 임베딩 수행 및 의미 강조 단어 추출
        """
        self.kiwi = Kiwi()
        self.filenames = filenames
        self.config = config
        self.config.save_batch = getattr(self.config, 'save_batch', 100)
        self.config.return_embed_max = getattr(self.config, 'return_embed_max', 200)
        self.config.important_words_ratio = getattr(self.config, 'important_words_ratio', 0.6)
        self.model = TaggingModel(self.config)
        self.model.load_state_dict(load_file(self.filenames.clause_model_pt))
        self.embedding_model = DebertaV2Model.from_pretrained(self.config.model, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.sentences = sentences
        self.cls_vectors = []
        option_map = {'all': ['E', 'E2', 'E3'], 'E2': ['E', 'E2'], 'E3': ['E', 'E3'], 'E': ['E']}
        self.elist = option_map.get(e_option, ['E'])
        self.threshold = Variables().confidence_avg * self.config.confidence_threshold if threshold else 0.0
        self.splited = self.split2Clause()

        # with open(self.filenames.splited_json, "w", encoding="utf-8-sig") as f:
        #     json.dump(self.splited, f, ensure_ascii=False, indent=2)

        self.embeds = self.clause_embedding(self.splited)

    def split2Clause(self):
        """
        문장을 tagging 모델을 통해 토큰 단위로 분류하고, 설정된 태그(E, E2, E3)에 따라 절 단위로 분할

        Returns:
            절 리스트 (str 단위 구문 분리된 결과)
        """
        if isinstance(self.sentences, str):
            _sentences = [self.sentences]
        else:
            _sentences = self.sentences

        results = []
        for sentence in tqdm(_sentences):
            predicted = prediction(self.model, self.tokenizer, sentence, LabelData().id2label, return_cls=True)
            self.cls_vectors.append(predicted[-1])  # [CLS] 벡터 저장
            predicted = predicted[0]  # 예측 결과만 추출

            clauses, clause, switch = [], [], False
            for i, (tok, label, confidence) in enumerate(predicted):
                if label in self.elist and confidence > self.threshold and not self.is_segm(tok, predicted[i][0]):
                    switch = True
                elif switch:
                    recovered = recover_wordpieces(clause)
                    if len(recovered.split()) < 2 and clauses:
                        clauses[-1] += ' ' + recovered.strip()
                    else:
                        clauses.append(recovered)
                    clause, switch = [], False
                clause.append(tok)
            if clause:
                clauses.append(recover_wordpieces(clause))
            results.append(clauses)

        return results if not isinstance(self.sentences, str) else results[0]

    def clause_embedding(self, splited):
        """
        절 단위로 DeBERTa [CLS] 임베딩을 추출하고, 각 토큰 임베딩과 cosine similarity를 계산해 중요한 단어 추출
        - 절의 [CLS] 임베딩은 temp에 저장
        - 각 토큰 벡터와 [CLS] 벡터 간 cosine 유사도를 계산하고
          가장 유사한 단어 상위 60%를 추출하여 highlight 대상 선정

        Returns:
            전체 절의 [CLS] 벡터 리스트 또는 None (저장 파일로 대체 가능)
        """
        # def save_batch_npy(batch_result, save_dir, batch_idx):
        #     os.makedirs(save_dir, exist_ok=True)
        #     path = os.path.join(save_dir, f'embedding_batch_{batch_idx}.npy')
        #     np.save(path, np.array(batch_result, dtype=object))

        # with open(self.filenames.significant_json, "w", encoding="utf-8") as f:
        #     pass  # 초기화

        all_result = [] if len(splited) < self.config.return_embed_max else None
        for batch_idx in range(0, len(splited), self.config.save_batch):
            batch = splited[batch_idx:batch_idx + self.config.save_batch]
            result, highlighted = [], []

            for ss in tqdm(batch, desc=f"Batch {batch_idx // self.config.save_batch}"):
                temp, highlight_temp = [], []
                for s in ss:
                    # 구문 단위 [CLS] 임베딩 추출
                    inputs = self.tokenizer(s, return_tensors='pt', add_special_tokens=True)
                    input_ids = inputs["input_ids"]
                    with torch.no_grad():
                        outputs = self.embedding_model(**inputs)
                    hidden_states, cls_vector = outputs.last_hidden_state, outputs.last_hidden_state[:, 0, :]
                    temp.append(cls_vector.squeeze(0).cpu().numpy())

                    # 단어 수준 의미 추출을 위한 사전처리
                    real = self.str2real(s, output_str=False)
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                    token_map = []
                    for idx, tok in enumerate(tokens):
                        if tok in self.tokenizer.all_special_tokens:
                            continue
                        clean_tok = tok[2:] if tok.startswith("##") else tok
                        for word in real:
                            if clean_tok in word:
                                token_map.append((word, idx))
                                break

                    # 각 단어별 토큰 인덱스 집계
                    word2indices = {}
                    for word, idx in token_map:
                        word2indices.setdefault(word, []).append(idx)

                    # 각 단어의 벡터들과 [CLS] 벡터 간 cosine similarity 평균 계산
                    word_scores = []
                    for word, indices in word2indices.items():
                        vecs = torch.stack([hidden_states[0, i] for i in indices])
                        sims = F.cosine_similarity(vecs, cls_vector[0].unsqueeze(0), dim=1)
                        score = self.rms(sims)
                        word_scores.append((word, float(score)))

                    # 유사도 기준 상위 60% 단어를 의미 강조 단어로 선정
                    word_scores_sorted = sorted(word_scores, key=lambda x: x[1], reverse=True)
                    top_n = max(1, int(len(word_scores_sorted) * self.config.important_words_ratio))
                    top_words = {word for word, _ in word_scores_sorted[:top_n]}
                    highlight_temp.append([word for word in real if word in top_words])
                highlighted.append(highlight_temp)
                result.append(temp)

            # for clauses, highlights in zip(batch, highlighted):
            #     item = {"clause": clauses, "highlight": highlights}
            #     with open(self.filenames.significant_json, "a", encoding="utf-8") as f:
            #         f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # save_batch_npy(result, self.filenames.saved_temp_dir, batch_idx)
            if all_result is not None:
                all_result.extend(result)


        # def load_and_merge_npy(save_dir: str, output_path: str):
        #     files = [f for f in os.listdir(save_dir) if f.startswith("embedding_batch_") and f.endswith(".npy")]
        #     if not files:
        #         raise FileNotFoundError("병합할 .npy 파일이 없습니다.")
        #     files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        #     if len(files) == 1:
        #         src = os.path.join(save_dir, files[0])
        #         dst = os.path.join(save_dir, output_path)
        #         np.save(dst, np.load(src, allow_pickle=True))
        #         return
        #     merged = []
        #     for file in files:
        #         batch_path = os.path.join(save_dir, file)
        #         data = np.load(batch_path, allow_pickle=True)
        #         merged.extend(data)
        #     np.save(output_path, np.array(merged, dtype=object))

        # load_and_merge_npy(self.filenames.saved_temp_dir, self.filenames.embedding_np)
        return all_result

    def is_gram(self, word):
        """주어진 단어가 조사(J), 어미(E), 접미사(XS)인지 여부 확인"""
        t = self.kiwi.tokenize(word)[-1].tag
        return t[0] in ['J', 'E'] or t[:2] == 'XS'

    def is_segm(self, word, prev):
        """두 토큰 결합 시 의미 단위(N/V/M/XR 등)로 분리될 수 있는지 판단"""
        combined = prev + word.strip('#') if word.startswith('#') else prev + ' ' + word
        t = self.kiwi.tokenize(combined)[-1].tag
        return t[0] in ['N', 'V', 'M'] or t[:2] == 'XR'

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        """Root Mean Square 연산 (유사도 평균 계산 시 활용)"""
        return torch.sqrt(torch.mean(x ** 2))

    def str2real(self, text, timecat=True, output_str=True):
        """
        텍스트를 형태소 분석하여 의미 요소만 추출
        timecat=True일 경우 시간 관련 숫자 묶음도 처리함
        """
        tokens = self.kiwi.tokenize(text)
        return ' '.join(self.bereal(tokens, timecat)) if output_str else self.bereal(tokens, timecat)

    def bereal(self, tokens, timecat=True):
        """
        형태소 리스트에서 의미 있는 형태소만 필터링
        - 명사, 동사, 숫자, 관형사 등 take 리스트 중심
        - 시간 정보는 붙여서 하나의 문자열로 처리
        """
        real, timeset = [], []
        take = ['NNG', 'NNP', 'NNB', 'NP', 'NR', 'XR', 'SN', 'SL', 'VV', 'VA', 'MM', 'MAJ', 'MAG']
        timeTrigger = ['년', '월', '일', '시', '분', '초', '세']
        for token in tokens:
            if token.tag in take:
                if not timecat:
                    real.append(token.form)
                    continue
                if token.tag in ['SN', 'NR']:
                    timeset.append(token.form)
                elif token.form in timeTrigger or token.tag == 'NNB':
                    if timeset:
                        timeset.append(token.form)
                elif len(timeset) > 1:
                    real.append(''.join(timeset))
                    timeset = []
                elif timeset:
                    real.append(timeset[0])
                    timeset = []
                real.append(token.form)
        return real

def split_sentences(sentences):
    config = Config()
    config.confidence_threshold = 0.15

    cs = ClauseSpliting(sentences, config= config, e_option= 'E3', threshold= True)

    return cs