# ===========================================================
# Token Classification 학습 파이프라인 (DeBERTa 기반)
# - 문장을 토큰 단위로 분류 (BIO 또는 절 경계 태그 등)
# - HuggingFace Transformers + Accelerate + PyTorch 기반
# ===========================================================

from transformers import DebertaV2ForTokenClassification, AutoTokenizer, DebertaV2Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
from dataclasses import dataclass, field

# ------------------------------
# Config 클래스: 학습 설정 저장
# ------------------------------
@dataclass
class Config:
    model: str = "models/kf-deberta-base"
    dropout: float = 0.5
    max_length: int = 128
    batch_size: int = 1
    epochs: int = 50
    lr: float = 3e-4
    enable_scheduler: bool = True
    scheduler: str = 'CosineAnnealingWarmRestarts'
    gradient_accumulation_steps: int = 2
    adam_eps: float = 1e-6
    freeze_encoder: bool = True
    tag_weight: list = field(default_factory=lambda: [0.1, 1.0, 1.2, 1.2])  # 클래스 불균형 보정
    confidence_threshold: float = 0.5

# 라벨 매핑 클래스
@dataclass
class LabelData:
    labels: list = field(default_factory=lambda: ["O", "E", "E2", "E3"])
    id2label: dict = field(init=False)
    label2id: dict = field(init=False)

    def __post_init__(self):
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

# 글로벌 변수 저장용
@dataclass
class Variables:
    confidence_avg : float = 1.0

# ------------------------------
# WordPiece 토큰 복구 함수
# ------------------------------
def recover_wordpieces(tokens: list) -> str:
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
    try:
        if words[-1] == '.':
            words[-2] += '.'
            words.pop(-1)
    except:
        pass
    return ' '.join(words)

# ------------------------------
# BIO 태깅 텍스트 파일 로드 함수
# ------------------------------
def open_file(file_name):
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        raw = f.read()
        result, sents, tags = [], [], []
        for r in raw.splitlines():
            r = r.strip()
            if len(r) > 0:
                rr = r.split()
                if len(rr) != 2:
                    print("잘못된 줄 형식 감지됨")
                sents.append(rr[0])
                tags.append(rr[1])
            else:
                result.append({'tokens': sents, 'labels': tags})
                sents, tags = [], []

    print("The number of data sentences:", len(result))

    for r in result:
        r['full_text'] = recover_wordpieces(r["tokens"])

    return pd.DataFrame(result[:166])  # 앞 166개만 사용

# ------------------------------
# 토큰 분류용 커스텀 Dataset 클래스
# ------------------------------
class TokenTaggingDataset:
    def __init__(self, df, config, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.config = config
        self.label2id = LabelData().label2id

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['full_text']
        tokens = row['tokens']
        labels = row['labels']

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors='pt'
        )
        iter_labels = iter(labels)
        label_ids = []
        for id in encoding['input_ids'].squeeze():
            if id < 6 or id >= 130000:  # 특수토큰 영역 제외 (kf-deberta)
                label_ids.append(-100)
            else:
                try:
                    label = next(iter_labels)
                    label_id = self.label2id[label]
                    label_ids.append(label_id)
                except StopIteration:
                    label_ids.append(-100)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return len(self.df)

# ------------------------------
# Mean Pooling 클래스 (사용 안 됨)
# ------------------------------
class MeanPooling(nn.Module):
    def forward(self, hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embed = torch.sum(hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return sum_embed / sum_mask

# ------------------------------
# 모델 클래스 정의
# ------------------------------
class TaggingModel(nn.Module):
    def __init__(self, config, num_classes=4):
        super().__init__()
        self.encoder = DebertaV2Model.from_pretrained(config.model, output_hidden_states=True, local_files_only=True)
        if config.freeze_encoder:
            for p in self.encoder.base_model.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, inputs, return_cls=False, out_last_hidden_state=False):
        out = self.encoder(**inputs, return_dict=True)
        sequence_output = self.dropout(out.last_hidden_state)
        logits = self.classifier(sequence_output)
        result = [logits]
        if return_cls:
            cls_vector = sequence_output[:, 0, :]
            result.append(cls_vector)
        if out_last_hidden_state:
            result.append(out.last_hidden_state)
        return result if any((return_cls, out_last_hidden_state)) else logits