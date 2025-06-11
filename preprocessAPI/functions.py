import re
from kiwipiepy import Kiwi

from gensim.models import LdaModel
from gensim.test.utils import datapath
from gensim.corpora import Dictionary
import kss
from transformers import pipeline, AutoTokenizer

import yt_dlp
import requests

from split_module.predict import *
from sklearn.feature_extraction.text import TfidfVectorizer

from neo4j import GraphDatabase
import ast
import os
from dotenv import load_dotenv
import time

load_dotenv()

NEO4J_URL = os.environ.get('NEO4J_URL')
NEO4J_PORT = os.environ.get('NEO4J_PORT')
NEO4J_ID = os.environ.get('NEO4J_ID')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

class CustomTokenizer:
    def __init__(self):
        self.tagger = Kiwi()

    def __call__(self, sent):
        morphs = self.tagger.analyze(sent)[0][0]  # 첫 번째 분석 결과 사용, normalize=True로 정규화
        result = [form for form, tag, _, _ in morphs if tag in ['NNG', 'NNP'] and len(form) > 1]
        return result

class YoutubeScrape:
    # yt-dlp 방식
    def get_video_text(video_url):
        video_id = video_url.split('v=')[1][:11]

        ydl_opts = {
            'skip_download': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['ko'],
            'outtmpl': '%(id)s.%(ext)s'
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.cache.remove()
            info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
            vtt_url = info.get('requested_subtitles')['ko']['url']
            subtitle = requests.get(vtt_url).text

            lines = []
            for line in subtitle.split('\n'):
                if line in lines:
                    continue
                if (
                    line.strip() == ''
                    or line.startswith('WEBVTT')
                    or line.startswith('Kind:')
                    or line.startswith('Language:')
                    or re.match(r'\d\d:\d\d:\d\d\.\d+ -->', line)
                ):
                    continue
                if re.match(r'\[.*\]', line.strip()):
                    continue
                clean = re.sub(r'<.*?>', '', line).strip()
                if clean:
                    lines.append(clean)

            text_formatted = ' '.join(lines)

        return text_formatted

    def preprocessing(text):
        text = re.sub('\n', ' ', text)
        sentences = [s for s in kss.split_sentences(text)]
        ### 중요도 낮은 문장 제거
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)

        tfidf_sums = X.sum(axis=1)    # 문장별 TF-IDF 합계 (2D 행렬)
        tfidf_sums = np.array(tfidf_sums).flatten()

        threshold = np.percentile(tfidf_sums, 30)

        filtered_sentences = [
            sent.replace('안녕하세요', '').replace('[음악]', '').replace(' 네', '').replace('네 ', '').strip() for sent, score in zip(sentences, tfidf_sums) if score > threshold
        ]
        pre_text = ' '.join(filtered_sentences)

        ### 오타 교정
        kiwi = Kiwi(model_type='sbg', typos='basic_with_continual_and_lengthening')

        pattern = r'(\(.+?\))'
        pre_text = re.sub(pattern, '',pre_text)
        pattern = r'(\[.+?\])'
        pre_text = re.sub(pattern, '',pre_text)
        pattern = r'[\\/:*?"<>|.]'
        pre_text = re.sub(pattern, '',pre_text)
        pattern = r"[^\sa-zA-Z0-9ㄱ-ㅎ가-힣!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~)※‘’·“”'͏'㈜ⓒ™©•]"
        pre_text = re.sub(pattern, '',pre_text).strip()

        tokens = kiwi.tokenize(pre_text)

        pre_text = kiwi.join(tokens)

        return pre_text

    def extract_keywords(text):
        tokenizer = CustomTokenizer()
        tokens = tokenizer(text)
        return tokens

class TopicSelect:
    def select_topic(script):
        except_topic_id = {11, 12, 18, 23, 28}

        common_dictionary = Dictionary.load("/home/regular/workspace/Earth/earth-api/preprocessAPI/models/topic/the_2293.id2word")
        bow = common_dictionary.doc2bow(script)
        temp_file = datapath("/home/regular/workspace/Earth/earth-api/preprocessAPI/models/topic/the_2293")
        lda = LdaModel.load(temp_file)
        # 토픽 리스트 (확인용)
        topicList = lda.print_topics(num_words=5, num_topics=30)

        # LDA 모델로 토픽 분포 예측
        topic_vector = lda.get_document_topics(bow)

        # 확률 높은 순으로 정렬 후 top-N만 추출
        sorted_topics = sorted(topic_vector, key=lambda x: x[1], reverse=True)
        top_topics = sorted_topics[:6]

        for topic_id, _ in top_topics:
            if topic_id in except_topic_id:
                continue
            else:
                # 정규 표현식을 사용해 키워드만 추출 (따옴표 안의 문자열)
                keywords = re.findall(r'"([^"]*)"', topicList[topic_id][1])
                break
        return keywords[:5] # 상위 5개 키워드만 반환

class CausalClassify:
    def inference_sentence(script):
        # 스크립트 전처리
        script = re.sub('\n', ' ', script)
        sentences = [s for s in kss.split_sentences(script)]

        # 모델 파이프라인 설정
        base_model_path = "./models/kf-deberta-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
        classify_model_path = './models/causal_detection'
        clf = pipeline(
            "text-classification",
            model=classify_model_path,
            tokenizer=tokenizer,
            device=0  # GPU 사용 시 주석 해제
        )

        # 인과 탐지
        causal_sentences = []
        general_sentencse = []
        for sen in sentences:
            output = clf(sen)[0]
            if output["label"] == "LABEL_1":
                causal_sentences.append(sen)
            else:
                general_sentencse.append(sen)

        return causal_sentences, general_sentencse

class SplitSentence:
    def result_split(sentences):
        cs = split_sentences(sentences)
        result = cs.splited
        embeds = cs.embeds
        emb_list = []
        for emb in embeds:
            emb_list.append([e.tolist() for e in emb])
        return result, emb_list

class UpdataNeo4j:
    def make_relation(split_result, emb_list):
        node_result = []
        rel_result = []
        for t_idx in range(len(split_result)):
            if len(split_result[t_idx]) == 1:
                node_result.append([split_result[t_idx][0], emb_list[t_idx][0]])
            else:
                for e_idx in range(len(split_result[t_idx])-1):
                    node_result.append([split_result[t_idx][e_idx], emb_list[t_idx][e_idx]])
                    rel_result.append([(split_result[t_idx][e_idx]), (split_result[t_idx][e_idx+1])])
                    if e_idx == len(split_result[t_idx])-2:
                        node_result.append([split_result[t_idx][e_idx+1], emb_list[t_idx][e_idx+1]])
        return node_result, rel_result

    def update_neo4j(nodes, relations, event_topics, rel_type):
        url = f"{NEO4J_URL}:{NEO4J_PORT}"
        auth = (NEO4J_ID, NEO4J_PASSWORD)

        current_timestamp = int(time.time() * 1000)
        node_list = []
        for node in nodes:
            node_info = {
                'name': node[0],
                'embedding': node[1],
                'createdTimestamp': current_timestamp,
                'oriTopic': event_topics
            }
            node_list.append(node_info)

        with GraphDatabase.driver(url, auth=auth) as driver:
            driver.verify_connectivity()
            with driver.session(database="neo4j") as session:
                # 노드 생성
                gen_youtube_node_q = '''
                    UNWIND $node_list AS row
                    CREATE (y:Youtube {
                        name: row.name,
                        embedding: row.embedding,
                        createdTimestamp: row.createdTimestamp,
                        oriTopic: row.oriTopic
                    })
                '''
                session.run(gen_youtube_node_q, node_list=node_list)

                # 관계 생성
                if rel_type == "causal":
                    rel_type_cypher = "isCauseOf"
                else:
                    rel_type_cypher = "isGeneralOf"

                gen_youtube_rel_q = f'''
                    UNWIND $relations AS pair
                    MATCH (a:Youtube {{name: pair[0]}})
                    MATCH (b:Youtube {{name: pair[1]}})
                    CREATE (a)-[:{rel_type_cypher}]->(b)
                '''
                session.run(gen_youtube_rel_q, relations=relations)

                # 이벤트 연결 (event_topics가 리스트/문자열 형태에 따라 달라질 수 있습니다)
                # event_topics가 리스트면, 각 토픽별로 연결하거나 IN 구문 사용
                # 여기서는 간단하게 문자열(단일 토픽)로 가정
                for node in nodes:
                    topic_connect_q = '''
                        MATCH (b:Youtube {name: $name})
                        MATCH (a:Event {topics: $event_topics})
                        CREATE (b)-[:connectedYoutube]->(a)
                    '''
                    session.run(topic_connect_q, name=node[0], event_topics=event_topics)

        return "ok"
