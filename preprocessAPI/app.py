from fastapi import FastAPI
from items import *
from functions import *

app = FastAPI()

@app.post("/script_preprocess")
def get_extension_info(request: EconomicNewsItem):
    """
    YouTube 뉴스 영상에서 자막을 수집하고, 전처리 및 인과 관계 문장 분석을 수행한 후
    Neo4j 그래프 데이터베이스에 결과를 저장합니다.

    주요 기능:
    1. 유튜브 자막 수집 및 전처리
    2. 핵심 키워드 기반 이벤트 토픽 추출
    3. 인과 문장 탐지 및 문장 재구성
    4. 문장 간 관계 도출 및 Neo4j 저장

    Args:
        request (EconomicNewsItem): YouTube URL을 담은 요청 모델

    Returns:
        dict: 추출된 대표 이벤트 토픽을 반환
    """
    # 1. 유튜브 스크립트 스크랩 및 전처리
    yt_url = request.yt_url
    transcript = YoutubeScrape.get_video_text(yt_url)
    pre_transcript = YoutubeScrape.preprocessing(transcript)

    # 2. 핵심 키워드 추출 및 이벤트 토픽 선택
    keywords = YoutubeScrape.extract_keywords(pre_transcript)
    event_topics = TopicSelect.select_topic(keywords)

    # 3. 인과/일반 문장 분류
    causal_sentences, general_sentences = CausalClassify.inference_sentence(pre_transcript)

    # 4. 인과 문장 및 일반 문장 재구성 및 임베딩
    causal_split_result, causal_emb_list = SplitSentence.result_split(causal_sentences)
    general_split_result, general_emb_list = SplitSentence.result_split(general_sentences)

    # 5. 문장 간 관계 도출
    causal_node, causal_rel = UpdataNeo4j.make_relation(causal_split_result, causal_emb_list)
    general_node, general_rel = UpdataNeo4j.make_relation(general_split_result, general_emb_list)

    # 6. 관계형 그래프 DB 업데이트
    UpdataNeo4j.update_neo4j(causal_node, causal_rel, event_topics, "causal")
    UpdataNeo4j.update_neo4j(general_node, general_rel, event_topics, "general")
    return {
                "result": event_topics
            }