from fastapi import FastAPI
from items import *
from functions import *

app = FastAPI()

@app.post("/script_preprocess")
def get_extension_info(request: EconomicNewsItem):
    ### 유튜브 스크립트 스크랩 및 전처리
    yt_url = request.yt_url

    transcript = YoutubeScrape.get_video_text(yt_url)
    pre_transcript = YoutubeScrape.preprocessing(transcript)

    ### 기능1: 이벤트 토픽 식별
    keywords = YoutubeScrape.extract_keywords(pre_transcript)
    event_topics = TopicSelect.select_topic(keywords)

    ### 기능2: 인과 문장 탐지
    causal_sentences, general_sentences = CausalClassify.inference_sentence(pre_transcript)

    ### 기능3: 구문화
    causal_split_result, causal_emb_list = SplitSentence.result_split(causal_sentences)
    general_split_result, general_emb_list = SplitSentence.result_split(general_sentences)

    ### 기능4: neo4j 업데이트
    causal_node, causal_rel = UpdataNeo4j.make_relation(causal_split_result, causal_emb_list)
    general_node, general_rel = UpdataNeo4j.make_relation(general_split_result, general_emb_list)

    UpdataNeo4j.update_neo4j(causal_node, causal_rel, event_topics, "causal")
    d = UpdataNeo4j.update_neo4j(general_node, general_rel, event_topics, "causal")
    return {
                # "event_topics": event_topics,
                # "causal_sentences":causal_sentences,
                # "general_sentences":general_sentences,
                # "causal_split_result": causal_split_result,
                # "causal_emb_list": [len(causal_emb_list), len(causal_emb_list[0])],
                # "general_split_result": general_split_result,
                # "general_emb_list": [len(general_emb_list), len(causal_emb_list[0])],
                # # "causal_rel": causal_node,
                # "general_rel": causal_rel
                "result": event_topics
            }