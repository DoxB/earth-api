from fastapi import FastAPI
from database import *
from models import *
from items import *
from functions import *

app = FastAPI()

dictionary_engine = DictionaryEngineconn()

@app.post("/economic_info")
def get_extension_info(request: EconomicNewsItem):
    dictionary_session = dictionary_engine.sessionmaker()
    try:
        ### 유튜브 스크립트 스크랩 및 전처리
        yt_url = request.yt_url
        transcript = YoutubeScrape.get_video_text(yt_url)
        pre_transcript = YoutubeScrape.preprocessing(transcript)
        ### 용어 탐지
        keywords = YoutubeScrape.extract_keywords(pre_transcript)

        # 포함
        # keyword_infos = dictionary_session.query(EconomicKeyword).filter(
        #     or_(*[EconomicKeyword.keyword.like(f'%{kw}%') for kw in keywords])
        # ).all()

        # 완전일치
        keyword_infos = dictionary_session.query(EconomicKeyword).filter(
            func.lower(EconomicKeyword.keyword).in_([kw.lower() for kw in keywords])
        ).all()
        ### 내용 요약
        summary_result = YoutubeScrape.summary_text(pre_transcript)

        return {"keywords": keyword_infos, "summary": summary_result}

    finally:
        dictionary_session.close()