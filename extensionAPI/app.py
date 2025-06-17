from fastapi import FastAPI
from database import *
from models import *
from items import *
from functions import *

app = FastAPI()

dictionary_engine = DictionaryEngineconn()

@app.post("/economic_info")
def get_extension_info(request: EconomicNewsItem):
    """
    경제 뉴스 영상으로부터 핵심 용어 및 요약 정보를 추출하는 API.

    처리 단계:
    1. YouTube 영상 자막 수집 및 텍스트 전처리
    2. 명사 키워드 추출
    3. 경제용어사전과 키워드 일치 여부 확인 (완전일치 기준)
    4. 전체 뉴스 텍스트 요약

    Args:
        request (EconomicNewsItem): yt_url 포함 요청 본문

    Returns:
        dict: {
            "keywords": List[Dict[str, str]],
            "summary": List[str]
        }
    """
    dictionary_session = dictionary_engine.sessionmaker()
    try:
        # 1. 유튜브 자막 수집 및 정제
        yt_url = request.yt_url
        transcript = YoutubeScrape.get_video_text(yt_url)
        pre_transcript = YoutubeScrape.preprocessing(transcript)

        # 2. 명사 키워드 추출
        keywords = YoutubeScrape.extract_keywords(pre_transcript)

        # 3. 경제용어사전에서 완전일치하는 용어 조회 (대소문자 무시)
        keyword_infos = dictionary_session.query(EconomicKeyword).filter(
            func.lower(EconomicKeyword.keyword).in_([kw.lower() for kw in keywords])
        ).all()

        # 4. 텍스트 요약
        summary_result = YoutubeScrape.summary_text(pre_transcript)

        # 5. 결과 반환 (SQLAlchemy 모델 객체 → dict 변환)
        keyword_dicts = [
            {
                "id": item.item_id,
                "keyword": item.keyword,
                "explanation": item.explanation,
                "details": item.details
            }
            for item in keyword_infos
        ]

        return {
            "keywords": keyword_dicts,
            "summary": summary_result
        }

    finally:
        dictionary_session.close()