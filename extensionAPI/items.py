from pydantic import BaseModel, Field

# 경제용어사전
class EconomicNewsItem(BaseModel):
    """
    EconomicNewsItem 모델은 경제 관련 YouTube 뉴스의 URL을 입력받기 위한 요청 데이터 구조입니다.

    Attributes:
        yt_url (str): 분석 대상 YouTube 영상의 URL. 예: "https://www.youtube.com/watch?v=abcd1234"
    """
    yt_url: str = None