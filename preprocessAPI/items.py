from pydantic import BaseModel, Field

class EconomicNewsItem(BaseModel):
    """
    EconomicNewsItem 모델은 경제 관련 뉴스 항목의 정보를 구조화된 형태로 표현합니다.

    Attributes:
        yt_url (str, optional): 해당 뉴스 항목의 YouTube 영상 URL입니다. 기본값은 None입니다.
    """
    yt_url: str = None