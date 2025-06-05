from pydantic import BaseModel, Field

# 경제용어사전
class EconomicNewsItem(BaseModel):
    yt_url: str = None