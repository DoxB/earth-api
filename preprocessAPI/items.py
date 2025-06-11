from pydantic import BaseModel, Field

class EconomicNewsItem(BaseModel):
    yt_url: str = None