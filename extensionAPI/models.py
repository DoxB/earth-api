from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# 경제용어사전
class EconomicKeyword(Base):
    """
    EconomicKeyword 테이블은 경제 용어 사전 정보를 저장합니다.

    Attributes:
        item_id (int): 고유 식별자 (기본키)
        keyword (str): 경제 용어 키워드 (예: '인플레이션')
        explanation (str): 간단한 요약 설명
        details (str): 상세 설명 또는 관련 예시
    """
    __tablename__ = "economic"

    item_id = Column('id', Integer, primary_key=True)
    keyword = Column(Text, nullable=False)
    explanation = Column(Text, nullable=True)
    details = Column(Text, nullable=True)