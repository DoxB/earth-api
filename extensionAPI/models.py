from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# 경제용어사전
class EconomicKeyword(Base):
    __tablename__ = "economic"

    item_id = Column('id', Integer, primary_key=True)
    keyword = Column(Text, nullable=False)
    explanation = Column(Text, nullable=True)
    details = Column(Text, nullable=True)