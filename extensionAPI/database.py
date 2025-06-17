import os
from dotenv import load_dotenv
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker

load_dotenv()

DB_ID = os.environ.get('DB_ID')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
DB_NAME = os.environ.get('DB_NAME')

### Dictionary DB 연결관리부
class DictionaryEngineconn:
    """
    MySQL 데이터베이스 연결 관리 클래스.

    이 클래스는 SQLAlchemy를 통해 DB 엔진을 생성하고,
    ORM 세션(sessionmaker) 또는 기본 커넥션(connection)을 제공합니다.

    Usage:
        db = DictionaryEngineconn()
        session = db.sessionmaker()  # ORM 쿼리용
        conn = db.connection()       # SQL 커넥션용

    Environment Variables (.env):
        DB_ID, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    """
    def __init__(self):
        DB_URL = f'mysql+pymysql://{DB_ID}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        self.engine = create_engine(DB_URL, pool_recycle = 500)

    def sessionmaker(self):
        """
        SQLAlchemy 세션 생성기

        Returns:
            Session: ORM 기반 세션 객체
        """
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def connection(self):
        """
        SQLAlchemy 기본 커넥션 객체 반환

        Returns:
            Connection: SQL 실행을 위한 DB 커넥션
        """
        conn = self.engine.connect()
        return conn