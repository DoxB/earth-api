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
    def __init__(self):
        DB_URL = f'mysql+pymysql://{DB_ID}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        self.engine = create_engine(DB_URL, pool_recycle = 500)

    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def connection(self):
        conn = self.engine.connect()
        return conn