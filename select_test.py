import pymysql
import pandas as pd
from insert_select_db import Database

# config.txt에서 설정값을 읽기
with open("config.txt", "r") as file:
    exec(file.read())

# Database 객체를 생성
db = Database(configs)
db.select_data(table_name='news', start_date='2022-02-01', end_date='2022-02-03', company_ids=[1], file_name='포스코주가데이터2')