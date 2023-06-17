import pymysql
import pandas as pd
from insert_select_db import Database
from insert_data_processor_ver1 import DataProcessor

# config.txt에서 설정값을 읽기
with open("config.txt", "r") as file:
    exec(file.read())

# Database 객체를 생성
db = Database(configs)

# DataProcessor 객체를 생성
processor = DataProcessor(db)

processor.db.select_data(table_name='stock_prediction', start_date='2018-10-10', end_date='2018-10-11', company_ids=[1], file_name='삼성전자stock_prediction2018-2023')