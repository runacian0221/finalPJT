import pymysql
import pandas as pd
from insert_select_db import Database
from insert_data_processor_ver1 import DataProcessor
from datetime import datetime, timedelta

# config.txt에서 설정값을 읽기
with open("config.txt", "r") as file:
    exec(file.read())

# Database 객체를 생성
db = Database(configs)

# DataProcessor 객체를 생성
processor = DataProcessor(db)
today = datetime.now().strftime("%Y-%m-%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
today_year = datetime.now().strftime("%YYYY")

# company_ids 목록
# 1. 삼성전자, 2. 현대차, 3. 삼성생명, 4. 셀트리온, 5. 포스코
# 테이블 목록
# news, stock, report, stock_prediction
# 사용방법 processor.db.select_data(table_name='테이블 이름', start_date='yyyy-mm-dd', end_date = 'yyyy-mm-dd', company_ids=[1,2,3,4,5], file_name='저장하고 싶은 파일 이름')
# report(재무제표) 데이터는 start_date = 'yyyy', end_date = 'yyyy' 
processor.db.select_data(table_name='stock_prediction', start_date='2018-10-10', end_date='2018-10-11', company_ids=[1], file_name='삼성전자stock_prediction2018-2023')
processor.db.select_data(table_name='news', start_date='2018-10-10', end_date='2018-10-11', company_ids=[1], file_name='삼성전자news')
processor.db.select_data(table_name='stock', start_date='2018-10-10', end_date='2018-10-11', company_ids=[1], file_name='삼성전자stock')
#processor.db.select_data(table_name='stock_prediction', start_date='2018-01-02', end_date=yesterday, company_ids=[1], file_name=f'삼성전자stock_prediction2018-{yesterday}')
#삼성전자stock_prediction2023-06-20-2023-06-21
processor.db.select_data(table_name='report', start_date='2018', end_date=today_year, company_ids=[1], file_name='삼성전자report')
