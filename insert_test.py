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

companies_df = pd.DataFrame({
    'company_name': ['삼성전자','현대차','삼성생명','셀트리온','포스코']
})
# 회사 데이터 삽입
db.insert_data('company', companies_df, ['company_name'])

processor.process_and_insert_stock_data('stock_data.csv', ['company_id', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 
                                                          'ma_5', 'fast_k', 'slow_k', 'slow_d', 'rsi', 'std', 'upper', 'lower'])
                                                          
csv_files = ['cell.csv', 'hyundai.csv', 'samsungelec_2018_2019.csv', 'samsungelec_2020.csv', 'samsungelec_2021_2023.csv', 'posco.csv']
for csv_file in csv_files:
    processor.process_and_insert_news_data(csv_file, ['company_id','company_name','title','content','url','date'])
    
processor.process_and_insert_report_data('api_final_result.csv',['company_id','company_name','date', 'quarter', 'dta', 'cogs', 'ca', 'gp', 'cce', 'tnga', 'cl', 'nca', 'inv', 'fi', 
                                                                'ncl', 'dctl', 'ip', 'cs', 'oci', 'tl', 'nci', 'tota', 'ata', 'aia', 'ia', 'ir'])