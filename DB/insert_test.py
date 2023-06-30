import pymysql
import pandas as pd
from .insert_select_db import Database
from .insert_data_processor_ver1 import DataProcessor

# config.txt에서 설정값을 읽기
with open("DB/config.txt", "r") as file:
    exec(file.read())

# Database 객체를 생성
db = Database(configs)

# DataProcessor 객체를 생성
processor = DataProcessor(db)

# companies_df = pd.DataFrame({
#     'company_name': ['삼성전자','현대차','삼성생명','셀트리온','포스코']
# })
# # 회사 데이터 삽입
# db.insert_data('company', companies_df, ['company_name'])

# processor.process_and_insert_stock_data('stock_data(5y).csv', ['company_id', 'company_name', 'date', 'code','open', 'high', 'low', 'close', 'volume', 'stock_change', 
#                                                           'ma_5', 'fast_k', 'slow_k', 'slow_d', 'rsi', 'std', 'upper', 'lower'])
                                                          
# csv_files = ['cell.csv', 'hyundai.csv', 'samsungelec_2018_2019.csv', 'samsungelec_2020.csv', 'samsungelec_2021_2023.csv', 'posco.csv']
# for csv_file in csv_files:
#     processor.process_and_insert_news_data(csv_file, ['company_id','company_name','title','url','date'])
    
# processor.process_and_insert_report_data('api_result_with_gpa.csv',['company_id', 'company_name','dctl', 'ncl', 'nci', 'dta', 'ca', 'aia', 'oci', 'cl', 'cs', 'ata', 'cce', 'inv', 
#                                                                     'cogs', 'tota', 'nca', 'ia', 'ip', 'tnga', 'ir', 'gp', 'tl', 'fi', 'date', 'quarter', 'fq', 'os', 'gp_a'])

# processor.process_and_insert_stock_prediction_data('final_data.csv',['company_id', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 'ma_5', 'std', 'upper',
#                                                                                 'lower', 'obv', 'ma', 'cci', 'fast_k', 'fast_d', 'roc', 'rsi', 'mfi', 'dp', 'ks_roc', 'ks_dp', 'ks_fast', 'score'])


csv_files = ['/home/ubuntu/workspace/csv/fcell_scoring.csv', '/home/ubuntu/workspace/csv/fhyundai_scoring.csv', '/home/ubuntu/workspace/csv/fposco_scoring.csv', '/home/ubuntu/workspace/csv/fsamsungelec_2018_2019_scoring.csv', '/home/ubuntu/workspace/csv/fsamsungelec_2020_scoring.csv', '/home/ubuntu/workspace/csv/fsamsungelec_2021_2023_scoring.csv', '/home/ubuntu/workspace/csv/fsamsunglife_scoring.csv']
for csv_file in csv_files:
    processor.process_and_insert_news_analysis_data(csv_file,['company_id','company_name','title','url','date','sentiment','score'])
    