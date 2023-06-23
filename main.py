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

processor.process_and_insert_stock_data(f'stock_data{today}.csv', ['company_id', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 
                                                                  'ma_5', 'fast_k', 'slow_k', 'slow_d', 'rsi', 'std', 'upper', 'lower'])

processor.process_and_insert_report_data(f'api_final_result{today}.csv',['company_id','company_name','date', 'quarter', 'dta', 'cogs', 'ca', 'gp', 'cce', 'tnga', 'cl', 'nca', 
                                                                         'inv', 'fi', 'ncl', 'dctl', 'ip', 'cs', 'oci', 'tl', 'nci', 'tota', 'ata', 'aia', 'ia', 'ir'])

processor.process_and_insert_stock_prediction_data(f'stock_data_scoring_rf{today}.csv',['company_id', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 'ma_5', 'std', 'upper',
                                                                                        'lower', 'obv', 'ma', 'cci', 'fast_k', 'fast_d', 'roc', 'rsi', 'mfi', 'ma_10', 'ks_roc', 'ks_fast', 'score'])

processor.db.select_data(table_name='stock_prediction', start_date=yesterday, end_date=today, company_ids=[1], file_name=f'삼성전자stock_prediction{yesterday}-{today}')
