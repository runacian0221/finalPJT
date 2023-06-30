import pymysql
import pandas as pd
from .insert_select_db import Database
from .realtime_insert_data_processor import RealTimeDataProcessor

# config.txt에서 설정값을 읽기
with open("DB/config.txt", "r") as file:
    exec(file.read())

# Database 객체를 생성
db = Database(configs)

# DataProcessor 객체를 생성
processor = RealTimeDataProcessor(db)

processor.process_and_insert_stock_data(stock_df, 
    required_columns=['company_id', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 
                      'ma_5', 'fast_k', 'slow_k', 'slow_d', 'rsi', 'std', 'upper', 'lower'], 
    table_name='stock')