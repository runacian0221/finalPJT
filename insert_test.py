import pymysql
import pandas as pd
from insert_select_db import Database
from insert_def import 

# config.txt에서 설정값을 읽기
with open("config.txt", "r") as file:
    exec(file.read())

# Database 객체를 생성
db = Database(configs)

process_and_insert_stock_data('stock_data.csv', ['company_id','company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 
                                                          'ma_5', 'fast_k', 'slow_k', 'slow_d', 'rsi', 'std', 'upper', 'lower'])

                                                          
csv_files = ['cell.csv', 'hyundai.csv', 'samsungelec_2018_2019.csv', 'samsungelec_2020.csv', 'samsungelec_2021_2023.csv', 'posco.csv']

for csv_file in csv_files:
    process_and_insert_news_data(csv_file, ['company_id','company_name','title','content','url','writed_at'])