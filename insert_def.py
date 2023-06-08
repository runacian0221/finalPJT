import pymysql
import pandas as pd
from insert_select_db import Database

# config.txt에서 설정값을 읽기
with open("config.txt", "r") as file:
    exec(file.read())

# Database 객체를 생성
db = Database(configs)

def remove_null_values(df):
    print(f'결측값 처리전 {csv_file}:')
    print(df.isnull().sum())
    df.dropna(axis=0, inplace=True)
    print(f'결측값 처리전 {csv_file}:')
    print(df.isnull().sum())
    return df

def replace_company_name(df):
    df['company_name'].replace({'삼성전자' : 'Samsung Electronics',
                                '현대차': 'Hyundai Motor Company',
                                '삼성생명': 'Samsung life Insurance',
                                '셀트리온': 'Celltrion',
                                '포스코' : 'POSCO'},
                               inplace=True)
    return df

# company_data = {
#     'company_name': ['POSCO', 'Samsung Electronics', 'Hyundai Motor Company', 'Celltrion', 'Samsung life Insurance']
# }
# company_df = pd.DataFrame(company_data)
# table_name = 'company'
# required_columns = ['company_name']
# db.insert_data(table_name, company_df, required_columns)
#def insert_company_data(table_name=company, company_df, required_columns)


def process_and_insert_stock_data(csv_file, required_columns, table_name='stock'):
    # csv파일 경로 설정
    stock_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
    # null 결측값 제거(dropna)
    stock_df = remove_null_values(stock_df)
    # 컬럼명 수정
    stock_df.rename(columns={'Name':'company_name', 'Change':'stock_change', 
                             'slow_%K':'slow_k', 'slow_%D':'slow_d' , 'fast_%K':'fast_k'},inplace=True)
    # company_name을 영어로 수정(streamlit에서 한글깨짐 현상으로 인해 영어로 통일)
    stock_df = replace_company_name(stock_df)
    # 컬럼명 소문자로 변경
    stock_df.columns = [column.lower() for column in stock_df.columns]
    # company_name을 기준으로 company_id를 매핑하고 데이터 삽입 
    db.add_company_id_and_insert(table_name, stock_df, required_columns)


def process_and_insert_news_data(csv_file, required_columns, table_name='news'):
    news_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
    news_df = remove_null_values(news_df)
    news_df.rename(columns={'keyword':'company_name','writed_at':'date'}, inplace=True)
    news_df = replace_company_name(news_df)
    
    news_df.columns = [column.lower() for column in news_df.columns]
    db.add_company_id_and_insert(table_name, news_df, required_columns)
    

# def process_and_insert_report_data(csv_file, required_columns, table_name=report):
#     report_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
#     report_df = remove_null_values(report_df)
#     #report_df.rename(columns={}, inplace=True)
#     report_df = replace_company_name(report_df)
    
#     report_df.columns = [column.lower() for column in report_df.columns]
#     db.add_company_id_and_insert(table_name, report_df, required_columns)

# def process_and_insert_prediction_data(csv_file, required_columns, table_name=report):
#     prediction_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
#     prediction_df = remove_null_values(prediction_df)
#     #prediction_df.rename(columns={}, inplace=True)
#     prediction_df = replace_company_name(prediction_df)
    
#     prediction_df.columns = [column.lower() for column in prediction_df.columns]
#     db.add_company_id_and_insert(table_name, prediction_df, required_columns)