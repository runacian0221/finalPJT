import pymysql
import pandas as pd
from insert_select_db import Database

class DataProcessor:
    def __init__(self, db):
        self.db = db

    def remove_null_values(self, df, csv_file):
        print(f'결측값 처리전 {csv_file}:')
        print(df.isnull().sum())
        df.dropna(axis=0, inplace=True)
        print(f'결측값 처리전 {csv_file}:')
        print(df.isnull().sum())
        return df

    def replace_company_name(self, df):
        df['company_name'].replace({'삼성전자' : 'Samsung Electronics',
                                    '현대차': 'Hyundai Motor Company',
                                    '삼성생명': 'Samsung life Insurance',
                                    '셀트리온': 'Celltrion',
                                    '포스코' : 'POSCO'},
                                inplace=True)
        return df

    def process_and_insert_stock_data(self, csv_file, required_columns, table_name='stock'):
        stock_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
        stock_df = self.remove_null_values(stock_df, csv_file)
        stock_df.rename(columns={'Name':'company_name', 'Change':'stock_change', 
                                'slow_%K':'slow_k', 'slow_%D':'slow_d' , 'fast_%K':'fast_k'},inplace=True)
        stock_df = self.replace_company_name(stock_df)
        stock_df.columns = [column.lower() for column in stock_df.columns]
        self.db.add_company_id_and_insert(table_name, stock_df, required_columns)

    def process_and_insert_news_data(self, csv_file, required_columns, table_name='news'):
        news_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
        news_df = self.remove_null_values(news_df, csv_file)
        news_df.rename(columns={'keyword':'company_name','writed_at':'date'}, inplace=True)
        news_df = self.replace_company_name(news_df)
        news_df.columns = [column.lower() for column in news_df.columns]
        self.db.add_company_id_and_insert(table_name, news_df, required_columns)

# 재무제표 데이터는 한국어 컬럼명부터 수정해야함
# select도 다른 기준으로 잡아야함
    def process_and_insert_report_data(self, csv_file, required_columns, table_name='report'):
        report_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
        report_df = self.remove_null_values(report_df, csv_file)
        report_df.drop(['Unnamed: 0','종목코드', '분기코드'], axis=1, inplace=True)
        new_column_names = ['company_name','year', 'quarter', 'dta', 'cogs', 'ca', 'gp', 'cce', 'tnga', 'cl', 'nca', 'inv', 'fi', 
                    'ncl', 'dctl', 'ip', 'cs', 'oci', 'tl', 'nci', 'tota', 'ata', 'aia', 'ia', 'ir']

        column_name_mapping = {old: new for old, new in zip(report_df.columns, new_column_names)}
        report_df.rename(columns=column_name_mapping, inplace=True)
        print(report_df.columns) # 확인용: 실제 컬럼 이름이 변경되었는지 출력
        report_df = self.replace_company_name(report_df)
        report_df.columns = [column.lower() for column in report_df.columns]
        self.db.add_company_id_and_insert(table_name, report_df, required_columns)

    def process_and_insert_prediction_data(self, csv_file, required_columns, table_name='prediction'):
        prediction_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
        prediction_df = self.remove_null_values(prediction_df, csv_file)
        #prediction_df.rename(columns={'keyword':'company_name','writed_at':'date'}, inplace=True)
        prediction_df = self.replace_company_name(prediction_df)
        prediction_df.columns = [column.lower() for column in prediction_df.columns]
        self.db.add_company_id_and_insert(table_name, prediction_df, required_columns)
