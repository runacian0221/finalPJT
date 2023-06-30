import pymysql
import pandas as pd
from .insert_select_db_ver1 import Database

class DataProcessor:
    def __init__(self, db):
        self.db = db

    # 중복 코드를 처리하기 위한 새로운 함수
    def process_and_insert_data(self, df, column_rename_map, csv_file, required_columns, table_name):
        df = self.remove_null_values(df, csv_file)
        df.rename(columns=column_rename_map, inplace=True)
        #df = self.replace_company_name(df)
        print(df.columns)
        print(df['company_name'].unique())
        df.columns = [column.lower() for column in df.columns]
        self.db.add_company_id_and_insert(table_name, df, required_columns)

    def remove_null_values(self, df, csv_file):
        print(f'결측값 처리전 {csv_file}:')
        print(df.isnull().sum())
        df.dropna(axis=0, inplace=True)
        print(f'결측값 처리후 {csv_file}:')
        print(df.isnull().sum())
        return df

    def process_and_insert_stock_data(self, csv_file, required_columns, table_name='stock1'):
        stock_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
        self.process_and_insert_data(stock_df, {'Name':'company_name', 'Change':'stock_change', 
                                'slow_%K':'slow_k', 'slow_%D':'slow_d' , 'fast_%K':'fast_k'}, 
                                csv_file, required_columns, table_name)

    def process_and_insert_news_data(self, csv_file, required_columns, table_name='news'):
        news_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
        self.process_and_insert_data(news_df, {'keyword':'company_name','writed_at':'date'}, 
                                csv_file, required_columns, table_name)

    # date=year
    def process_and_insert_report_data(self, csv_file, required_columns, table_name='report'):
        report_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
        report_df.fillna(0, inplace=True)
        report_df.drop(['기준일자','종목코드','분기코드','PSR','주가코드'], axis=1, inplace=True)
        new_column_names = ['company_name','dctl', 'ncl', 'nci', 'dta', 'ca', 'aia', 'oci', 'cl', 'cs', 'ata', 'cce', 'inv', 
                            'cogs', 'tota', 'nca', 'ia', 'ip', 'tnga', 'ir', 'gp', 'tl', 'fi', 'date', 'quarter', 'fq', 'os', 'gp_a']
        column_name_mapping = {old: new for old, new in zip(report_df.columns, new_column_names)}
        self.process_and_insert_data(report_df, column_name_mapping, csv_file, required_columns, table_name)   

    def process_and_insert_stock_prediction_data(self, csv_file, required_columns, table_name='stock_prediction'):
        stock_prediction_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
        self.process_and_insert_data(stock_prediction_df, {'Change':'stock_change'}, csv_file, required_columns, table_name)    

    def process_and_insert_news_analysis_data(self, csv_file, required_columns, table_name='news_analysis1'):
        global news_df
        news_df = pd.read_csv(csv_file, encoding='UTF-8')
        news_df.fillna(0, inplace=True)
        self.process_and_insert_data(news_df, {'keyword':'company_name','writed_at':'date'}, 
                                     csv_file, required_columns, table_name)
