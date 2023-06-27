import pymysql
import pandas as pd
from insert_select_db import Database

class RealTimeDataProcessor:
    def __init__(self, db):
        self.db = db

    # 중복 코드를 처리하기 위한 새로운 함수
    def process_and_insert_data(self, df, column_rename_map, required_columns, table_name):
        df = self.remove_null_values(df)
        df.rename(columns=column_rename_map, inplace=True)
        df.columns = [column.lower() for column in df.columns]
        self.db.add_company_id_and_insert(table_name, df, required_columns)

    def remove_null_values(self, df):
        print(f'결측값 처리 전 DataFrame:')
        print(df.isnull().sum())
        df.dropna(axis=0, inplace=True)
        print(f'결측값 처리 후 DataFrame:')
        print(df.isnull().sum())
        return df


    def process_and_insert_stock_data(self, stock_df, required_columns, table_name='stock'):
        self.process_and_insert_data(stock_df, {'Name':'company_name', 'Change':'stock_change', 
                                'slow_%K':'slow_k', 'slow_%D':'slow_d' , 'fast_%K':'fast_k'}, 
                                None, required_columns, table_name)

    def process_and_insert_news_data(self, news_df, required_columns, table_name='news'):
        self.process_and_insert_data(news_df, {'keyword':'company_name','writed_at':'date'}, 
                                None, required_columns, table_name)

    def process_and_insert_report_data(self, report_df, required_columns, table_name='report'):
        report_df.fillna(0, inplace=True)
        report_df.drop(['기준일자','종목코드','분기코드','PSR','주가코드'], axis=1, inplace=True)
        new_column_names = ['company_name','dctl', 'ncl', 'nci', 'dta', 'ca', 'aia', 'oci', 'cl', 'cs', 'ata', 'cce', 'inv', 
                            'cogs', 'tota', 'nca', 'ia', 'ip', 'tnga', 'ir', 'gp', 'tl', 'fi', 'date', 'quarter', 'fq', 'os', 'gp_a']
        column_name_mapping = {old: new for old, new in zip(report_df.columns, new_column_names)}
        self.process_and_insert_data(report_df, column_name_mapping, None, required_columns, table_name)   

    def process_and_insert_stock_prediction_data(self, stock_prediction_df, required_columns, table_name='stock_prediction1'):
        self.process_and_insert_data(stock_prediction_df, {'Change':'stock_change'}, None, required_columns, table_name)    

    def process_and_insert_news_analysis_data(self, news_analysis_df, required_columns, table_name='news_analysis'):
        self.process_and_insert_data(news_analysis_df, {'keyword':'company_name','writed_at':'date'}, 
                                None, required_columns, table_name)

processor.process_and_insert_stock_data(stock_df, 
    required_columns=['company_id', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 
                      'ma_5', 'fast_k', 'slow_k', 'slow_d', 'rsi', 'std', 'upper', 'lower'], 
    table_name='stock')

processor.process_and_insert_news_data(news_df,
    required_columns= ['company_id','company_name','title','content','url','date'],
    table_name='news')

processor.process_and_insert_report_data(report_df,
    required_columns = ['company_id', 'company_name','dctl', 'ncl', 'nci', 'dta', 'ca', 'aia', 'oci', 'cl', 'cs', 'ata', 'cce', 'inv', 
                        'cogs', 'tota', 'nca', 'ia', 'ip', 'tnga', 'ir', 'gp', 'tl', 'fi', 'date', 'quarter', 'fq', 'os', 'gp_a'],
    table_name='report')

processor.process_and_insert_stock_prediction_data(stock_prediction_df,
    required_columns = ['company_id', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 'ma', 'std', 'upper',
                        'lower', 'obv', 'cci', 'fast_k', 'fast_d', 'roc', 'rsi', 'rsi_dp','mfi', 'dp', 'ks_roc', 'ks_dp', 'ks_fast', 'score'],
    table_name='stock_prediction1')

processor.process_and_insert_news_analysis_data(news_analysis_df,
    required_columns = ['company_id','company_name','title','content','url','date','sentiment','score'],
    table_name='news_analysis')
    