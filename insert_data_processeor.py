class DataProcessor:
    def __init__(self, db):
        self.db = db

    # 중복 코드를 처리하기 위한 새로운 함수
    def process_and_insert_data(self, df, column_rename_map, csv_file, required_columns, table_name):
        df = self.remove_null_values(df, csv_file)
        df.rename(columns=column_rename_map, inplace=True)
        df = self.replace_company_name(df)
        df.columns = [column.lower() for column in df.columns]
        self.db.add_company_id_and_insert(table_name, df, required_columns)

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
        report_df = self.remove_null_values(report_df, csv_file)
        report_df.drop(['Unnamed: 0','종목코드', '분기코드'], axis=1, inplace=True)
        new_column_names = ['company_name','date', 'quarter', 'dta', 'cogs', 'ca', 'gp', 'cce', 'tnga', 'cl', 'nca', 'inv', 'fi', 
                            'ncl', 'dctl', 'ip', 'cs', 'oci', 'tl', 'nci', 'tota', 'ata', 'aia', 'ia', 'ir']
        column_name_mapping = {old: new for old, new in zip(report_df.columns, new_column_names)}
        self.process_and_insert_data(report_df, column_name_mapping, csv_file, required_columns, table_name)

    def process_and_insert_prediction_data(self, csv_file, required_columns, table_name='prediction'):
        prediction_df = pd.read_csv('csv/' + csv_file, encoding='UTF-8')
        prediction_df = self.remove_null_values(prediction_df, csv_file)
        self.process_and_insert_data(prediction_df, {}, csv_file, required_columns, table_name)
