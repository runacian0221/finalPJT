import pymysql
import pandas as pd

class Database:
    # DB 연결   
    def __init__(self, configs) -> None:
        try:
            self.DB = pymysql.connect(**configs)
            print('데이터베이스 연결 성공')
        except pymysql.err.OperationalError as e:
            print('데이터베이스 연결 실패:',e)

    # DB 연결 해제
    def __del__(self) -> None:
        self.DB.close()
        print('데이터베이스 연결 해제')

    # 모든 'company' 테이블의 company_id 검색
    def get_company_ids(self):
        with self.DB.cursor() as cur:
            sql = f"SELECT company_id, company_name FROM company"
            cur.execute(sql)
            result = cur.fetchall()
            print({name: id for id, name in result})
            return {name: id for id, name in result}
    
    # 데이터에 company_id 추가하고 테이블에 삽입(수정)
    def add_company_id_and_insert(self, table_name, df, required_columns, company_name_column):
        # company_id 추가
        company_ids = self.get_company_ids()
        print(company_ids)
        df['company_id'] = df[company_name_column].map(company_ids)
        # 테이블에 삽입
        self.insert_data(table_name, df, required_columns)

    def insert_data(self, table_name, df, required_columns):
        assert not set(required_columns) - set(df.columns), str(set(required_columns) - set(df.columns))

        df.fillna('', inplace=True)

        insert_sql = f"INSERT INTO {table_name} (`{'`,`'.join(required_columns)}`) VALUES ({','.join(['%s']*len(required_columns))})"
        for i in range((len(df.values) // 10000)+1):
            start_idx = i * 10000
            data = [tuple(value) for value in df[required_columns].values[start_idx:start_idx+10000]]
            with self.DB.cursor() as cur:
                cur.executemany(insert_sql, data)
            self.DB.commit()

        print('입력 완료')

    def select_data(self, table_name=None, start_date=None, end_date=None, company_ids=None, file_name=None):
        where_sql = []

        if start_date and end_date:
            where_sql.append(f"date BETWEEN '{start_date}' AND '{end_date}'")
        elif start_date:
            where_sql.append(f"date >= '{start_date}'")
        elif end_date:
            where_sql.append(f"date <= '{end_date}'")
        
        if company_ids:
            company_ids =', '.join(str(id) for id in company_ids)
            where_sql.append(f"company_id IN ({company_ids})")
        
        main_query = f"SELECT * from {table_name} "

        if where_sql:
            main_query += f"WHERE {' AND '.join(where_sql)}"

        pagination_sql = ' LIMIT 100000 OFFSET {}'
        offset = 0
        final_result = []
        while True:
            with self.DB.cursor(pymysql.cursors.DictCursor) as cur:  # DictCursor를 사용
                cur.execute(main_query + pagination_sql.format(offset))
                result = cur.fetchall()
                final_result.extend(result)

            if len(result) < 100000:
                break

            offset += 100000 # LIMIT
        print(final_result[:10]) # 10개만 보여줌
        
        df = pd.DataFrame(final_result)
        df.to_csv(f'{file_name}.csv', index=False, encoding='UTF-8')
        print(df.head(10))
        return final_result

