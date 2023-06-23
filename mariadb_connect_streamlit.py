import streamlit as st
import pymysql
import pandas as pd
from datetime import datetime, timedelta

class MariaDB:
    def __init__(self, configs):
        self.connection = pymysql.connect(**configs)

    def execute_query(self, query):
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            column_names = [desc[0] for desc in cursor.description]
            result = cursor.fetchall()
        return result, column_names

with open("config.txt", "r") as file:
    exec(file.read())

db = MariaDB(configs)

# today = datetime.now().strftime("%Y-%m-%d")
# query = f"""
# select * FROM stock WHERE date BETWEEN '2018-01-01' AND '{today}'
# """

# # SQL 쿼리를 실행하여 데이터를 가져옵니다
# data, column_names = db.execute_query(query)

# # 데이터를 DataFrame으로 변환합니다
# df = pd.DataFrame(data, columns=column_names)

# # Streamlit에서 데이터를 출력합니다
# st.write(df)
