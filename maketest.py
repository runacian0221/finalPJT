import pymysql

connection = None
try:
    connection = pymysql.connect(host='localhost',
                           port=3306,
                           user='root',
                           password='password',
                           database='finalpjt',
                           charset='utf8mb4')

    with connection.cursor() as cursor:
        sql = "SELECT VERSION()"
        cursor.execute(sql)
        result = cursor.fetchone()
    print("Database version : %s " % result)

except pymysql.err.OperationalError as e:
    print("Error while connecting to MySQL: {}".format(e))

# change -> stock_change : 예약어라 컬럼명 수정해야됨
# name -> company_name : 일관성
# keyword -> company_name : 일관성
tables = [

"""
CREATE TABLE company(
    company_id INT NOT NULL AUTO_INCREMENT,
    company_name VARCHAR(64),
    PRIMARY KEY(company_id)
);
""",

"""
CREATE TABLE news(
    news_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64),
    title text,
    content mediumtext,
    url VARCHAR(2048),
    date DATETIME,
    PRIMARY KEY(news_id),
    FOREIGN KEY (company_id) REFERENCES company (company_id)
);
""",

"""
CREATE TABLE stock(
    stock_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64),
    date DATETIME,
    open INT,
    high INT,
    low INT,
    close INT,
    volume INT,
    stock_change FLOAT,
    ma_5 FLOAT,
    fast_k FLOAT,
    slow_k FLOAT,
    slow_d FLOAT,
    rsi FLOAT,
    std FLOAT,
    upper FLOAT,
    lower FLOAT,
    PRIMARY KEY(stock_id),
    FOREIGN KEY (company_id) REFERENCES company (company_id)
);
""",

"""
CREATE TABLE report(
    report_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64),
    date INT,
    quarter VARCHAR(10),
    dta FLOAT,
    cogs FLOAT,
    ca FLOAT,
    gp FLOAT,
    cce FLOAT,
    tnga FLOAT,
    cl FLOAT,
    nca FLOAT,
    inv FLOAT,
    fi FLOAT,
    ncl FLOAT,
    dctl FLOAT,
    ip FLOAT,
    cs FLOAT,
    oci FLOAT,
    tl FLOAT,
    nci FLOAT,
    tota FLOAT,
    ata FLOAT,
    aia FLOAT,
    ia FLOAT,
    ir FLOAT,
    PRIMARY KEY(report_id),
    FOREIGN KEY (company_id) REFERENCES company (company_id)
);
""",
]

with connection.cursor() as cursor:
    for table_creation_sql in tables:
        cursor.execute(table_creation_sql)
    connection.commit()  # 모든 테이블 생성 쿼리를 실행한 후에 commit

def get_table_count(cursor):
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    return len(tables)

with connection.cursor() as cursor:
    count = get_table_count(cursor)
    print(f"The number of tables: {count}")


# 예측 결과 테이블은 기사, 재무제표, 주가데이터 테이블 3개?

# CREATE TABLE news_prediction(
#     company_id INT,
#     company_name VARCHAR(64),
#     date DATETIME,
#     predicted_value ???,
# ),

# CREATE TABLE report_prediction(
#     company_id INT,
#     company_name VARCHAR(64),
#     date DATETIME,
#     predicted_value ???,
# ),

# CREATE TABLE stock_prediction(
#     company_id INT,
#     company_name VARCHAR(64),
#     date DATETIME,
#     predicted_value ???,
# )



# """
# CREATE TABLE prediction(
#     prediction_id INT NOT NULL AUTO_INCREMENT,
#     company_id INT,
#     company_name VARCHAR(64),
#     prediction_price FLOAT,
#     date DATETIME,
#     PRIMARY KEY (prediction_id),
#     FOREIGN KEY (company_id) REFERENCES company (company_id)
# )
# """,