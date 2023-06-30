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
# 사업년도 (Fiscal Year, DATE)
# 분기 (Quarter, Quarter)
# 유형자산의 처분 (Disposal of Tangible Assets, DTA)
# 매출원가 (Cost of Goods Sold, COGS)
# 유동자산 (Current Assets, CA)
# 매출총이익 (Gross Profit, GP)
# 현금및현금성자산 (Cash and Cash Equivalents, CCE)
# 유형자산 (Tangible Assets, TNGA)
# 유동부채 (Current Liabilities, CL)
# 비유동자산 (Non-Current Assets, NCA)
# 재고자산 (Inventory, INV)
# 금융수익 (Financial Income, FI)
# 비유동부채 (Non-Current Liabilities, NCL)
# 이연법인세부채 (Deferred Corporate Tax Liability, DCTL)
# 이자의 지급 (Interest Payment, IP)
# 자본금 (Capital Stock, CS)
# 기타포괄손익 (Other Comprehensive Income, OCI)
# 부채총계 (Total Liabilities, TL)
# 비지배지분 (Non-controlling Interest, NCI)
# 자산총계 (Total Assets, TOTA)
# 유형자산의 취득 (Acquisition of Tangible Assets, ATA)
# 무형자산의 취득 (Acquisition of Intangible Assets, AIA)
# 무형자산 (Intangible Assets, IA)
# 이자의 수취 (Interest Received, IR)
# 분기 종가 (Quarterly closing price, FQ)
# 발행 주식수 (outstanding shares, OS)
"""
CREATE TABLE report(
    report_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64),
    dctl FLOAT, 
    ncl FLOAT, 
    nci FLOAT, 
    dta FLOAT, 
    ca FLOAT, 
    aia FLOAT, 
    oci FLOAT, 
    cl FLOAT, 
    cs FLOAT, 
    ata FLOAT, 
    cce FLOAT, 
    inv FLOAT, 
    cogs FLOAT, 
    tota FLOAT, 
    nca FLOAT, 
    ia FLOAT, 
    ip FLOAT, 
    tnga FLOAT, 
    ir FLOAT, 
    gp FLOAT, 
    tl FLOAT, 
    fi FLOAT, 
    date INT, 
    quarter VARCHAR(10), 
    fq FLOAT, 
    os FLOAT, 
    gp_a FLOAT,
    PRIMARY KEY(report_id),
    FOREIGN KEY (company_id) REFERENCES company (company_id)
);
"""
,
"""
CREATE TABLE stock_prediction1(
    stock_prediction_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64),
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT,
    stock_change FLOAT,
    date DATETIME,
    ma FLOAT,
    std FLOAT,
    upper FLOAT,
    lower FLOAT,
    obv FLOAT,
    cci FLOAT,
    fast_k FLOAT,
    fast_d FLOAT,
    roc FLOAT,
    rsi FLOAT,
    rsi_dp FLOAT,
    mfi FLOAT,
    dp FLOAT,
    ks_roc FLOAT,
    ks_dp FLOAT,
    ks_fast FLOAT,
    score FLOAT,
    PRIMARY KEY(stock_prediction_id),
    FOREIGN KEY (company_id) REFERENCES company (company_id)
)
""",
"""
CREATE TABLE news_analysis(
    news_analysis_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64),
    title text,
    content mediumtext,
    url VARCHAR(2048),
    date DATETIME,
    sentiment VARCHAR(10),
    score FLOAT,
    PRIMARY KEY(news_analysis_id),
    FOREIGN KEY (company_id) REFERENCES company (company_id)
);
""",
"""
CREATE TABLE stock_prediction(
    stock_prediction_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64),
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT,
    stock_change FLOAT,
    date DATETIME,
    ma_5 FLOAT,
    std FLOAT,
    upper FLOAT,
    lower FLOAT,
    obv FLOAT,
    ma FLOAT,
    cci FLOAT,
    fast_k FLOAT,
    fast_d FLOAT,
    roc FLOAT,
    rsi FLOAT,
    rsi_dp FLOAT,
    mfi FLOAT,
    dp FLOAT,
    ks_roc FLOAT,
    ks_dp FLOAT,
    ks_fast FLOAT,
    score FLOAT,
    PRIMARY KEY(stock_prediction_id),
    FOREIGN KEY (company_id) REFERENCES company (company_id)
)
""",
"""
CREATE TABLE news_analysis1(
    news_analysis_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64),
    title VARCHAR(1024),
    url VARCHAR(2048),
    date DATETIME,
    sentiment VARCHAR(10),
    score FLOAT,
    PRIMARY KEY(news_analysis_id),
    FOREIGN KEY (company_id) REFERENCES company (company_id)
);
"""
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
