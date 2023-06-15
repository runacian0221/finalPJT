import pymysql

connection = pymysql.connect(host='13.208.115.44',
                       port=3306,
                       user='runa',
                       password='password',
                       database='finalpjt',
                       charset='utf8mb4')

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
"""

# 사업년도 (Fiscal Year, FY)
# 분기 (Quarter, Q)
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

"""
CREATE TABLE report(
    report_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64)
    date INT,
    quarter VARCHAR(10)
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
    FOREIGN KEY (company_id) REFERENCE company (company_id)
);
""",

"""
CREATE TABLE prediction(
    prediction_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64)
    prediction_price FLOAT,
    date DATETIME,
    PRIMARY KEY (prediction_id),
    FOREIGN KEY (company_id) REFERENCE company (company_id)
)
""",
]

with connection.cursor() as cursor:
    print(cursor.execute('describe news')) #테이블 개수 보여줌
    print(cursor.fetchall())
    