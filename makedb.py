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
    title VARCHAR(64),
    content VARCHAR(64),
    url VARCHAR(2048),
    writed_at DATETIME,
    PRIMARY KEY(news_id),
    FOREIGN KEY (company_id) REFERENCES company (company_id)
);
""",

"""
CREATE TABLE stock(
    stock_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    name VARCHAR(64)
    date DATETIME,
    open INT,
    high INT,
    low INT,
    close INT,
    volume INT,
    stock_change FLOAT,
    ma_5 FLOAT,
    fast_%k FLOAT,
    slow_%k FLOAT,
    rsi FLOAT,
    std FLOAT,
    upper FLOAT,
    lower FLOAT,
    PRIMARY KEY(stock_id),
    FOREIGN KEY (company_id) REFERENCE company (company_id)
);
""",

"""
CREATE TABLE report(
    report_id INT NOT NULL AUTO_INCREMENT,
    company_id INT,
    company_name VARCHAR(64)
    sales FLOAT,
    operating_income FLOAT,
    assets FLOAT,
    liabilities FLOAT,
    stockholders_equity FLOAT,
    capital_stock FLOAT,
    cfo FLOAT,
    cfi FLOAT,
    cff FLOAT,
    roe FLOAT,
    roa FLOAT,
    per FLOAT,
    pbr FLOAT,
    psr FLOAT,
    pcr FLOAT,
    dy FLOAT,
    date DATETIME,
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
    