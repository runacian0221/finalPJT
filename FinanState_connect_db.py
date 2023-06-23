import requests
from bs4 import BeautifulSoup
import io
import zipfile
import xmltodict
from io import BytesIO
from zipfile import ZipFile
from xml.etree.ElementTree import parse
import pandas as pd
from lxml import html
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus, unquote
import numpy as np
import pymysql
from insert_select_db_f import Database
from insert_data_processor_ver1_f import DataProcessor

# # config.txt에서 설정값을 읽기
# with open("config.txt", "r") as file:
#     exec(file.read())

# # Database 객체를 생성
# db = Database(configs)

# # DataProcessor 객체를 생성
# processor = DataProcessor(db)

# https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019018
KEY = '5b8f0aa6967605a94ba6e9a39b461c2e86cf0d85'

url = '	https://opendart.fss.or.kr/api/corpCode.xml'
params = {'crtfc_key' : KEY}

response = requests.get(url, params = params).content


# 압축 파일 풀어서 xml파일 디렉토리에 저장하기
with ZipFile(BytesIO(response)) as zipfile:
    zipfile.extractall('corpCode')

#corpcode xml 파일 생김

# xml 파일 읽어오기, 여기서 고유번호와 종목명이 포함된 리스트를 만들 것임
xmlTree = parse('./corpCode/CORPCODE.xml')
root = xmlTree.getroot()
raw_list = root.findall('list')

corp_list = []

for l in range(len(raw_list)):
    corp_code = raw_list[l].findtext('corp_code')
    corp_name = raw_list[l].findtext('corp_name')
    stock_code = raw_list[l].findtext('stock_code')
    modify_date = raw_list[l].findtext('modify_date')
    
    corp_list.append([corp_code, corp_name, stock_code, modify_date])
    
#corp_list 형성
# corp_list

#corp_list -> corp_df


from pandas import DataFrame
from datetime import datetime

#위에서 생성한 corp_list에 라벨링하여 데이터프레임화

corp_df = DataFrame(corp_list, columns=[
    '고유번호',
    '정식명칭', 
    '종목코드', 
    '최종변경일자'
])

#corp_df 의 결측치 drop시킨 stock_df (종목코드가 없는 종목을 drop하고, 최종변경일자는 쓰지 않으므로 칼럼 전체를 drop)

stock_df = corp_df[corp_df['종목코드'] != " "]
stock_df = stock_df[['고유번호', '정식명칭', '종목코드', '최종변경일자']].dropna()
# stock_df

# stock_df에서 고유번호를 호출해오는 함수 형성
def goyu_searcher(COMPANYNAME):
    temp_df = stock_df.loc[(stock_df['정식명칭'] == COMPANYNAME)]
    return temp_df.iloc[0, 0]

    # stock_df에서 종목 이름을 호출해오는 함수 형성
def corpname_searcher(CORP_CODE):
    temp_df = stock_df.loc[(stock_df['고유번호'] == CORP_CODE)]
    return temp_df.iloc[0, 1]

    #######

crno_list = []

corp_list_temp = ['삼성전자', '삼성생명', '현대자동차', '셀트리온', 'POSCO홀딩스']
for corp in corp_list_temp:
    crno_list.append(goyu_searcher(corp))

# 가져온 corp_code를 바탕으로 

def get_items(KEY, CORP_CODE, YEAR, RPT_CODE, FS_DIV):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from lxml import html
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode, quote_plus, unquote
    
    
    url = 'https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json'
    params={'crtfc_key' : KEY,
            'corp_code' : CORP_CODE, 
            'bsns_year' : YEAR, 
            'reprt_code' : RPT_CODE, 
            'fs_div' : FS_DIV}
    
    response = requests.get(url, params = params)
    
    import json
    try:
            json_obj = json.loads(response.text)["list"]
    except:
        #     json_obj = ["## ERROR", corpname_searcher(CORP_CODE), YEAR, RPT_CODE]
            json_obj = ["## ERROR", corpname_searcher(CORP_CODE), YEAR] #ok
            # json_obj = [corpname_searcher(CORP_CODE), RPT_CODE] ok
        #     json_obj = [corpname_searcher(CORP_CODE), YEAR, RPT_CODE]
        
    
    return json_obj

    #get_items()를 사용하여 구하고자 하는 기업들의 재무제표 정보 가져오기

corp_list = ['삼성전자','삼성생명', '현대자동차', '셀트리온', 'POSCO홀딩스']
year_list = range(2022, 2023)
reprtlist = ['11013', '11012', '11014', '11011']
fs_divs = ['CFS', 'OFS']

result_list = []

for corp in corp_list:
    for year in year_list:
        for reprt in reprtlist:
            for fs_div in fs_divs:
                result_list.extend(get_items(KEY, goyu_searcher(corp), year, reprt, fs_div))
                
# 구한 결과값을 df로 변환


#틀을 먼저 만들고
rcept_no_dfcol = []
corp_name_dfcol = []
corp_code_dfcol = []
bsns_year_dfcol = []
reprt_code_dfcol = []

result_list = [dict_row for dict_row in result_list if isinstance(dict_row, dict)]

for dict_row in result_list:
    rcept_no_dfcol.append(dict_row['rcept_no'])
    corp_name_dfcol.append(corpname_searcher(dict_row['corp_code']))
    corp_code_dfcol.append(dict_row['corp_code'])
    bsns_year_dfcol.append(dict_row['bsns_year'])
    reprt_code_dfcol.append(dict_row['reprt_code'])

data = {'기준일자': rcept_no_dfcol, '기업명': corp_name_dfcol, '종목코드': corp_code_dfcol, '사업년도': bsns_year_dfcol, '분기코드': reprt_code_dfcol}

result_df = pd.DataFrame(data = data)

# 지표 데이터를 넣을 컬럼 리스트 생성

data_dfcol = [] # 바로 데이터프레임에 컬럼 추가할거긴 한데, 일단 어떤 내용이 들어간지도 모이면 좋을 것 같아서

for dict_row in result_list:
    data_dfcol.append(dict_row['account_nm'])
data_dfcol = list(set(data_dfcol))

for data_col in data_dfcol:
    result_df[data_col] = None
    

result_df = result_df.drop_duplicates().reset_index(drop=True)

for dict_row in result_list:
    # 종목코드, 사업년도, 분기코드가 같은 행을 df에서 찾아서 숫자를 탐색, df에 저장
    
    corp_code = dict_row['corp_code']
    bsns_year = dict_row['bsns_year']
    reprt_code = dict_row['reprt_code']
    
    acc_name = dict_row['account_nm']
    acc_data = dict_row['thstrm_amount']
    
    condition = (result_df['종목코드'] == corp_code) & (result_df['사업년도'] == bsns_year) & (result_df['분기코드'] == reprt_code)
    
    result_df.at[result_df.loc[condition].index, acc_name] = acc_data

dict_row = result_list[0]


corp_code = dict_row['corp_code']
bsns_year = dict_row['bsns_year']
reprt_code = dict_row['reprt_code']

acc_name = dict_row['account_nm']
acc_data = dict_row['thstrm_amount']

# print(corp_code, bsns_year, reprt_code, acc_name, acc_data)
result_df['연차배당금']


# condition = (result_df['종목코드'] == corp_code) & (result_df['사업년도'] == bsns_year) & (result_df['분기코드'] == reprt_code)

# result_df.at[result_df.loc[condition].index, acc_name] = acc_data


test_result_df = result_df.dropna(axis='columns')

# '분기명' 열 생성
test_result_df['분기명'] = test_result_df['분기코드'].map({'11013': '1분기', '11012': '2분기', '11014': '3분기', '11011': '4분기'})
cols = test_result_df.columns.tolist()  # 열 인덱스 리스트로 변환
cols[3], cols[26] = cols[26], cols[3]  # 열 위치 교환
test_result_df = test_result_df[cols]

stock_codes = {'삼성전자' : '005930', '셀트리온' : '068270', '삼성생명' : '032830', 'POSCO홀딩스' : '005490', '현대자동차' : '005380'}

test_result_df['주가코드'] = test_result_df.apply(lambda row: stock_codes.get(row.기업명), axis=1)



#2. PSR = 당기 공시기점의 종가 / (당기매출액 / 발행주식수) 

# corp_list랑 corp_code_list는 api_final_result 파일에서 가져오면 됨

api_final_result = pd.read_csv('csv/api_final_result.csv', dtype={'종목코드':str, '주가코드': str})

# 법인등록번호 가져오기
corp_list = ['삼성전자', '삼성생명', '현대자동차', '셀트리온', 'POSCO홀딩스']
corp_code_list = ['00126380', '00126256', '00164742', '00413046', '00155319']
crnos = []
crnos_dict = {}
for corp_code in corp_code_list:
    params_1 = {
        'crtfc_key' : '5b8f0aa6967605a94ba6e9a39b461c2e86cf0d85',
        'corp_code' : corp_code
    }
    r_temp = requests.get('https://opendart.fss.or.kr/api/company.json', params = params_1)
    import json
    crnos.append(json.loads(r_temp.text)['jurir_no'])
    
    crnos_dict[corp_code] = json.loads(r_temp.text)['jurir_no']


#가져온 법인등록번호로 발행주식수 가져오기

URL = "http://apis.data.go.kr/1160100/service/GetStocIssuInfoService/getItemBasiInfo"
psr_dict = {}

for crno in crnos:
    params = {
        'pageNo' : 1,
        'numOfRows' : 1,
        'resultType' : 'json',
        'serviceKey' : '+21XqtAVW3QkzmAlufEeZE+QJaIxraq0IUyEwMLo0KbKkVfrEecTdm2PLj3j4iNwjSWxzydNtkaFlYQluK68jA==',
        'crno' : crno
    }
    r = requests.get(URL, params = params)
    import json
    try: 
        json_obj = json.loads(r.text)['response']['body']['items']['item'][0]
        
        # 키 : 주식발행회사명, 값 : 발행주식수 -> psr_dict에 저장
        psr_dict[json_obj["stckIssuCmpyNm"]] = json_obj['issuStckCnt'] 
    except: 
        print(json_obj["stckIssuCmpyNm"])
        continue


# stock_data에서 날짜에 맞는 종가 가져오기
import datetime
stock_data = pd.read_csv('./stock_data(5y).csv', dtype={'Code':str})
stock_data = stock_data[['Date', 'Code', 'Close']]
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data['Date'] = stock_data['Date'].dt.strftime('%Y%m%d')


# api_final_result에 stock_data의 기준일자에 맞는 종가를 붙이기
for i in range(len(api_final_result['기준일자'])):
    api_final_result.loc[i, '기준일자'] = api_final_result['기준일자'][i]//1000000


# 분기별 종가를 새 컬럼으로 붙임. 이름은 '분기종가'
api_final_result['분기종가'] = None

for i in range(len(api_final_result)):
    for j in range(len(stock_data)):
        if (str(api_final_result.loc[i, '기준일자']) == stock_data.loc[j, 'Date']) and (str(api_final_result.loc[i, '주가코드']) == stock_data.loc[j, 'Code']):
            api_final_result.loc[i, '분기종가'] = stock_data.loc[j, 'Close']
        else:
            continue


api_final_result #주가코드, 분기별 종가 추가 완료

#발행주식수 추가하자
crnos_list = list(crnos_dict.keys())

api_final_result['발행주식수'] = None

for i in range(len(api_final_result)):
    for j in range(len(crnos_dict.keys())):
        if (api_final_result.loc[i, '종목코드'] == crnos_list[j]):
            api_final_result.loc[i, '발행주식수'] = crnos_dict[crnos_list[j]]

api_final_result['PSR'] = None

for i in range(len(api_final_result)):
    mca = int(api_final_result.loc[i, '매출총이익'] + api_final_result.loc[i, '매출원가'])*1000
    api_final_result.loc[i, 'PSR'] = int(api_final_result.loc[i, '분기종가']) * int(api_final_result.loc[i, '발행주식수']) / mca

api_final_result['GP/A'] = None

for i in range(len(api_final_result)):
    gpa = int(api_final_result.loc[i, '매출총이익']) / int(api_final_result.loc[i, '자산총계'])
    api_final_result.loc[i, 'GP/A'] = gpa*100

# required_columns=['company_id', 'company_name','dctl', 'ncl', 'nci', 'dta', 'ca', 'aia', 'oci', 'cl', 'cs', 'ata', 'cce', 'inv', 'cogs', 'tota', 'nca', 'ia', 'ip', 'tnga', 'ir', 'gp', 'tl', 'fi', 'date', 'quarter', 'fq', 'os', 'gp_a']
# table_name = 'report'
# processor.process_and_insert_report_data(api_final_result, required_columns, table_name, log_info='api_final_result')


    
