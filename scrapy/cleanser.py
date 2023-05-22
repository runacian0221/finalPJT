import re

# 통합 클렌징 코드
def cleansing(text:str) -> str:

    # 특수기호 제거
    # text = re.sub('[▶△▶️◀️▷■◆●]', '', text)
    # ·ㆍ■◆△▷▶▼�"'…※↑↓▲☞ⓒ⅔
    
    text = text.replace('“','"').replace('”','"')
    text = text.replace("‘","'").replace("’","'")
    text = text.replace('·',', ').replace('ㆍ',', ').replace('…','...')

    # 인코딩오류 해결 (공백으로 치환)
    text = re.sub('[https?\:\/*]', '', text)
    text = re.sub('↑', '상승', text)
    text = re.sub('↓', '하락', text)
    text = re.sub('[\xa0\u2008\u2190]', ' ', text)
    text = re.sub('[\(\{\[\<].*?[\)\}\]\>]', '', text)
    text = re.sub('\☞.*', '', text)
    text = re.sub('\▶.*', '', text)
    text = re.sub('연합뉴스TV.*', '', text)
    text = re.sub('[\w\-\.]+(\@|\.)[\w\-\.]+\%?', '', text)
    text = re.sub('[^가-힣a-zA-Z0-9 \.\,]', '', text)
    
    # ., 공백, 줄바꿈 여러개 제거 
    # \s -> 공백( ), 탭(\t), 줄바꿈(\n)
    text = re.sub('[\.]{2,}', '.', text)
    text = re.sub('[\t]+', ' ', text)
    text = re.sub('[ ]{2,}', ' ', text)
    text = re.sub('(\\n)+[(\\n) ]+', '\n', text)
    text = text.strip()

    return text