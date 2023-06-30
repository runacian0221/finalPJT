import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from DB.insert_select_db_ver1 import Database

with open("DB/config.txt", "r") as file:
    exec(file.read())

db = Database(configs)
        
st.set_page_config(
    page_title='Multipage App',
    page_icon = "!@!@",
)

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

#재정상태 분석
def analyze_financials(company_data):
    company_data["재정상태"] = company_data["자산총계"] - company_data["부채총계"]
    recent_trend = company_data["재정상태"].iloc[-1] > company_data["재정상태"].iloc[0]
    positive_trend = "긍정적" if recent_trend else "부정적"
    return company_data, positive_trend

#주가 분석
#def analyze_stock_price(company_data):
#    company_data["주가"] = company_data["시가총액"] / company_data["연간 총매출액"]
#    return company_data

#수익성 계산
def calculate_profitability(company_data):
    company_data['매출총이익'] = company_data['매출총이익'].astype(float)
    company_data['자산총계'] = company_data['자산총계'].astype(float)
    company_data['Profitability'] = round((company_data['매출총이익'] / company_data['자산총계']) * 100,2)
    company_data['Profitability'] = company_data['Profitability'].map("{:.2f}%".format)
    return company_data

def make_clickable(url):
    return f'<a href="{url}" target="_blank">{url}</a>'

#주가데이터 
today = datetime.now().strftime("%Y-%m-%d")
query = f"select * FROM stock WHERE date BETWEEN '2018-01-01' AND '{today}'"
# SQL 쿼리를 실행하여 데이터를 가져옵니다
data, column_names = db.select_data(query=query)
# 데이터를 DataFrame으로 변환합니다
df = pd.DataFrame(data, columns=column_names)

df1 = load_data('api.csv')

query1 = f"select * FROM news_analysis1 WHERE date BETWEEN '2018-01-01' AND '{today}'"
data, column_names = db.select_data(query=query1)
df2 = pd.DataFrame(data, columns=column_names)

query2 = f"select * FROM stock_prediction WHERE date BETWEEN '2018-01-01' AND '{today}'"
data, column_names = db.select_data(query=query2)
df3 = pd.DataFrame(data, columns=column_names)

st.markdown('<h1 style="font-size:30px;">- 💹주가, 기사, 재무제표 분석을 통한 데이터 분석</h1>', unsafe_allow_html=True)
st.sidebar.success("Select a page above.")

def main():
    tab11, tab12 = st.tabs(["개요","종목 데이터"])
    with tab11:
        st.markdown( 
        """
        # 1. 🌱Homepage \n     
        #### 1.1 개요 \n
        #### 1.2 종목 데이터
            (1) 데이터 \n
                - 최근종가 \n
                    - 최근 종가(Close)값과 모맨텀 점수. \n
            (2) 재정상태 \n
                    - 시간에 흐름에 따른 추세.(긍정/부정) \n
                    - 재무상태 변화 그래프(연별) \n
                    - 재무상태 변화 그래프(전체:분기별) \n
            (3) 수익성 \n
                    - 년도 및 분기별 수익성 \n
                
            (4) 뉴스데이터 \n
                    - 해당 일자의 뉴스 데이터  \n
                    - 긍정기사 \n
                    - 부정기사 \n
            
            
        # 2. 🌲EDA (주식 변동 추이)
                - 종목선택 \n
                - 날짜 선택 \n 
                - 추가지표 \n
        
        
        # 3. 🌳EDA3(데이터분석) \n
            (1) 상관관계 분석(히트맵 ,정비례, 반비례, 산점도 그래프)\n
                - 히트맵\n
                - 정비례\n
                - 반비례\n
                - 산점도 그래프\n
            (2) 거래량 Top 주가 변동\n
                - 거래량과 주가의 관계\n
                - 거래량과 주가의 상관관계분석\n
                - 그래프\n
            (3) 거래량 변동\n
                - 해당 기간 전체 거래량 그래프\n
                - 해당 기간 거래량 그래프\n
            (4) 등락률 산점도\n
                - 해당 기간 등락률 분포\n
        """
        )

    with tab12:
        st.markdown('<p style="font-size: 24px; font-weight: bold; color: #336699;">- 주식 종목명을 사이드바에서 선택해주세요!</p>', unsafe_allow_html=True)

        grouped_data = df1.groupby("기업명")
        tab_list = ["삼성전자", "현대차", "포스코", "셀트리온", "삼성생명"]

        # 탭 선택
        selected_tab = st.sidebar.selectbox("기업 선택", tab_list)
        
        
        if selected_tab == '삼성전자':
            company_data = grouped_data.get_group('삼성전자')
            st.subheader("삼성전자")
            tab101, tab102, tab104, tab105 = st.tabs(["종가, 모맨텀 스코어", "재정상태", "수익성","뉴스데이터"])
            with tab101:
                company_name = "삼성전자"
                df_samsung = df[df["company_name"] == company_name]

                if not df_samsung.empty:
                    # 당일 종가 및 전날 대비 등락율 계산
                    df_samsung["change"] = df_samsung["close"].diff()
                    df_samsung["Change_pct"] = df_samsung["change"] / df_samsung["close"].shift() * 100

                    # 최신 데이터 가져오기.
                    latest_close = df_samsung["close"].iloc[-1]
                    latest_change_pct = df_samsung["Change_pct"].iloc[-1]
                        
                    latest_close_formatted = '{:,.0f}'.format(latest_close)
                        

                    # 등락 여부에 따라 색상 설정
                    if latest_change_pct > 0:
                        change_color = "red"
                    elif latest_change_pct < 0:
                        change_color = "blue"
                    else:
                        change_color = "black"
                        
                    st.markdown('''
                               <div style="text-align: center; padding: 5px; background-color: #E8F0FE; border-radius: 5px; color: black;">
                               <h3 style="font-size: 18px; font-weight: normal;">- 최근종가와 모멘텀 점수를 안내해줍니다. </h3>
                              </div>
                             ''', unsafe_allow_html=True)

                    # 당일 종가 메트릭 표시
                    st.metric("Latest Close Price", f"{latest_close_formatted}원")

                    # 전날 대비 등락율 텍스트 표시
                    st.markdown(f"<font color='{change_color}'>change: {latest_change_pct:.2f}%</font>", unsafe_allow_html=True)
                    
                    # 필요한 컬럼 추출
                    
                    # 모멘텀 스코어 계산
                    #N = 30  # 최근 30일을 기준으로 모멘텀 스코어 계산
                    #df_samsung['Return'] = df_samsung['Close'].pct_change(N)  # N일 동안의 주가 수익률 계산
                    #df_samsung['Momentum Score'] = df_samsung['Return'].rolling(N).mean()  # 평균을 내어 모멘텀 스코어 계산

                    # 모멘텀 스코어 그래프
                    #st.subheader("Momentum Score")
                    #st.line_chart(df_samsung[['Date', 'Momentum Score']])
                    
                    recent_data = df_samsung.tail(30)

                    # 종가 컬럼과 날짜 컬럼 추출
                    close_prices = recent_data['close']
                    dates = recent_data['date']

                    # 모멘텀 스코어 계산
                    momentum_scores = (close_prices / close_prices.shift(1) - 1) * 100


                    # 날짜(date)와 점수(score) 컬럼 추출
                    df_subset = df3[['date', 'company_name', 'score']]

                    # company_name과 이름이 일치하는 데이터만 추출
                    company_name = "삼성전자"  # 원하는 회사 이름으로 변경
                    df_subset = df_subset[df_subset['company_name'] == company_name]

                    # 날짜(date) 컬럼을 날짜 형식으로 변환
                    df_subset['date'] = pd.to_datetime(df_subset['date'])

                    # 날짜(date) 컬럼을 기준으로 최신 순으로 정렬
                    df_subset = df_subset.sort_values(by='date', ascending=False)

                    # Streamlit 앱 구성
                    #st.subheader(' {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))
                    st.subheader('-Momentum Score(최근 60일)')
                    
                    st.markdown('#### - {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))

                    st.dataframe(df_subset)


            with tab102:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 18px; font-weight: normal;">- 재무제표 데이터를 통해 시간 흐름에 따른 추세를 안내해줍니다. </h3>
                    </div>
                ''', unsafe_allow_html=True)
                analyzed_data, positive_trend = analyze_financials(company_data)
                if positive_trend == "긍정적":
                    st.write("재정상태 시간 흐름에 따른 추세:", f"<font color='red'>{positive_trend}</font>", "입니다!", unsafe_allow_html=True)
                else:
                    st.write("재정상태 시간 흐름에 따른 추세:", f"<font color='blue'>{positive_trend}</font>", "입니다!", unsafe_allow_html=True)

                sorted_data = analyzed_data.sort_values("사업년도", ascending=False)
                st.write(sorted_data[["사업년도", "분기명", "재정상태"]])

                #st.write(analyzed_data[["사업년도", "분기명","재정상태"]])
                
                fig = px.line(analyzed_data, x="분기명", y="재정상태", color="사업년도", title="재정상태 변화")
                fig.update_layout(xaxis_title="분기", yaxis_title="재정상태")
                st.plotly_chart(fig)
                
                # 사업년도와 분기명을 문자열로 변환하여 합치는 새로운 칼럼 생성
                analyzed_data["기간"] = analyzed_data["사업년도"].astype(str) + "-" + analyzed_data["분기명"].astype(str)

                # 재정상태 변화 그래프
                fig = px.line(analyzed_data, x="기간", y="재정상태", title="재정상태 변화", template="plotly_white")
                fig.update_layout(xaxis_title="기간", yaxis_title="재정상태")
                st.plotly_chart(fig)
 
            with tab104:
                # 수익성
                st.markdown('''
                             <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                            <h3 style="font-size: 16px; font-weight: normal;">- 재무제표 데이터를 통해 해당 기업이 자산을 얼마나 효율적으로 사용하는지 안내합니다. </h3>
                           </div>
                ''', unsafe_allow_html=True)
                st.write("수익성(ROA) = 매출총이익 / 총자산")
                
                # 선택한 종목에 해당하는 데이터 추출
                filtered_df = df1[df1['기업명'] == selected_tab]
                
                if filtered_df.empty:
                    st.write("선택한 종목에 대한 데이터가 없습니다.")
                else:
                    st.write(f"{selected_tab} 종목의 년도 및 분기별 수익성")

                # 사업년도, 분기명 최신부터
                filtered_df = filtered_df.sort_values(["사업년도", "분기명"], ascending=[False, False])

                # 수익성 계산(함수화)
                filtered_df = calculate_profitability(filtered_df)

                st.dataframe(filtered_df[['사업년도', '분기명', '매출총이익', '자산총계', 'Profitability']])

                fig = px.line(filtered_df, x='분기명', y='Profitability', color='사업년도', title='수익성 변화')
                fig.update_layout(xaxis_title='분기', yaxis_title='수익성(%)', xaxis={'categoryorder': 'category ascending'})
                st.plotly_chart(fig)
                
            with tab105:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 긍정 or 부정 뉴스데이터를 확인할 수 있습니다. url 컬럼을 통해 기사 본문도 확인해보세요! </h3>
                    </div>
                ''', unsafe_allow_html=True)
            
                # 종목 선택
                filtered_df = df2[df2['company_name'] == selected_tab]
                            
                if filtered_df.empty:
                    st.write("선택한 종목에 대한 데이터가 없습니다.")
                else:
                    st.write(f"{selected_tab} 종목의 뉴스 데이터")
                
                # 날짜 선택
                date_range = pd.date_range(start=pd.to_datetime(df2['date']).min().date(), end=pd.to_datetime(df2['date']).max().date(), freq='D')
                selected_date = st.selectbox("날짜 선택", date_range[::-1])
                
                if selected_date is not None:
                    selected_date = selected_date.strftime("%Y-%m-%d")

                    # 선택한 종목 및 날짜에 해당하는 데이터 추출
                    selected_news = df2[(df2['company_name'] == selected_tab) & (pd.to_datetime(df2['date']).dt.date == pd.to_datetime(selected_date).date())]

                    # 선택한 종목 및 날짜에 해당하는 기사 제목 및 긍정/부정 정보 출력
                    if not selected_news.empty:
                        st.write(f"선택한 종목 ({selected_tab}), 날짜 ({selected_date})의 뉴스 데이터:")
                        st.write(selected_news[['title', 'sentiment']])
                        
                        # Positive 기사
                        positive_articles = selected_news[selected_news['sentiment'] == 'positive']
                        # Negative 기사
                        negative_articles = selected_news[selected_news['sentiment'] == 'negative']

                        st.header("Positive Articles")
                        st.write(positive_articles[['date','title','sentiment','score','url']])

                        st.header("Negative Articles")
                        st.write(negative_articles[['date','title','sentiment','score','url']])
                    else:
                        st.write("선택한 종목 및 날짜에 해당하는 데이터가 없습니다.")
                else:
                    st.write("날짜를 선택해주세요.")
                    
        elif selected_tab == '현대차':
            company_data = grouped_data.get_group('현대차')
            st.subheader("현대차")
            tab101, tab102, tab104, tab105 = st.tabs(["종가, 모맨텀 스코어", "재정상태", "수익성","뉴스데이터"])
            with tab101:
                company_name = "현대차"
                df_hyundai = df[df['company_name'] == company_name]
                #당일 종가 및 전날 대비 등락율 계산
                df_hyundai["change"] = df_hyundai["close"].diff()
                    
                df_hyundai["Change_pct"] = df_hyundai["change"] / df_hyundai["close"].shift() * 100
                # 최신 데이터 가져오기
                latest_close = df_hyundai["close"].iloc[-1]
                latest_change_pct = df_hyundai["Change_pct"].iloc[-1]
                    
                latest_close_formatted = '{:,.0f}'.format(latest_close)

                    # 등락 여부에 따라 색상 설정
                if latest_change_pct > 0:
                    change_color = "red"
                elif latest_change_pct < 0:
                    change_color = "blue"
                else:
                    change_color = "black"
                    
                st.markdown('''
                            <div style="text-align: center; padding: 5px; background-color: #E8F0FE; border-radius: 5px; color: black;">
                            <h3 style="font-size: 18px; font-weight: normal;">- 최근종가와 모멘텀 점수를 안내해줍니다. </h3>
                            </div>
                           ''', unsafe_allow_html=True)

                # 당일 종가 메트릭 표시
                st.metric("Latest Close Price", f"{latest_close_formatted}원")
                
                # 전날 대비 등락율 텍스트 표시
                st.markdown(f"<font color='{change_color}'>change: {latest_change_pct:.2f}%</font>", unsafe_allow_html=True)
                
                

                 # 날짜(date)와 점수(score) 컬럼 추출
                df_subset = df3[['date', 'company_name', 'score']]

                # company_name과 이름이 일치하는 데이터만 추출
                company_name = "현대차"  # 원하는 회사 이름으로 변경
                df_subset = df_subset[df_subset['company_name'] == company_name]

                # 날짜(date) 컬럼을 날짜 형식으로 변환
                df_subset['date'] = pd.to_datetime(df_subset['date'])

                # 날짜(date) 컬럼을 기준으로 최신 순으로 정렬
                df_subset = df_subset.sort_values(by='date', ascending=False)

                # Streamlit 앱 구성
                #st.subheader(' {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))
                st.subheader('-Momentum Score(최근 60일)')
                                    
                st.markdown('#### - {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))

                st.dataframe(df_subset)
                
                
                
            with tab102:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 18px; font-weight: normal;">- 재무제표 데이터를 통해 시간 흐름에 따른 추세를 안내해줍니다. </h3>
                    </div>
                ''', unsafe_allow_html=True)
                analyzed_data, positive_trend = analyze_financials(company_data)
                if positive_trend == "긍정적":
                    st.write("재정상태 시간 흐름에 따른 추세:", f"<font color='red'>{positive_trend}</font>", "입니다!", unsafe_allow_html=True)
                else:
                    st.write("재정상태 시간 흐름에 따른 추세:", f"<font color='blue'>{positive_trend}</font>", "입니다!", unsafe_allow_html=True)

                sorted_data = analyzed_data.sort_values("사업년도", ascending=False)
                st.write(sorted_data[["사업년도", "분기명", "재정상태"]])

                #st.write(analyzed_data[["사업년도", "분기명","재정상태"]])
                
                fig = px.line(analyzed_data, x="분기명", y="재정상태", color="사업년도", title="재정상태 변화")
                fig.update_layout(xaxis_title="분기", yaxis_title="재정상태")
                st.plotly_chart(fig)
                
                # 사업년도와 분기명을 문자열로 변환하여 합치는 새로운 칼럼 생성
                analyzed_data["기간"] = analyzed_data["사업년도"].astype(str) + "-" + analyzed_data["분기명"].astype(str)

                # 재정상태 변화 그래프
                fig = px.line(analyzed_data, x="기간", y="재정상태", title="재정상태 변화", template="plotly_white")
                fig.update_layout(xaxis_title="기간", yaxis_title="재정상태")
                st.plotly_chart(fig)
                
            with tab104:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 재무제표 데이터를 통해 해당 기업이 자산을 얼마나 효율적으로 사용하는지 안내합니다. </h3>
                    </div>
                ''', unsafe_allow_html=True)
                selected_stock = "현대차"  # Replace "삼성전자" with the default stock you want to display

                filtered_df = df1[df1['기업명'] == selected_stock]
                if filtered_df.empty:
                    st.write("선택한 종목에 대한 데이터가 없습니다.")
                else:
                    st.write(f"{selected_stock} 종목의 년도 및 분기별 수익성")

                # 사업년도, 분기명 최신부터
                filtered_df = filtered_df.sort_values(["사업년도", "분기명"], ascending=[False, False])

                # 수익성 계산(함수화)
                filtered_df = calculate_profitability(filtered_df)

                st.dataframe(filtered_df[['사업년도', '분기명', '매출총이익', '자산총계', 'Profitability']])

                fig = px.line(filtered_df, x='분기명', y='Profitability', color='사업년도', title='수익성 변화')
                fig.update_layout(xaxis_title='분기', yaxis_title='수익성(%)', xaxis={'categoryorder': 'category ascending'})
                st.plotly_chart(fig)
                
            with tab105:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 긍정 or 부정 뉴스데이터를 확인할 수 있습니다. url 컬럼을 통해 기사 본문도 확인해보세요! </h3>
                    </div>
                ''', unsafe_allow_html=True)
            
                # 종목 선택
                filtered_df = df2[df2['company_name'] == selected_tab]
                            
                if filtered_df.empty:
                    st.write("선택한 종목에 대한 데이터가 없습니다.")
                else:
                    st.write(f"{selected_tab} 종목의 뉴스 데이터")
                
                # 날짜 선택
                date_range = pd.date_range(start=pd.to_datetime(df2['date']).min().date(), end=pd.to_datetime(df2['date']).max().date(), freq='D')
                selected_date = st.selectbox("날짜 선택", date_range[::-1])
                
                if selected_date is not None:
                    selected_date = selected_date.strftime("%Y-%m-%d")

                    # 선택한 종목 및 날짜에 해당하는 데이터 추출
                    selected_news = df2[(df2['company_name'] == selected_tab) & (pd.to_datetime(df2['date']).dt.date == pd.to_datetime(selected_date).date())]

                    # 선택한 종목 및 날짜에 해당하는 기사 제목 및 긍정/부정 정보 출력
                    if not selected_news.empty:
                        st.write(f"선택한 종목 ({selected_tab}), 날짜 ({selected_date})의 뉴스 데이터:")
                        st.write(selected_news[['title', 'sentiment']])
                        
                        # Positive 기사
                        positive_articles = selected_news[selected_news['sentiment'] == 'positive']
                        # Negative 기사
                        negative_articles = selected_news[selected_news['sentiment'] == 'negative']

                        st.header("Positive Articles")
                        st.write(positive_articles[['date','title','sentiment','score','url']])

                        st.header("Negative Articles")
                        st.write(negative_articles[['date','title','sentiment','score','url']])
                    else:
                        st.write("선택한 종목 및 날짜에 해당하는 데이터가 없습니다.")
                else:
                    st.write("날짜를 선택해주세요.")
                

                    
        elif selected_tab == "포스코":
            company_data = grouped_data.get_group('포스코')
            st.header("포스코")
            tab101, tab102, tab104, tab105 = st.tabs(["종가, 모맨텀 스코어", "재정상태", "수익성","뉴스데이터"])
            with tab101:
                company_name = '포스코'
                df_posco = df[df["company_name"] == company_name]

                df_posco["change"] = df_posco["close"].diff()
                    
                df_posco["Change_pct"] = df_posco["change"] / df_posco["close"].shift() * 100
                # 최신 데이터 가져오기
                latest_close = df_posco["close"].iloc[-1]
                latest_change_pct = df_posco["Change_pct"].iloc[-1]
                    
                latest_close_formatted = '{:,.0f}'.format(latest_close)

                # 등락 여부에 따라 색상 설정
                if latest_change_pct > 0:
                    change_color = "red"
                elif latest_change_pct < 0:
                    change_color = "blue"
                else:
                    change_color = "black"
                    
                st.markdown('''
                            <div style="text-align: center; padding: 5px; background-color: #E8F0FE; border-radius: 5px; color: black;">
                            <h3 style="font-size: 18px; font-weight: normal;">- 최근종가와 모멘텀 점수를 안내해줍니다. </h3>
                            </div>
                           ''', unsafe_allow_html=True)

                # 당일 종가 메트릭 표시
                st.metric("Latest Close Price", f"{latest_close_formatted}원")

                # 전날 대비 등락율 텍스트 표시
                st.markdown(f"<font color='{change_color}'>change: {latest_change_pct:.2f}%</font>", unsafe_allow_html=True)
                
                df_subset = df3[['date', 'company_name', 'score']]

                # company_name과 이름이 일치하는 데이터만 추출
                company_name = "포스코"  # 원하는 회사 이름으로 변경
                df_subset = df_subset[df_subset['company_name'] == company_name]

                # 날짜(date) 컬럼을 날짜 형식으로 변환
                df_subset['date'] = pd.to_datetime(df_subset['date'])

                # 날짜(date) 컬럼을 기준으로 최신 순으로 정렬
                df_subset = df_subset.sort_values(by='date', ascending=False)

                # Streamlit 앱 구성
                #st.subheader(' {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))
                
                st.subheader('-Momentum Score(최근 60일)')
            
                st.markdown('#### - {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))

                st.dataframe(df_subset)
                
            with tab102:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 18px; font-weight: normal;">- 재무제표 데이터를 통해 시간 흐름에 따른 추세를 안내해줍니다. </h3>
                    </div>
                ''', unsafe_allow_html=True)
                analyzed_data, positive_trend = analyze_financials(company_data)
                if positive_trend == "긍정적":
                    st.write("재정상태 시간 흐름에 따른 추세:", f"<font color='red'>{positive_trend}</font>", "입니다!", unsafe_allow_html=True)
                else:
                    st.write("재정상태 시간 흐름에 따른 추세:", f"<font color='blue'>{positive_trend}</font>", "입니다!", unsafe_allow_html=True)

                sorted_data = analyzed_data.sort_values("사업년도", ascending=False)
                st.write(sorted_data[["사업년도", "분기명", "재정상태"]])

                #st.write(analyzed_data[["사업년도", "분기명","재정상태"]])
                
                fig = px.line(analyzed_data, x="분기명", y="재정상태", color="사업년도", title="재정상태 변화")
                fig.update_layout(xaxis_title="분기", yaxis_title="재정상태")
                st.plotly_chart(fig)
                
                # 사업년도와 분기명을 문자열로 변환하여 합치는 새로운 칼럼 생성
                analyzed_data["기간"] = analyzed_data["사업년도"].astype(str) + "-" + analyzed_data["분기명"].astype(str)

                # 재정상태 변화 그래프
                fig = px.line(analyzed_data, x="기간", y="재정상태", title="재정상태 변화", template="plotly_white")
                fig.update_layout(xaxis_title="기간", yaxis_title="재정상태")
                st.plotly_chart(fig)
                

                
                # 사업년도와 분기명을 문자열로 변환하여 합치는 새로운 칼럼 생성

                
                    
            with tab104:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 재무제표 데이터를 통해 해당 기업이 자산을 얼마나 효율적으로 사용하는지 안내합니다. </h3>
                    </div>
                ''', unsafe_allow_html=True)
                selected_stock = "삼성전자"  # Replace "삼성전자" with the default stock you want to display

                filtered_df = df1[df1['기업명'] == selected_stock]
                if filtered_df.empty:
                    st.write("선택한 종목에 대한 데이터가 없습니다.")
                else:
                    st.write(f"{selected_stock} 종목의 년도 및 분기별 수익성")

                # 사업년도, 분기명 최신부터
                filtered_df = filtered_df.sort_values(["사업년도", "분기명"], ascending=[False, False])

                # 수익성 계산(함수화)
                filtered_df = calculate_profitability(filtered_df)

                st.dataframe(filtered_df[['사업년도', '분기명', '매출총이익', '자산총계', 'Profitability']])

                fig = px.line(filtered_df, x='분기명', y='Profitability', color='사업년도', title='수익성 변화')
                fig.update_layout(xaxis_title='분기', yaxis_title='수익성(%)', xaxis={'categoryorder': 'category ascending'})
                st.plotly_chart(fig)
                
            with tab105:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 긍정 or 부정 뉴스데이터를 확인할 수 있습니다. url 컬럼을 통해 기사 본문도 확인해보세요! </h3>
                    </div>
                ''', unsafe_allow_html=True)
            
                # 종목 선택
                filtered_df = df2[df2['company_name'] == selected_tab]
                            
                if filtered_df.empty:
                    st.write("선택한 종목에 대한 데이터가 없습니다.")
                else:
                    st.write(f"{selected_tab} 종목의 뉴스 데이터")
                
                # 날짜 선택
                date_range = pd.date_range(start=pd.to_datetime(df2['date']).min().date(), end=pd.to_datetime(df2['date']).max().date(), freq='D')
                selected_date = st.selectbox("날짜 선택", date_range[::-1])
                
                if selected_date is not None:
                    selected_date = selected_date.strftime("%Y-%m-%d")

                    # 선택한 종목 및 날짜에 해당하는 데이터 추출
                    selected_news = df2[(df2['company_name'] == selected_tab) & (pd.to_datetime(df2['date']).dt.date == pd.to_datetime(selected_date).date())]

                    # 선택한 종목 및 날짜에 해당하는 기사 제목 및 긍정/부정 정보 출력
                    if not selected_news.empty:
                        st.write(f"선택한 종목 ({selected_tab}), 날짜 ({selected_date})의 뉴스 데이터:")
                        st.write(selected_news[['title', 'sentiment']])
                        
                        # Positive 기사
                        positive_articles = selected_news[selected_news['sentiment'] == 'positive']
                        # Negative 기사
                        negative_articles = selected_news[selected_news['sentiment'] == 'negative']

                        st.header("Positive Articles")
                        st.write(positive_articles[['date','title','sentiment','score','url']])

                        st.header("Negative Articles")
                        st.write(negative_articles[['date','title','sentiment','score','url']])
                    else:
                        st.write("선택한 종목 및 날짜에 해당하는 데이터가 없습니다.")
                else:
                    st.write("날짜를 선택해주세요.")
                
        elif selected_tab == '셀트리온':
            company_data = grouped_data.get_group('셀트리온')
            st.header("셀트리온")
            tab101, tab102, tab104, tab105 = st.tabs(["종가, 모맨텀 스코어", "재정상태", "수익성","뉴스데이터"])
            with tab101:
                company_name = "셀트리온"
                df_celltrion = df[df["company_name"] == company_name]
                df_celltrion["change"] = df_celltrion["close"].diff()
                    
                df_celltrion["Change_pct"] = df_celltrion["change"] / df_celltrion["close"].shift() * 100
                # 최신 데이터 가져오기
                latest_close = df_celltrion["close"].iloc[-1]
                latest_change_pct = df_celltrion["Change_pct"].iloc[-1]
                
                latest_close_formatted = '{:,.0f}'.format(latest_close)

                # 등락 여부에 따라 색상 설정
                if latest_change_pct > 0:
                    change_color = "red"
                elif latest_change_pct < 0:
                    change_color = "blue"
                else:
                    change_color = "black"
                    
                st.markdown('''
                            <div style="text-align: center; padding: 5px; background-color: #E8F0FE; border-radius: 5px; color: black;">
                            <h3 style="font-size: 18px; font-weight: normal;">- 최근종가와 모멘텀 점수를 안내해줍니다. </h3>
                            </div>
                           ''', unsafe_allow_html=True)      

                # 당일 종가 메트릭 표시
                st.metric("Latest Close Price", f"{latest_close_formatted}원")

                # 전날 대비 등락율 텍스트 표시
                st.markdown(f"<font color='{change_color}'>change: {latest_change_pct:.2f}%</font>", unsafe_allow_html=True)
                
                df_subset = df3[['date', 'company_name', 'score']]

                # company_name과 이름이 일치하는 데이터만 추출
                company_name = "셀트리온"  # 원하는 회사 이름으로 변경
                df_subset = df_subset[df_subset['company_name'] == company_name]

                # 날짜(date) 컬럼을 날짜 형식으로 변환
                df_subset['date'] = pd.to_datetime(df_subset['date'])

                # 날짜(date) 컬럼을 기준으로 최신 순으로 정렬
                df_subset = df_subset.sort_values(by='date', ascending=False)

                # Streamlit 앱 구성
                #st.subheader(' {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))
                
                st.subheader('-Momentum Score(최근 60일)')
                    
                st.markdown('#### - {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))

                st.dataframe(df_subset)                
                
            
            with tab102:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 18px; font-weight: normal;">- 재무제표 데이터를 통해 시간 흐름에 따른 추세를 안내해줍니다. </h3>
                    </div>
                ''', unsafe_allow_html=True)
                analyzed_data, positive_trend = analyze_financials(company_data)
                if positive_trend == "긍정적":
                    st.write("재정상태 시간 흐름에 따른 추세:", f"<font color='red'>{positive_trend}</font>", "입니다!", unsafe_allow_html=True)
                else:
                    st.write("재정상태 시간 흐름에 따른 추세:", f"<font color='blue'>{positive_trend}</font>", "입니다!", unsafe_allow_html=True)

                sorted_data = analyzed_data.sort_values("사업년도", ascending=False)
                st.write(sorted_data[["사업년도", "분기명", "재정상태"]])

                #st.write(analyzed_data[["사업년도", "분기명","재정상태"]])
                
                fig = px.line(analyzed_data, x="분기명", y="재정상태", color="사업년도", title="재정상태 변화")
                fig.update_layout(xaxis_title="분기", yaxis_title="재정상태")
                st.plotly_chart(fig)
                
                # 사업년도와 분기명을 문자열로 변환하여 합치는 새로운 칼럼 생성
                analyzed_data["기간"] = analyzed_data["사업년도"].astype(str) + "-" + analyzed_data["분기명"].astype(str)

                # 재정상태 변화 그래프
                fig = px.line(analyzed_data, x="기간", y="재정상태", title="재정상태 변화", template="plotly_white")
                fig.update_layout(xaxis_title="기간", yaxis_title="재정상태")
                st.plotly_chart(fig)
            with tab104:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 재무제표 데이터를 통해 해당 기업이 자산을 얼마나 효율적으로 사용하는지 안내합니다. </h3>
                    </div>
                ''', unsafe_allow_html=True)
                selected_stock = "셀트리온"  # Replace "삼성전자" with the default stock you want to display

                filtered_df = df1[df1['기업명'] == selected_stock]
                if filtered_df.empty:
                    st.write("선택한 종목에 대한 데이터가 없습니다.")
                else:
                    st.write(f"{selected_stock} 종목의 년도 및 분기별 수익성")

                # 사업년도, 분기명 최신부터
                filtered_df = filtered_df.sort_values(["사업년도", "분기명"], ascending=[False, False])

                # 수익성 계산(함수화)
                filtered_df = calculate_profitability(filtered_df)

                st.dataframe(filtered_df[['사업년도', '분기명', '매출총이익', '자산총계', 'Profitability']])

                fig = px.line(filtered_df, x='분기명', y='Profitability', color='사업년도', title='수익성 변화')
                fig.update_layout(xaxis_title='분기', yaxis_title='수익성(%)', xaxis={'categoryorder': 'category ascending'})
                st.plotly_chart(fig)
            
            
            #센트리온만 적용해보자


            with tab105:

                # 종목 선택
                filtered_df = df2[df2['company_name'] == selected_tab]
                
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 긍정 or 부정 뉴스데이터를 확인할 수 있습니다. url 컬럼을 통해 기사 본문도 확인해보세요! </h3>
                    </div>
                ''', unsafe_allow_html=True)
                            
                if filtered_df.empty:
                    st.write("선택한 종목에 대한 데이터가 없습니다.")
                else:
                    st.write(f"{selected_tab} 종목의 뉴스 데이터")
                
                # 날짜 선택
                date_range = pd.date_range(start=pd.to_datetime(df2['date']).min().date(), end=pd.to_datetime(df2['date']).max().date(), freq='D')
                selected_date = st.selectbox("날짜 선택", date_range[::-1])
                
                if selected_date is not None:
                    selected_date = selected_date.strftime("%Y-%m-%d")

                    # 선택한 종목 및 날짜에 해당하는 데이터 추출
                    selected_news = df2[(df2['company_name'] == selected_tab) & (pd.to_datetime(df2['date']).dt.date == pd.to_datetime(selected_date).date())]

                    # 선택한 종목 및 날짜에 해당하는 기사 제목 및 긍정/부정 정보 출력
                    if not selected_news.empty:
                        st.write(f"선택한 종목 ({selected_tab}), 날짜 ({selected_date})의 뉴스 데이터:")
                        st.write(selected_news[['title', 'sentiment']])
                        
                        # Positive 기사
                        positive_articles = selected_news[selected_news['sentiment'] == 'positive']
                        # Negative 기사
                        negative_articles = selected_news[selected_news['sentiment'] == 'negative']

                        st.header("Positive Articles")
                        st.write(positive_articles[['date','title','sentiment','score','url']])

                        st.header("Negative Articles")
                        st.write(negative_articles[['date','title','sentiment','score','url']])
                    else:
                        st.write("선택한 종목 및 날짜에 해당하는 데이터가 없습니다.")
                else:
                    st.write("날짜를 선택해주세요.")
                
                    
        elif selected_tab == '삼성생명':
            grouped_data = df.groupby("company_name")            
            company_data = grouped_data.get_group("삼성생명")
            st.header("삼성생명")
            tab101, tab102, tab104, tab105 = st.tabs(["종가, 모맨텀 스코어", "재정상태", "수익성","뉴스데이터"])
            with tab101:
                company_name = "삼성생명"
                df_s_life = df[df["company_name"] == company_name]
                df_s_life["change"] = df_s_life["close"].diff()
                
                df_s_life["Change_pct"] = df_s_life["change"] / df_s_life["close"].shift() * 100
                # 최신 데이터 가져오기
                latest_close = df_s_life["close"].iloc[-1]
                latest_change_pct = df_s_life["Change_pct"].iloc[-1]
                
                latest_close_formatted = '{:,.0f}'.format(latest_close)

                # 등락 여부에 따라 색상 설정
                if latest_change_pct > 0:
                    change_color = "red"
                elif latest_change_pct < 0:
                    change_color = "blue"
                else:
                    change_color = "black"
                    
                st.markdown('''
                            <div style="text-align: center; padding: 5px; background-color: #E8F0FE; border-radius: 5px; color: black;">
                            <h3 style="font-size: 18px; font-weight: normal;">- 최근종가와 모멘텀 점수를 안내해줍니다. </h3>
                            </div>
                           ''', unsafe_allow_html=True)                      
                
                st.subheader(" - 최근 종가")

                # 당일 종가 메트릭 표시
                st.metric("Latest Close Price", f"{latest_close_formatted}원")

                    # 전날 대비 등락율 텍스트 표시
                st.markdown(f"<font color='{change_color}'>change: {latest_change_pct:.2f}%</font>", unsafe_allow_html=True)
                
                df_subset = df3[['date', 'company_name', 'score']]

                # company_name과 이름이 일치하는 데이터만 추출
                company_name = "삼성생명"  # 원하는 회사 이름으로 변경
                df_subset = df_subset[df_subset['company_name'] == company_name]

                # 날짜(date) 컬럼을 날짜 형식으로 변환
                df_subset['date'] = pd.to_datetime(df_subset['date'])

                # 날짜(date) 컬럼을 기준으로 최신 순으로 정렬
                df_subset = df_subset.sort_values(by='date', ascending=False)

                # Streamlit 앱 구성
                #st.subheader(' {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))
                    
                st.markdown('#### - {}의 금일 종가에 비해 60일 이후 종가 평균이 클 확률'.format(company_name))

                st.dataframe(df_subset)
                
            with tab102:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 해당 데이터가 존재하지 않습니다. </h3>
                    </div>
                ''', unsafe_allow_html=True)


            
            with tab104:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 해당 데이터가 존재하지 않습니다. </h3>
                    </div>
                ''', unsafe_allow_html=True)
            with tab105:
                st.markdown('''
                    <div style="text-align: center; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- 긍정 or 부정 뉴스데이터를 확인할 수 있습니다. url 컬럼을 통해 기사 본문도 확인해보세요! </h3>
                    </div>
                ''', unsafe_allow_html=True)
            
                # 종목 선택
                filtered_df = df2[df2['company_name'] == selected_tab]
                            
                if filtered_df.empty:
                    st.write("선택한 종목에 대한 데이터가 없습니다.")
                else:
                    st.write(f"{selected_tab} 종목의 뉴스 데이터")
                
                # 날짜 선택
                date_range = pd.date_range(start=pd.to_datetime(df2['date']).min().date(), end=pd.to_datetime(df2['date']).max().date(), freq='D')
                selected_date = st.selectbox("날짜 선택", date_range[::-1])
                
                if selected_date is not None:
                    selected_date = selected_date.strftime("%Y-%m-%d")

                    # 선택한 종목 및 날짜에 해당하는 데이터 추출
                    selected_news = df2[(df2['company_name'] == selected_tab) & (pd.to_datetime(df2['date']).dt.date == pd.to_datetime(selected_date).date())]

                    # 선택한 종목 및 날짜에 해당하는 기사 제목 및 긍정/부정 정보 출력
                    if not selected_news.empty:
                        st.write(f"선택한 종목 ({selected_tab}), 날짜 ({selected_date})의 뉴스 데이터:")
                        st.write(selected_news[['title', 'sentiment']])
                        
                        # Positive 기사
                        positive_articles = selected_news[selected_news['sentiment'] == 'positive']
                        # Negative 기사
                        negative_articles = selected_news[selected_news['sentiment'] == 'negative']

                        st.header("Positive Articles")
                        st.write(positive_articles[['date','title','sentiment','score','url']])

                        st.header("Negative Articles")
                        st.write(negative_articles[['date','title','sentiment','score','url']])
                    else:
                        st.write("선택한 종목 및 날짜에 해당하는 데이터가 없습니다.")
                else:
                    st.write("날짜를 선택해주세요.")
                    

if __name__ == '__main__' :
    main()
    
# 해야할일
# 삼성전자 뉴스 크롤링 잘린거 마저 해야함
# 자동화 코드 - 가능할지 미지수
# docker에 올리기
# 지표설명 넣기 - 킹무위키
# 모멘텀 지표 넣기
# 사업년도 콤마(,) 지우기
# 발표자료