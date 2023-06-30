import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import locale
from DB.insert_select_db import Database
import random


with open("DB/config.txt", "r") as file:
    exec(file.read())

db = Database(configs)

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    query = f"select * FROM stock_prediction WHERE date BETWEEN '2018-01-01' AND '{today}'"
    # SQL 쿼리를 실행하여 데이터를 가져옵니다
    data, column_names = db.execute_query(query)

    # 데이터를 DataFrame으로 변환합니다
    df = pd.DataFrame(data, columns=column_names)
    st.title('주식추천')
    st.subheader('1')
    stocks = sorted(df['company_name'].unique())
    selected_stocks = st.sidebar.multiselect('Select Brands', stocks, default=stocks[0])

    default_start_date = pd.to_datetime(df['date']).min().to_pydatetime().date()
    start_date = st.sidebar.date_input('Start Date', value=default_start_date)
    end_date = st.sidebar.date_input('End Date', value=datetime.now())
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    fig = go.Figure()
    
    color_palette = ['blue', 'red', 'green', 'orange', 'purple', 'yellow']
    
    for i, stock in enumerate(selected_stocks):
        stock_data = df[(df['company_name'] == stock) & (df['date'] >= start_date) & (df['date'] <= end_date)]
        color = color_palette[i % len(color_palette)]  # 순환적으로 색상 선택
        fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name=stock,
                                hovertemplate='날짜: %{x}<br>주식 가격: %{y:,.0f}',
                                line=dict(color=color)))
        price_diff = stock_data['close'].diff()
        color = ['blue' if diff < 0 else 'red' for diff in price_diff]
        fig.add_trace(go.Bar(x=stock_data['date'], y=stock_data['high'] - stock_data['low'],
                            base=stock_data['low'], name='Price Range', marker=dict(color=color),
                            hovertemplate='날짜: %{x}<br>가격 범위: %{y:,.0f}'))

    fig.update_layout(
        title='주식 가격 변동 추이',
        xaxis_title='날짜',
        yaxis_title='주식 가격',
        hovermode='x',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        barmode='stack',
        autosize=False,
        width=800,
        height=500,
        
    )

    # 천단위 쉼표 추가
    locale.setlocale(locale.LC_ALL, '')
    fig.update_yaxes(tickformat=",.0f")

    # 그래프 출력
    st.plotly_chart(fig)

    # 추가 그래프
    st.sidebar.subheader('추가 그래프')
    options = ['Fast STC', 'MA','Volume','STD','Bolinger Band', 'OBV','CCI','RSI','MFI','ROC','DP'] # 'SLOW STC'
    selected_graphs = st.sidebar.multiselect('그래프를 선택하세요', options)

    for stock in selected_stocks:
        stock_data = df[(df['company_name'] == stock) & (df['date'] >= start_date) & (df['date'] <= end_date)]

        if 'Fast STC' in selected_graphs:
            st.subheader(f'Fast STC(스토캐스틱) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;">- Fast STC란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> fast_k는 일정기간(9일) 동안의 최고가와 최저가의 범위 중에서 현재 가격의 위치를 백분율로 나타내는 기술적 지표입니다.  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> fast_d는 fast_k의 이동평균(5일)을 나타낸 값입니다.</h3>
                        </div>
                        ''', unsafe_allow_html=True)    
            fig_fast_stc = go.Figure()
            fig_fast_stc.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['fast_k'], name='fast_%K',
                                              hovertemplate='날짜: %{x}<br>SLOW %K: %{y:.2f}'))
            fig_fast_stc.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['fast_d'], name='fast_%D',
                                              hovertemplate='날짜: %{x}<br>SLOW %D: %{y:.2f}'))
            fig_fast_stc.add_trace(go.Scatter(x=stock_data['date'], y=[80] * len(stock_data), name='Upper Bound', line=dict(color='red', dash='dash')))
            fig_fast_stc.add_trace(go.Scatter(x=stock_data['date'], y=[20] * len(stock_data), name='Lower Bound', line=dict(color='green', dash='dash')))
            fig_fast_stc.update_layout(
                autosize=False,
                width=900,
                height=600 
            )
            st.plotly_chart(fig_fast_stc) 


        if 'MA' in selected_graphs:
            st.subheader(f'MA(이동평균선) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- MA(Moving Average))란?  </h3>

                        <h3 style="font-size: 14px; font-weight: normal;"> 최근(20일) 주가의 평균값을 나타내는 지표입니다.  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 이동평균선과 현재 가격을 비교하여 추세의 변화를 파악 할 수 있습니다..</h3>
                        </div>
                        ''', unsafe_allow_html=True)
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['ma'], name='MA',
                                        hovertemplate='날짜: %{x}<br>MA_5: %{y:.2f}'))
            #fig_ma.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MA'], name='MA_20',
            #                           hovertemplate='날짜: %{x}<br>MA_20: %{y:.2f}'))
            fig_ma.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Close',
                                        hovertemplate='날짜: %{x}<br>Close: %{y:.2f}'))
            #fig_ma.add_trace(go.Scatter(x=stock_data['Date'], y=[700000] * len(stock_data), name='Upper Bound', line=dict(color='red', dash='dash')))
            #fig_ma.add_trace(go.Scatter(x=stock_data['Date'], y=[300000] * len(stock_data), name='Under Bound', line=dict(color='green', dash='dash')))
            fig_ma.update_layout(
                autosize=False,
                width=900,
                height=600
            
            )
            st.plotly_chart(fig_ma)
            
            
        if 'Volume' in selected_graphs:
            st.subheader(f'거래량 - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- Volume 이란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 하루의 거래된 주식의 수를 나타냅니다. </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 거래량은 주식이 얼마나 활발하게 거래되는지 나타냅니다.  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 주식에 대한 수급 강도를 보여줍니다.  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 가격에 선행하는 지표로 여겨지기도 합니다. </h3>
                        </div>
                        ''', unsafe_allow_html=True)
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['volume'], name='거래량',
                                        hovertemplate='날짜: %{x}<br>Volume: %{y:.2f}'))
            #fig_volume.add_trace(go.Scatter(x=stock_data['Date'], y=[700000] * len(stock_data), name='Lower Bound', line=dict(color='red', dash='dash')))
            #fig_volume.add_trace(go.Scatter(x=stock_data['Date'], y=[300000] * len(stock_data), name='Upper Bound', line=dict(color='green', dash='dash')))
            fig_volume.update_layout(
                autosize=False,
                width=900,
                height=600  
            )
            st.plotly_chart(fig_volume)
            
        if 'STD' in selected_graphs:
            st.subheader(f'STD(표준편차) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- STD(Standard Deviation)란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 일정기간(20일) 종가의 표준편차  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 크게 변동하고 있는 종목들을 검색 할 수 있습니다. (주로 보조로만 사용) </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 해당 기간 가격의 변화폭 크기를 나타냅니다. </h3>

                        </div>
                        ''', unsafe_allow_html=True)
            fig_STD = go.Figure()
            fig_STD.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['std'], name='STD',
                                        hovertemplate='날짜: %{x}<br>STD: %{y:.2f}'))
            #fig_STD.add_trace(go.Scatter(x=stock_data['Date'], y=[700000] * len(stock_data), name='Lower Bound', line=dict(color='red', dash='dash')))
            #fig_STD.add_trace(go.Scatter(x=stock_data['Date'], y=[300000] * len(stock_data), name='Upper Bound', line=dict(color='green', dash='dash')))
            fig_STD.update_layout(
                autosize=False,
                width=900,
                height=600  
            )
            st.plotly_chart(fig_STD)
            
        if 'Bolinger Band' in selected_graphs:
            st.subheader(f'Bolinger Band(볼린저밴드) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- Bolinger Band란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> Upper, Lower : 종가의 이동평균(5일)에서 표준편차(20일)를 더하고 빼준 값.   </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 종가(close)값과 비교하여 어느정도 위치에 있는지 알 수 있습니다.  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 종가가 움직일만한 범위를 나타냅니다. 상한, 하한 사이를 종가가 벗어난 경우 과매수 혹은 과매도라고 판단할 수 있습니다. </h3>

                        </div>
                        ''', unsafe_allow_html=True)
            fig_bol = go.Figure()
            fig_bol.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['upper'], name='Upper',
                                        hovertemplate='날짜: %{x}<br>Upper: %{y:.2f}'))
            fig_bol.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['lower'], name='Lower',
                                        hovertemplate='날짜: %{x}<br>Lower: %{y:.2f}'))
            fig_bol.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Close',
                                        hovertemplate='날짜: %{x}<br>Close: %{y:.2f}'))
            #fig_ma.add_trace(go.Scatter(x=stock_data['Date'], y=[700000] * len(stock_data), name='Lower Bound', line=dict(color='red', dash='dash')))
            #fig_ma.add_trace(go.Scatter(x=stock_data['Date'], y=[300000] * len(stock_data), name='Upper Bound', line=dict(color='green', dash='dash')))
            fig_bol.update_layout(
                autosize=False,
                width=900,
                height=600  
            )
            st.plotly_chart(fig_bol)
            
        if 'OBV' in selected_graphs:
            st.subheader(f'On Balance Volume(온밸런스볼륨) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- OBV(On Balance Volume)이란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 가격 상승 시 거래량을 더해가고 하락시 빼가며 누적된 값으로 가격과 유사하게 움직입니다. </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 가격이 상승하는 경우, OBV도 상승하고, 거래량이 클수록 크게 변동합니다.</h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 상승하는 것을 매집이라고 하며, 매집은 주식이 집중적으로 매입 되는 것을 의미합니다. </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 하락하는 것을 분산이라고 하며, 분산은 주식이 집중적으로 매도 되는 것을 의미합니다. </h3>
                        </div>
                        ''', unsafe_allow_html=True)
            fig_OBV = go.Figure()
            fig_OBV.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['std'], name='STD',
                                        hovertemplate='날짜: %{x}<br>OBV: %{y:.2f}'))
            fig_OBV.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Close',
                                        hovertemplate='날짜: %{x}<br>Close: %{y:.2f}'))
            
            # 주식 종목에 따라 상한선, 하한선 표기할듯.
            #fig_STD.add_trace(go.Scatter(x=stock_data['Date'], y=[700000] * len(stock_data), name='Lower Bound', line=dict(color='red', dash='dash')))
            #fig_STD.add_trace(go.Scatter(x=stock_data['Date'], y=[300000] * len(stock_data), name='Upper Bound', line=dict(color='green', dash='dash')))
            fig_OBV.update_layout(
                autosize=False,
                width=900,
                height=600  
            )
            st.plotly_chart(fig_OBV)
            
        if 'CCI' in selected_graphs:
            st.subheader(f'Commodity Channel Index(상품채널지수) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- CCI(Commodity Channel Index) 란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> Typical Price(고가,저가,종가의 평균)의 이동평균(45일)에 대한 종가의 위치를 나타내는 지표입니다.  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 100보다 크면 과매수, 100보다 작으면 과매도로 볼 수 있습니다. </h3>
                        </div>
                        ''', unsafe_allow_html=True)
            fig_CCI = go.Figure()
            fig_CCI.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['cci'], name='CCI',
                                        hovertemplate='날짜: %{x}<br>CCI: %{y:.2f}'))
            
            # 주식 종목에 따라 상한선, 하한선 표기할듯.
            fig_CCI.add_trace(go.Scatter(x=stock_data['date'], y=[100] * len(stock_data), name='과매수', line=dict(color='red', dash='dash')))
            fig_CCI.add_trace(go.Scatter(x=stock_data['date'], y=[-100] * len(stock_data), name='과매도', line=dict(color='green', dash='dash')))
            fig_CCI.update_layout(
                autosize=False,
                width=900,
                height=600  
            )
            st.plotly_chart(fig_CCI)
            
            
        #RSI MFI와 같이 달기
        if 'RSI' in selected_graphs:
            st.subheader(f'Relative Strength Index(상대강도지수) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- RSI(Relative Strength Index) 란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 현재 가격의 상승 압력과 하락 압력간에 상대적인 강도를 나타냅니다.  </h3>

                        <h3 style="font-size: 14px; font-weight: normal;"> 일정 기간(6일) 동안 주가가 전일 가격에 비해 상승한 변화량과 하락한 변화량의 평균값을 구하여, 그 비율을 사용하여 계산됩니다.</h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 상승한 변화량이 크면 과매수로, 하락한 변화량이 크면 과매도로 판단하는 방식으로,  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 70보다 높으면 과매수, 30보다 낮으면 과매도로 판단할 수 있습니다.  </h3>
                        </div>
                        ''', unsafe_allow_html=True)
            fig_RSI = go.Figure()
            fig_RSI.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['rsi'], name='RSI',
                                        hovertemplate='날짜: %{x}<br>RSI: %{y:.2f}'))
            
            # 주식 종목에 따라 상한선, 하한선 표기할듯.
            fig_RSI.add_trace(go.Scatter(x=stock_data['date'], y=[70] * len(stock_data), name='초과매수', line=dict(color='red', dash='dash')))
            fig_RSI.add_trace(go.Scatter(x=stock_data['date'], y=[30] * len(stock_data), name='초과매도', line=dict(color='green', dash='dash')))
            fig_RSI.update_layout(
                autosize=False,
                width=900,
                height=600  
            )
            st.plotly_chart(fig_RSI)  
            
        if 'MFI' in selected_graphs:
            st.subheader(f'Money Flow Index(상대강도지수) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- MFI(Money Flow Index) 란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 매수와 매도의 상대강도를 측정하기 위해 사용되는 기술적 지표입니다.  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 주식의 가격 뿐만 아니라 거래량도 함께 사용합니다.  </h3>

                        <h3 style="font-size: 14px; font-weight: normal;"> 일정 기간(14일) 동안 주가 변화분에 거래량을 곱한 RMF를 구한 후 양의 RMF와 음의 RMF의 비율로 계산됩니다. </h3>
                        
                        </div>
                        ''', unsafe_allow_html=True)
            fig_MFI = go.Figure()
            fig_MFI.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['mfi'], name='MFI',
                                        hovertemplate='날짜: %{x}<br>MFI: %{y:.2f}'))
            fig_MFI.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['rsi'], name='RSI',
                                        hovertemplate='날짜: %{x}<br>RSI: %{y:.2f}'))
            
            # 주식 종목에 따라 상한선, 하한선 표기할듯.
            fig_MFI.add_trace(go.Scatter(x=stock_data['date'], y=[80] * len(stock_data), name='과잉매도', line=dict(color='red', dash='dash')))
            fig_MFI.add_trace(go.Scatter(x=stock_data['date'], y=[20] * len(stock_data), name='과잉매수', line=dict(color='green', dash='dash')))
            fig_MFI.update_layout(
                autosize=False,
                width=900,
                height=600  
            )
            st.plotly_chart(fig_MFI)  
            
            
        if 'ROC' in selected_graphs:
            st.subheader(f'Rate Of Change(추세지표) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- ROC(Rate Of Change) 란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 기준일(45일 전)의 주가에 비해 주가가 얼마나 변화하였는지를 의미합니다. </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 과거 일정시점의 가격과 현재가격을 비교하여 현재 추세를 알려주는 중요한 지표입니다.  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 값이 0인 경우 기준일과 주가가 동일함을 의미합니다.  </h3>

                        </div>
                        ''', unsafe_allow_html=True)
            fig_ROC = go.Figure()
            fig_ROC.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['roc'], name='ROC',
                                        hovertemplate='날짜: %{x}<br>ROC: %{y:.2f}'))
            fig_ROC.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['ks_roc'], name='ks_ROC',
                                        hovertemplate='날짜: %{x}<br>ks_ROC: %{y:.2f}'))
            #0이상이면 상승 추세
            
            
            fig_ROC.add_trace(go.Scatter(x=stock_data['date'], y=[0] * len(stock_data), name='Bound', line=dict(color='red', dash='dash')))
            fig_ROC.update_layout(
                autosize=False,
                width=900,
                height=600  
            )
            st.plotly_chart(fig_ROC)
            
            
        if 'DP' in selected_graphs:
            st.subheader(f'disparity(이격도) - {stock}')
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 17px; font-weight: normal;">- Disparity(이격도) 란?  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 주가와 이동평균선 사이가 얼마나 떨어져있는지(괴리율)을 나타내는 지표입니다. </h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 종가를 이동평균(20일)으로 나누고 100을 곱하여 계산된 값입니다.</h3>
                        <h3 style="font-size: 14px; font-weight: normal;"> 값이 100인 경우 종가와 이동평균선 값이 일치합니다.</h3>
                        
                        </div>
                        ''', unsafe_allow_html=True)
            fig_DP = go.Figure()
            fig_DP.add_trace(go.Scatter(x=stock_data['date'], y=(stock_data['dp'] + 1) * 100, name='DP',
                                        hovertemplate='날짜: %{x}<br>DP: %{y:.2f}'))
            fig_DP.add_trace(go.Scatter(x=stock_data['date'], y=(stock_data['ks_dp'] +1) * 100, name='ks_DP',
                                        hovertemplate='날짜: %{x}<br>ks_DP: %{y:.2f}'))
            #0이상이면 상승 추세
            
            
            fig_DP.add_trace(go.Scatter(x=stock_data['date'], y=[100] * len(stock_data), name='기준선', line=dict(color='red', dash='dash')))
            fig_DP.update_layout(
                autosize=False,
                width=900,
                height=600  
            )
            st.plotly_chart(fig_DP)
            
        
            
            
        
            
        
            
        
            
        
            
        
        
        
        

if __name__ == '__main__':
    main()