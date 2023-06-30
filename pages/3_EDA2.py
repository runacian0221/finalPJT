import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.graph_objects as go
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
import plotly.express as px
from datetime import datetime, timedelta
from DB.insert_select_db import Database

with open("DB/config.txt", "r") as file:
    exec(file.read())

db = Database(configs)

today = datetime.now().strftime("%Y-%m-%d")
query = f"select * FROM stock WHERE date BETWEEN '2018-01-01' AND '{today}'"
# SQL 쿼리를 실행하여 데이터를 가져옵니다
data, column_names = db.execute_query(query)

# 데이터를 DataFrame으로 변환합니다
df = pd.DataFrame(data, columns=column_names)

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def main():
    global df
    tab11, tab12, tab13, tab14, tab15 = st.tabs(["상관관계 분석","거래량 Top 주가 변동","거래량 변동","등락률 산점도","테스트3"])
    with tab11:
        st.title('- 변수 간 상관관계 히트맵')
        st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;"> 상관관계는 다양한 변수 간의 관계를 알기 쉽습니다.  </h3>
                        <h3 style="font-size: 16px; font-weight: normal;"> 또한 히트맵과 데이터 프레임으로 구현한다면 상관관계를 더욱 쉽게 파악할 수 있습니다.  </h3>
                        <h3 style="font-size: 14px; font-weight: normal;">   </h3>
                        </div>
                        ''', unsafe_allow_html=True)

        # 필요한 변수들 선택
        selected_columns = ['close', 'volume', 'stock_change', 'ma_5', 'fast_k', 'slow_k',
                            'slow_d', 'rsi', 'std', 'upper', 'lower']
        data = df[selected_columns]

        # 상관계수 계산
        correlation = data.corr()

        # 히트맵 생성
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, cmap='coolwarm', annot=True, fmt=".2f", annot_kws={"fontsize": 8}, ax=ax)

        # 그래프 출력
        st.pyplot(fig)
        
        

        st.title('- 변수들간의 상관관계 표')
        # 상관관계가 0.5보다 큰 변수들 출력
        high_correlation = correlation.unstack().reset_index()
        high_correlation = high_correlation[high_correlation['level_0'] != high_correlation['level_1']]
        high_correlation.columns = ['Variable 1', 'Variable 2', 'Correlation']
        high_correlation = high_correlation.dropna()
        high_correlation = high_correlation.drop_duplicates(subset=['Correlation'], keep='first')
        high_correlation = high_correlation.sort_values(by='Correlation', ascending=False)
        high_correlation.reset_index(drop=True, inplace=True)  # 인덱스 초기화

        # 변수들을 정비례 관계와 반비례 관계로 분할
        positive_correlation = high_correlation[high_correlation['Correlation'] > 0]
        negative_correlation = high_correlation[high_correlation['Correlation'] < 0]
        
        col5,col6 = st.columns(2)
        
        with col5:
            st.subheader('모든 변수')
            st.write(high_correlation)
            
        with col6:
            st.write("")
            

        # Streamlit의 columns 레이아웃을 사용하여 col1, col2로 분할하여 출력
        col2, col3 = st.columns(2)

       # with col1:
       #     st.subheader('모든 변수')
       #     st.write(high_correlation)

        with col2:
            st.subheader('정비례 관계')
            st.write(positive_correlation)
        
        with col3:
            st.subheader('반비례 관계')
            st.write(negative_correlation)
            
        
        st.subheader('상관관계에 따른 산점도 그래프')

        

        # 사용자 입력을 받아 상위 순위 선택
        threshold_options = [0.95, 0.96, 0.97, 0.98, 0.99]
        
        threshold_input = col2.selectbox('상관관계의 임계값을 선택하세요:', threshold_options)
        
        # 버튼 클릭 시 산점도 그래프 출력
        if col2.button('그래프 그리기'):
            for _, row in high_correlation.iterrows():
                var1 = row['Variable 1']
                var2 = row['Variable 2']
                corr = row['Correlation']

                if corr >= threshold_input:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=data, x=var1, y=var2)
                    # 반올림 되지 않게 소수점 4자리까지 확인
                    plt.title(f'{var1} vs {var2} (Correlation: {corr:.4f})')
                    plt.xlabel(var1)
                    plt.ylabel(var2)
                    plt.show()
                    st.pyplot(plt)
                
    with tab12:
        selected_stock = st.selectbox("- 종목을 선택해주세요.", df['company_name'].unique())

        # 선택한 종목의 데이터 필터링
        filtered_df = df[df['company_name'] == selected_stock]
        
        st.markdown('<p style="font-size: 24px; font-weight: bold; color: #336699;">- 거래량과 주가의 관계</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 18px;">거래량이 증가하면 주가도 상승하는 경향이 있습니다.</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 18px;">반면, 거래량이 감소하면 주가는 하락하는 경향이 있습니다.</p>', unsafe_allow_html=True)


        st.markdown('<p style="font-size: 24px; font-weight: bold; color: #336699;">- 거래량과 주가의 상관관계 분석</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 18px;">거래량이 많은 날들 이후에 주가는 어떻게 변화했을까요?</p>', unsafe_allow_html=True)

        st.markdown('')
        if filtered_df.empty:
            st.write("선택한 종목에 대한 데이터가 없습니다.")
        else:
            st.write(f"- {selected_stock} 종목의 가장 높은 거래량 Top 5의 주가 변화")

            # 가장 높은 Volume 종목 Top 5
            top_5_stocks = filtered_df.sort_values('volume', ascending=False).head(5)

            for days in [30, 60, 90, 365]:
                top_5_stocks[f'Price {days} Days Later'] = None
                for i, row in top_5_stocks.iterrows():
                    target_date = pd.to_datetime(row['date']) + pd.DateOffset(days=days)  # 날짜를 날짜 형식으로 변환
                    target_date_str = target_date.strftime('%Y-%m-%d')  # 날짜를 문자열로 변환
                    closest_date = filtered_df.loc[filtered_df['date'] >= target_date_str, 'date'].min()
                    if pd.notnull(closest_date):
                        closest_date_price = filtered_df.loc[filtered_df['date'] == closest_date, 'close'].iloc[0]
                        top_5_stocks.loc[i, f'Price {days} Days Later'] = closest_date_price

            st.dataframe(top_5_stocks[['date', 'volume', 'close'] + [f'Price {days} Days Later' for days in [30, 60, 90, 365]]])
            
            
            st.write(f"- {selected_stock} 종목의 30, 60, 90, 365일 뒤의 종가 변화")
            fig = go.Figure()

            for index, row in top_5_stocks.iterrows():
                volume = row['volume']
                price_values = [row[f'Price {days} Days Later'] for days in [30, 60, 90, 365]]
                fig.add_trace(go.Scatter(x=[30, 60, 90, 365], y=price_values, mode='lines', name=f'volume: {volume}'))

            fig.update_layout(xaxis_title="Days Later", yaxis_title="Price")
            #타이틀 설정
            #title=f"- {selected_stock} 종목의 30, 60, 90, 365일 뒤의 종가 변화", 
            st.plotly_chart(fig)




        # 선택한 종목 리스트
        #selected_stocks = ['삼성전자', '현대차', '포스코', '삼성생명', '셀트리온']


        # 선택한 종목들의 데이터 필터링
        #filtered_df = df[df['Name'].isin(selected_stocks)]

        # 중앙값을 계산
        #median_df = filtered_df.groupby('Name')['Close'].median().reset_index()

        # 이상치를 제외한 데이터를 필터링
        #filtered_data = filtered_df[~filtered_df['Close'].isin(median_df['Close'])]

        # Boxplot을 생성하기 위한 데이터 준비
        #boxplot_data = []
        #for name, group in filtered_df.groupby('Name'):
        #    boxplot_data.append(go.Box(
        #        name=name,
        #        y=group['Close'],
        #        boxpoints='outliers',
        #        marker=dict(color='rgb(107, 174, 214)'),
        #        line=dict(color='rgb(8, 81, 156)')
        #    ))

        # 중앙값을 표시하기 위한 Scatter 데이터 준비
        #scatter_data = []
        #for name, group in median_df.groupby('Name'):
        #    scatter_data.append(go.Scatter(
        #        name=name,
        #        x=[name],
        #        y=group['Close'],
        #        mode='markers',
        #        marker=dict(color='red', size=8),
        #    ))

        # 그래프 레이아웃 설정
        #layout = go.Layout(
        #    title='주식 종가의 분포와 중앙값 비교',
        #    xaxis=dict(title='종목'),
        #    yaxis=dict(title='종가'),
        #    showlegend=False
        #)

        # Boxplot과 Scatterplot을 결합한 Figure 생성
        #fig = go.Figure(data=boxplot_data + scatter_data, layout=layout)

        # 그래프 출력
        #st.plotly_chart(fig, use_container_width=True)

        #st.subheader("박스플롯(Box plot)이란? : 중앙값, 이상치, 분포 등을 시각화하고 파악하는데 도움이 되는 그래프.")
        #st.write("1. 2018년부터 2023년까지 셀트리온, 포스코, 현대자동차는 사분위 범위(박스의 높이)를 보면 주식 가격의 변동에 커다란 변화가 있었음을 알 수 있음.")
        #st.write("2. 2018년부터 2023년까지 삼성생명과 삼성전자는 다른 종목들에 비해 (박스플롯)q1~q3까지의 크기가 작게 형성")
        #st.write("3. 삼성생명과 삼성전자가 다른 3종목보다는 안정적이라는 것을 알 수 있음.")
        #st.write("4. 하지만 삼성생명에서 박스의 위와 아래에 있는 가로선을 벗어난 데이터 포인트(이상치)를 발견.")
        #st.write("5. ")
    
    with tab13:
            #(3) 날짜 별, 연도 별 거래량 그래프 확인. ()
        available_names = df['company_name'].unique()
        st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;"> 거래량은 시장의 활발성을 나타냅니다. 얼마나 활발하게 거래되는지 나타냅니다.  </h3>
                        <h3 style="font-size: 16px; font-weight: normal;"> 주식의 흐름을 이해하는 데 도움을 줍니다. 패턴을 분석하면 주심의 흐름을 이해할 수 있습니다.  </h3>
                        <h3 style="font-size: 16px; font-weight: normal;"> 기술적 지표에 자주 쓰이는 중요한 지표입니다.  </h3>
                        </div>
                        ''', unsafe_allow_html=True)

        # 멀티셀렉트 박스로 종목명 선택
        selected_names = st.multiselect('Select Names', available_names)

        # 선택한 종목명에 해당하는 데이터 필터링
        filtered_df = df[df['company_name'].isin(selected_names)] if selected_names else df

        # 날짜를 datetime 형식으로 변환
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])

        # 사용자가 선택한 날짜 범위
        min_date = filtered_df['date'].min().date()
        max_date = filtered_df['date'].max().date()
        start_date = st.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date)
        end_date = st.date_input('End Date', min_value=min_date, max_value=max_date, value=max_date)

        # 날짜 범위에 따른 데이터 필터링
        start_date = datetime(start_date.year, start_date.month, start_date.day)
        end_date = datetime(end_date.year, end_date.month, end_date.day)
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]

        # 날짜 별 거래량 그래프
        fig1 = px.line(filtered_df, x='date', y='volume', color='company_name',
                        title='Volume by Date', labels={'volume': 'volume', 'date': 'date'})
        fig1.update_traces(hovertemplate='Date: %{x}<br>Volume: %{y}')
        
        

        # 연도 별 거래량 그래프
        filtered_df['Year'] = filtered_df['date'].dt.year
        yearly_volumes = filtered_df.groupby(['Year', 'company_name'])['volume'].sum().reset_index()
        fig2 = px.bar(yearly_volumes, x='Year', y='volume', color='company_name',
                        title='년도별 거래량 추이', labels={'volume': 'volume', 'year': 'year'})
        fig2.update_traces(hovertemplate='Year: %{x}<br>Volume: %{y}')

        fig1.update_layout(width=1000, height=600)
        fig2.update_layout(width=800, height=600)

        # 그래프 출력
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)


        
        with tab14:

            # 등락률과 날짜 데이터 추출
            #change = df['Change']
            #dates = pd.to_datetime(df['Date'])
            st.markdown('''
                        <div style="text-align: left; padding: 10px; background-color: #E8F0FE; border-radius: 10px; color: black;">
                        <h3 style="font-size: 16px; font-weight: normal;"> 주식의 변동성을 파악하는 지표입니다.   </h3>
                        <h3 style="font-size: 16px; font-weight: normal;"> 패턴(긍정, 부정)을 분석하여 향후 주가 움직임을 예측하고, 투자 전략을 세울 수 있습니다.  </h3>
                        <h3 style="font-size: 16px; font-weight: normal;"> 리스크와 수익의 관계를 분석하는 데 활용될 수 있습니다  </h3>
                        <h3 style="font-size: 16px; font-weight: normal;"> 보다 높은 등락률을 가진 주식은 일반적으로 더 높은 수익을 제공할 수 있지만, 동시에 더 큰 리스크도 내포하고 있습니다.  </h3>
                        </div>
                        ''', unsafe_allow_html=True)
            
            # 종목명 선택을 위한 드롭다운 메뉴
            selected_name = st.selectbox('종목명 선택', df['company_name'].unique())

            # 기본값 설정
            default_start_date = pd.to_datetime(df['date']).min().to_pydatetime().date()
            start_date = st.date_input('Start Date', value=default_start_date)
            end_date = st.date_input('End Date', value=datetime.now().date())
            start_date = start_date.strftime('%Y-%m-%d')
            end_date = end_date.strftime('%Y-%m-%d')

            # 선택된 종목명과 날짜에 해당하는 데이터 추출
            selected_data = df[(df['company_name'] == selected_name) & (df['date'] >= start_date) & (df['date'] <= end_date)]

            # 그래프 생성
            fig = go.Figure()

            # 선택된 종목명과 날짜에 해당하는 등락률 산점도 추가
            positive_changes = selected_data[selected_data['stock_change'] > 0]
            fig.add_trace(go.Scatter(
                x=positive_changes['date'],
                y=positive_changes['stock_change'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Positive Changes'
            ))
            
            # 음수 등락률 산점도 추가
            negative_changes = selected_data[selected_data['stock_change'] < 0]
            fig.add_trace(go.Scatter(
                x=negative_changes['date'],
                y=negative_changes['stock_change'],
                mode='markers',
                marker=dict(color='blue', size=8),
                name='Negative Changes'
            ))
                        
            

            # 레이아웃 설정
            fig.update_layout(
                title="등락률",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Change"),
                hovermode="x",
                width=900,
                height=500
            )

            # 그래프 출력
            st.plotly_chart(fig)
            
            positive_count = len(positive_changes)
            negative_count = len(negative_changes)
            total_count = positive_count + negative_count
            positive_ratio = positive_count / total_count
            negative_ratio = negative_count / total_count

            # 긍정 부정 비율 데이터프레임 생성
            ratio_data = pd.DataFrame({
                'Type': ['Positive', 'Negative'],
                'Count': [positive_count, negative_count],
                'Ratio': [positive_ratio, negative_ratio]
            })
            st.dataframe(ratio_data)

if __name__ == '__main__':
    main()