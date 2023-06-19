import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = pd.read_csv('포스코주가데이터2018-2023.csv', encoding='UTF-8')
df.drop(['stock_change','fast_k','slow_k','slow_d','rsi','std','upper','lower','open','high','low','volume','stock_id'], axis=1, inplace=True)
df['ma_5'] = df['close'].rolling(window=5).mean().round(-2)
df['ma_20'] = df['close'].rolling(window=20).mean().round(-2) # 20일 이동평균선
df['ma_60'] = df['close'].rolling(window=60).mean().round(-2) # 60일 이동평균선
df.fillna(0, inplace=True)

df.rename(columns={'date':'ds', 'close':'y'}, inplace=True)
df_prophet = Prophet(changepoint_prior_scale=0.15, daily_seasonality=True)
df_prophet.add_regressor('ma_5')
df_prophet.add_regressor('ma_20')
df_prophet.add_regressor('ma_60')
df_prophet.fit(df)

fcast_time = 90
df_forecast = df_prophet.make_future_dataframe(periods = fcast_time, freq = 'D')

df_forecast['ma_5'] = df['ma_5'].append(pd.Series(df['ma_5'].iloc[-1], index=np.arange(fcast_time))).reset_index(drop=True)
df_forecast['ma_20'] = df['ma_20'].append(pd.Series(df['ma_20'].iloc[-1], index=np.arange(fcast_time))).reset_index(drop=True)
df_forecast['ma_60'] = df['ma_60'].append(pd.Series(df['ma_60'].iloc[-1], index=np.arange(fcast_time))).reset_index(drop=True)

# 검은 점 : 원래 데이터 포인트
# 푸른 선 : 예측된 값
# 푸른 영역 : 예측값의 신뢰구간
forecast = df_prophet.predict(df_forecast)
fig1 = df_prophet.plot(forecast)

fig1.savefig("forecast.png")
