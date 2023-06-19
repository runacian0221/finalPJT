import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# 시퀀스 데이터셋을 생성하는 함수를 정의
# 특징과 레이블을 입력받아, 주어진 윈도우 크기에 따라 시퀀스 데이터셋을 생성
def make_sequene_dataset(feature, label, window_size):
    feature_list = []
    label_list = []
    for i in range(len(feature)-window_size):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])
    return np.array(feature_list), np.array(label_list)

# 데이터 정규화를 위한 MinMaxScaler를 생성 (0~1사이값)
scaler = MinMaxScaler()

df = pd.read_csv('포스코주가데이터2018-2023.csv', encoding='UTF-8')
df.drop(['stock_change','fast_k','slow_k','slow_d','rsi','std','upper','lower','open','high','low','volume','stock_id'], axis=1, inplace=True)
df['ma_20'] = df['close'].rolling(window=20).mean().round(-2) # 20일 이동평균선
df['ma_60'] = df['close'].rolling(window=60).mean().round(-2) # 60일 이동평균선

df.to_csv('포스코주가데이터2018-2023.csv', index=False, encoding='UTF-8')

for i in range(20):
    df = pd.read_csv('포스코주가데이터2018-2023.csv', encoding='UTF-8')
    # null값 채우기
    df.fillna(0, inplace=True)
    # 정보 출력
    print(df.info())

    # 정규화할 컬럼을 선택
    scale_cols = ['close', 'ma_5', 'ma_20', 'ma_60']

    # 선택한 컬럼을 정규화하고, 데이터프레임으로 변환
    scaled_df = scaler.fit_transform(df[scale_cols])
    scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)
    print(scaled_df)

    # 특징과 레이블로 사용할 컬럼을 선택
    feature_cols = ['close', 'ma_5', 'ma_20', 'ma_60']
    label_cols = ['close']

    label_df = pd.DataFrame(scaled_df, columns=label_cols)
    feature_df = pd.DataFrame(scaled_df, columns=feature_cols)

    # 데이터프레임을 numpy 배열로 변환
    label_np = label_df.to_numpy()
    feature_np = feature_df.to_numpy()

    # 시퀀스 윈도우 크기를 설정하고, 데이터셋을 생성
    window_size = 60
    X, Y = make_sequene_dataset(feature_np, label_np, window_size)

    # 테스트를 위해 마지막 200개의 데이터를 분리
    split = -200
    x_train = X[0:split]
    y_train = Y[0:split]

    x_test = X[split:]
    y_test = Y[split:]

    # LSTM 모델을 생성
    model = Sequential()
    model.add(LSTM(256, activation='tanh', input_shape=x_train[0].shape))
    model.add(Dense(1, activation='linear'))
    model.summary()

    # 모델을 컴파일
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # 조기 종료를 위한 EarlyStopping을 설정
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    # 모델을 학습
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=16, callbacks=[early_stop])

    # 'close' 컬럼만을 위한 스케일러를 생성합니다.
    close_scaler = MinMaxScaler()
    close_scaler.fit(df[['close']])

    # 가장 최근 데이터(윈도우 크기만큼)를 가져옵니다.
    last_sequence = np.array(feature_np[-window_size:])

    # 모델의 입력 형태에 맞게 배열을 변형합니다.
    last_sequence = last_sequence.reshape((1, window_size, -1))

    # 모델을 이용하여 예측을 수행합니다.
    pred = model.predict(last_sequence)

    # 결과를 정규화를 해제하여 실제 주가 범위로 변환합니다.
    pred_price = close_scaler.inverse_transform(pred)

    print("다음날의 예측 주가는", pred_price, "입니다.")

    # 마지막 거래일로부터 하루 뒤 날짜를 얻기
    last_date = datetime.strptime(df.iloc[-1]['date'], '%Y-%m-%d')
    next_date = last_date

    # 다음날이 주말인지 확인하고, 주말이면 다음 주의 첫 영업일로 날짜를 설정
    while next_date.weekday() > 4:  # 5: 토요일, 6: 일요일
        next_date += timedelta(days=1)

    next_date += timedelta(days=1)  # 다음날로 설정

    # 예측한 종가와 필요한 정보로 새로운 행 생성
    new_row = {
        'company_id': 1,
        'company_name': 'POSCO',
        'date': next_date.strftime('%Y-%m-%d'), 
        'close': pred_price[0][0], 
        'ma_5': np.nan,
        'ma_20': np.nan,
        'ma_60': np.nan,
    }

    # 새로운 행 추가
    df = df.append(new_row, ignore_index=True)

    # 이동 평균 구하기
    df['close'] = df['close'].rolling(window=5).mean().round(-2)
    df['ma_5'] = df['close'].rolling(window=5).mean().round(-2)
    df['ma_20'] = df['close'].rolling(window=20).mean().round(-2)
    df['ma_60'] = df['close'].rolling(window=60).mean().round(-2)

    # 데이터 저장하기
    df.to_csv('포스코주가데이터2018-2023.csv', index=False, encoding='UTF-8')