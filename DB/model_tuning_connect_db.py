import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import FinanceDataReader as fdr

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from hyperopt import fmin, tpe, space_eval, Trials, hp, rand, anneal

import xgboost as xgb
from xgboost import plot_importance

import pickle
# import pymysql
# from insert_select_db import Database
# from insert_data_processor_ver1 import DataProcessor

# # config.txt에서 설정값을 읽기
# with open("config.txt", "r") as file:
#     exec(file.read())

# # Database 객체를 생성
# db = Database(configs)

# # DataProcessor 객체를 생성
# processor = DataProcessor(db)

stock_data = pd.read_csv('/home/ubuntu/workspace/csv/stock_data(5y).csv')

# stock_data = pd.read_csv('./Stock_price/stock_data.csv')
# 코스피 데이터 생성
ks11 = fdr.DataReader('KS11', '2012-06-01')
ks11['ks_fast'] = ((ks11['Close'] - ks11['Low'].rolling(14).min()) / (ks11['High'].rolling(14).max() - ks11['Low'].rolling(14).min())) * 100

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    
def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

def adf_kpss_test_for_cols(df, columns):
    print('분석 대상 :', df['Code'].values[0])
    for col in columns:
        print("--------------------{0:^10}---------------------".format(col))
        adfuller_test = adfuller(df[col], autolag= "AIC")
        print("ADF test statistic: {}".format(adfuller_test[0]))
        print("p-value: {}".format(adfuller_test[1]))
        kpsstest = kpss(df[col], regression="c", nlags="auto")
        print("KPSS test statistic: {}".format(kpsstest[0]))
        print("p-value: {}".format(kpsstest[1]))

        # 5일 이동평균
def add_ma_5(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean().values
    return

# 볼린저 밴드
def add_bollinger_bands(df, window=20, std=2):
    # 표준편차 계산
    df['STD'] = df['Close'].rolling(window=window).std()
    # 상단 볼린저 밴드 계산
    df['Upper'] = df['MA_5'] + (std * df['STD'])
    # 하단 볼린저 밴드 계산
    df['Lower'] = df['MA_5'] - (std * df['STD'])
    return

def add_Out(df):
    df['Out'] = ((df['Upper'] - df['Close'] < 0) + 0) + -((df['Lower'] > df['Close']) + 0)
    return

# 이동평균
def add_moving_avg(df, window=20):
    df['MA'] = df['Close'].rolling(window=window).mean().values
    return

# 거래량 관련
def add_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i-1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i-1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return

def add_cci(df, window=20):
    m = (df['High'] + df['Low'] + df['Close']) / 3
    n = m.rolling(window).mean()
    # d = (abs(m - n)).rolling(window).mean()
    d = m.rolling(window).apply(lambda x: pd.Series(x).sub(x.mean()).abs().mean())
    # d = m.rolling(window).apply(lambda x : pd.Series(x).mad())
    df['CCI'] = (m - n) / (0.015 * d)
    return

def add_stochastic_fast_k(df, window=14):
    df['fast_k'] = ((df['Close'] - df['Low'].rolling(window).min()) / (df['High'].rolling(window).max() - df['Low'].rolling(window).min())) * 100
    return

def add_stochastic_fast_d(df, window=9):
    df['fast_d'] = df['fast_k'].rolling(window=window).mean().values
    return

def add_ROC(df, window=1):
    df['ROC'] = df.Close.diff(window) / df.Close.shift(window)
    return

def add_RSI(df, period=14):
    delta = df["Close"].diff() # 종가의 차이를 계산
    up, down = delta.copy(), delta.copy() # 상승분과 하락분을 따로 계산하기 위해 복사
    up[up < 0] = 0 # 상승분, U
    down[down > 0] = 0 # 하락분, D
    _gain = up.ewm(com=(period - 1), min_periods=period).mean() # AU(U값의 평균)
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean() # DU(D값의 평균)
    RS = _gain / _loss
    rsi = pd.Series(100 - (100 / (1 + RS)), name="RSI")
    df['RSI'] = rsi
    return

def add_RSI_DP(df, window=2):
    df['RSI_DP'] = np.round(df['RSI'] / df['RSI'].rolling(window=window).mean(), 4) - 1
    return

def add_DP(df, window=20):
    df['DP'] = np.round(df['Close'] / df['Close'].rolling(window=window).mean(), 4) - 1
    return

def add_MFI(df, period = 14): 
	# 우선적으로 각 기간에 맞게 평균 가격(TP)을 구합니다. 
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3  # TP
    money_flow = typical_price * df['Volume']
    positive_flow =[] 
    negative_flow = []
    
    # 반복문을 돌면서 기간별 양의 RMF, 음의 RMF를 구현합니다.
    # Loop through the typical price 
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]: 
            positive_flow.append(money_flow[i-1])
            negative_flow.append(0) 
        elif typical_price[i] < typical_price[i-1]:
            negative_flow.append(money_flow[i-1])
            positive_flow.append(0)
        else: 
            positive_flow.append(0)
            negative_flow.append(0)
    
    positive_mf = []
    negative_mf = [] 
    # 기간동안 평균 가격들의 분류가 끝난 경우, RMF 계산식을 이용해 평균 가격과 당일 거래량을 곱합니다. 
    # Get all of the positive money flows within the time period
    for i in range(period-1, len(positive_flow)):
        positive_mf.append(sum(positive_flow[i+1-period : i+1]))
    # Get all of the negative money flows within the time period  
    for i in range(period-1, len(negative_flow)):
        negative_mf.append(sum(negative_flow[i+1-period : i+1]))
    # for i in range(len(positive_mf)):
    #     if positive_mf[i] == 0 and negative_mf[i] == 0:
    #         negative_mf[i] = 1
    # MFR을 계산합니다. 그 후 MFI를 구합니다. 
    mfi = list(100 * (np.array(positive_mf) / (np.array(positive_mf)  + np.array(negative_mf))))
    mfi = list(np.repeat(np.nan,len(df)-len(mfi))) + mfi
    df['MFI'] = mfi
    return

# 종목 데이터에 코스피 데이터를 날짜 기준으로 병합
def add_ks_data(df, ks11, window=1):
    # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    ks11['ks_ROC'] = np.round(ks11.Close.diff(window) / ks11.Close.shift(window), 4)
    ks11['ks_DP'] = np.round(ks11.Close / ks11.Close.rolling(window=window).mean().values, 4) - 1
    data = pd.merge(df, ks11[['ks_ROC', 'ks_DP', 'ks_fast']], left_on='Date', right_index=True)
    return data

def labeling(data, a_window=5, b_window=5):
    a_name = 'MA_'+str(a_window)
    b_name = 'MA_'+str(b_window)
    data[a_name] = data['Close'].rolling(window=a_window).mean().values
    data[b_name] = data['Close'].rolling(window=b_window).mean().values
    data['label'] = None
    for i in range(len(data)-b_window):
        data.loc[i, 'label'] = 1 if data.loc[i, a_name] <= data.loc[i+b_window, b_name] else 0
    return

def add_FROC(df):
    df['FROC_60'] = df.Close.shift(-60) / df.Close - 1
    df['FROC_30'] = df.Close.shift(-30) / df.Close - 1
    df['FROC_10'] = df.Close.shift(-10) / df.Close - 1
    df['FROC_5'] = df.Close.shift(-5) / df.Close - 1
    return

def add_features(df, window_params, label_window):
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)
    add_ma_5(df)
    add_bollinger_bands(df)
    add_Out(df)
    add_obv(df)
    add_moving_avg(df, window=int(window_params['MA']))
    add_cci(df, window=int(window_params['CCI']))
    add_stochastic_fast_k(df, window=int(window_params['fast_k']))
    add_stochastic_fast_d(df, window=int(window_params['fast_d']))
    add_ROC(df, window=int(window_params['ROC']))
    add_RSI(df, int(window_params['RSI']))
    add_RSI_DP(df, int(window_params['RSI_DP']))
    add_MFI(df, int(window_params['MFI']))
    add_DP(df, int(window_params['DP']))
    
    # add_FROC(df)
    # labeling(df, *label_window)

    # df.dropna(axis=0, inplace=True)
    # df.reset_index(drop=True, inplace=True)
    return

def feature_scaling(df, scaling_features):
    rbc = RobustScaler()
    mmc = MinMaxScaler()
    df[scaling_features['rbc']] = rbc.fit_transform(df[scaling_features['rbc']])
    df[scaling_features['mmc']] = mmc.fit_transform(df[scaling_features['mmc']])
    return df

def make_data(stock_dict, window_params, date_range, label_window, scaling_features=None, model_features=None, test_features=None, get_scaler=False):
    data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()
    scaled_data = pd.DataFrame()
    scaled_val_data = pd.DataFrame()
    scaled_test_data = pd.DataFrame()
    scalers = {}
    # 종목 코드에 대해 데이터를 생성해 data, val_data, test_data에 합치기
    for name, code in stock_dict.items():
        # 종목 코드, 정한 기간에 대한 주가 데이터
        df = fdr.DataReader(code, *date_range)
        df = df.loc[~(df[['Open', 'High', 'Low', 'Close', 'Volume']] == 0).any(axis=1)]
        df['company_name'] = name
        df['Code'] = code
        # 입력한 윈도우 크기를 사용하여 주가 데이터 기반으로 여러 지표를 생성, 정해둔 방식으로 라벨링
        add_features(df, window_params=window_params, label_window=label_window)
        # 데이터에 코스피 데이터 날짜 기준으로 병합
        df = add_ks_data(df, ks11, window=int(window_params['ks_ROC']))
        # 널값 제거
        # print(df.isnull().sum())
        df.dropna(axis=0, inplace=True)
        # 라벨 타입 변경
        if label_window:
            df['label'] = df['label'].astype(int)
        # 분석기간부터의 데이터만 남김
        df = df[df.Date >= date_range[0]]
        df.reset_index(drop=True, inplace=True)

        # 데이터 검증 True인 경우 -> 데이터 정상성 검증
        if test_features:
            adf_kpss_test_for_cols(df, test_features)
        
        # 모델 학습에 사용할 지표가 지정된 경우 -> data, test_data에 기간을 나누어 합침
        if model_features:
            train_length = int(len(df)*0.7)
            val_length = int(len(df)*0.85)

            train_df = df.loc[:train_length, :].copy()
            val_df = df.loc[train_length+1:val_length, :].copy()
            test_df = df.loc[val_length+1:, :].copy()
            
            # train_length = int(len(df)*0.3)
            # val_length = int(len(df)*0.15)

            # train_df = df.loc[train_length:, :].copy()
            # val_df = df.loc[val_length+1:train_length-1, :].copy()
            # test_df = df.loc[:val_length, :].copy()

            data = pd.concat([data, train_df], axis=0)
            val_data = pd.concat([val_data, val_df], axis=0)
            test_data = pd.concat([test_data, test_df], axis=0)

            # 스케일링 대상 feature가 지정된 경우 -> 대상에 대해 스케일링 진행
            if scaling_features:
                rbc = RobustScaler()
                mmc = MinMaxScaler()

                if len(train_df[scaling_features['rbc']])==0:
                    print(df)
                    print(train_length, val_length, len(df))
                    print('train')
                if len(val_df[scaling_features['rbc']])==0:
                    print('val')
                if len(test_df[scaling_features['rbc']])==0:
                    print('test')
                
                train_df[scaling_features['rbc']] = rbc.fit_transform(train_df[scaling_features['rbc']])
                val_df[scaling_features['rbc']] = rbc.transform(val_df[scaling_features['rbc']])
                test_df[scaling_features['rbc']] = rbc.transform(test_df[scaling_features['rbc']])

                scalers[name] = rbc

                train_df[scaling_features['mmc']] = mmc.fit_transform(train_df[scaling_features['mmc']])
                val_df[scaling_features['mmc']] = mmc.transform(val_df[scaling_features['mmc']])
                test_df[scaling_features['mmc']] = mmc.transform(test_df[scaling_features['mmc']])

                scaled_data = pd.concat([scaled_data, train_df], axis=0)
                scaled_val_data = pd.concat([scaled_val_data, val_df], axis=0)
                scaled_test_data = pd.concat([scaled_test_data, test_df], axis=0)
                pass
            continue
        # model_features=None인 경우 하나의 DateFrame에 합쳐서 return
        data = pd.concat([data, df], axis=0)
    
    data.reset_index(drop=True, inplace=True)
    # model_features가 지정된 경우 데이터를 split하여 return
    if model_features and scaling_features:
        original_data = pd.concat([data, val_data, test_data], axis=0)
        original_data.reset_index(drop=True, inplace=True)
        
        scaled_data.reset_index(drop=True, inplace=True)
        scaled_val_data.reset_index(drop=True, inplace=True)
        scaled_test_data.reset_index(drop=True, inplace=True)

        train_X = scaled_data[model_features].values
        val_X = scaled_val_data[model_features].values
        test_X = scaled_test_data[model_features].values

        train_y = scaled_data['label'].values
        val_y = scaled_val_data['label'].values
        test_y = scaled_test_data['label'].values
        if get_scaler:
            return original_data, scalers, train_X, val_X, test_X, train_y, val_y, test_y
        return original_data, train_X, val_X, test_X, train_y, val_y, test_y
    return data

# make_data 입력값 예시

# 종목 이름과 코드 정보가 담긴 dictionary
stock_dict = {
    '포스코': '005490',
    '삼성전자': '005930',
    '현대차': '005380',
    '셀트리온': '068270',
    '삼성생명': '032830'
}

# 각 지표의 window 크기를 지정하는 dictionary
window_params = {
    'MA' : 5, 
    'CCI' : 20,
    'fast_k' : 14, 
    'fast_d' : 9, 
    'ROC' : 5, 
    'ks_ROC' : 5,
    'RSI' : 14,
    'MFI' : 14
}

# 라벨링 윈도우 크기 지정 -> 5일 이동평균과 20일 이후 20일 이동평균을 비교
label_window = (5, 20)

# 분석 대상 기간 설정
date_range = ['2017-01-01', '2023-05-31']

# 스케일링 필요시 시정 -> rbc, mmc 나누어 지정
scaling_features = {'rbc' : ['Volume', 'Change', 'ROC', 'ks_ROC'],
                    'mmc' : ['Open', 'High', 'Low', 'Close', 'MA_5',
                                'STD', 'Upper', 'Lower', 'OBV', 'MA', 'CCI', 'fast_k', 'fast_d',
                                'RSI', 'MFI', 'MA_20', 'ks_fast']}

# 모델 학습에 사용할 지표 지정 -> 지정 시 train, test split하여 return
model_features = ['Volume', 'Change', 'ROC', 'ks_ROC', 'MA_5',
                    'STD', 'Upper', 'Lower',
                    'OBV', 'MA', 'CCI', 'fast_k', 'fast_d',
                    'RSI', 'MFI', 'MA_20', 'ks_fast']

def objective(params, eval=False):
    # params에서 윈도우 크기 불러오기
    window_params = {
        'MA': int(params['MA']),
        'CCI': int(params['CCI']),
        'fast_k': int(params['fast_k']),
        'fast_d': int(params['fast_d']),
        'ROC': int(params['ROC']),
        'ks_ROC': int(params['ks_ROC']),
        'RSI': int(params['RSI']),
        'MFI': int(params['MFI']),
        'RSI_DP' : int(params['RSI_DP']),
        'DP' : int(params['DP'])
    }
    
    # 데이터셋 생성
    original_data, train_X, val_X, test_X, train_y, val_y, test_y = make_data(stock_dict, window_params, date_range, label_window, scaling_features=scaling_features, model_features=model_features)
    negative_rate = 1 - np.round(np.concatenate([train_y, val_y, test_y], axis=0).sum() / len(np.concatenate([train_y, val_y, test_y], axis=0)), 4)

    # 모델 생성
    model = xgb.XGBClassifier(
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        gamma=params['gamma'],
        reg_lambda=params['reg_lambda'],
        reg_alpha=params['reg_alpha'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        scale_pos_weight=negative_rate,
        random_state=0
    )
    model.fit(train_X, train_y)
    y_pred = model.predict(val_X)
    accuracy = accuracy_score(val_y, y_pred)

    if eval:
        model.fit(np.concatenate([train_X, val_X], axis=0), np.concatenate([train_y, val_y], axis=0))
        print('train + valid 정확도', accuracy_score(np.concatenate([train_y, val_y], axis=0), model.predict(np.concatenate([train_X, val_X], axis=0))))
        print('test          정확도', accuracy_score(test_y, model.predict(test_X)))
        return model, original_data, train_X, val_X, test_X, train_y, val_y, test_y
    return -accuracy

def corr_objective(params, eval=False):
    # params에서 윈도우 크기 불러오기
    window_params = {
        'MA': int(params['MA']),
        'CCI': int(params['CCI']),
        'fast_k': int(params['fast_k']),
        'fast_d': int(params['fast_d']),
        'ROC': int(params['ROC']),
        'ks_ROC': int(params['ks_ROC']),
        'RSI': int(params['RSI']),
        'MFI': int(params['MFI']),
        'RSI_DP' : int(params['RSI_DP']),
        'DP' : int(params['DP'])
    }
    
    # 데이터셋 생성
    original_data, train_X, val_X, test_X, train_y, val_y, test_y = make_data(stock_dict, window_params, date_range, label_window, scaling_features=scaling_features, model_features=model_features)
    negative_rate = 1 - np.round(np.concatenate([train_y, val_y, test_y], axis=0).sum() / len(np.concatenate([train_y, val_y, test_y], axis=0)), 4)

    # 모델 생성
    model = xgb.XGBClassifier(
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        gamma=params['gamma'],
        reg_lambda=params['reg_lambda'],
        reg_alpha=params['reg_alpha'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        scale_pos_weight=negative_rate,
        random_state=0
    )
    model.fit(train_X, train_y)
    y_prob = model.predict_proba(val_X)[:, 1]
    froc = np.mean(original_data.loc[len(train_y):len(train_y)+len(val_y)-1, ['FROC_10', 'FROC_30']].values, axis=1)
    corr = np.corrcoef(y_prob, froc)[0, 1]
    if eval:
        model.fit(np.concatenate([train_X, val_X], axis=0), np.concatenate([train_y, val_y], axis=0))
        print('train + valid 정확도', accuracy_score(np.concatenate([train_y, val_y], axis=0), model.predict(np.concatenate([train_X, val_X], axis=0))))
        print('test          정확도', accuracy_score(test_y, model.predict(test_X)))
        print(np.corrcoef(model.predict_proba(test_X)[:, 1], np.mean(original_data.loc[len(val_y)+len(train_y):, ['FROC_10', 'FROC_30']].values, axis=1))[0, 1])
        return model, original_data, train_X, val_X, test_X, train_y, val_y, test_y
    return -corr

def loss_objective(params, eval=False):
    # params에서 윈도우 크기 불러오기
    window_params = {
        'MA': int(params['MA']),
        'CCI': int(params['CCI']),
        'fast_k': int(params['fast_k']),
        'fast_d': int(params['fast_d']),
        'ROC': int(params['ROC']),
        'ks_ROC': int(params['ks_ROC']),
        'RSI': int(params['RSI']),
        'MFI': int(params['MFI']),
        'RSI_DP' : int(params['RSI_DP']),
        'DP' : int(params['DP'])
    }
    
    # 데이터셋 생성
    original_data, train_X, val_X, test_X, train_y, val_y, test_y = make_data(stock_dict, window_params, date_range, label_window, scaling_features=scaling_features, model_features=model_features)
    negative_rate = 1 - np.round(np.concatenate([train_y, val_y, test_y], axis=0).sum() / len(np.concatenate([train_y, val_y, test_y], axis=0)), 4)

    # 모델 생성
    model = xgb.XGBClassifier(
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        gamma=params['gamma'],
        reg_lambda=params['reg_lambda'],
        reg_alpha=params['reg_alpha'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        scale_pos_weight=negative_rate,
        random_state=0
    )
    model.fit(train_X, train_y)
    proba = model.predict_proba(val_X)[:, 1]
    loss = -np.mean(val_y * np.log(proba) + (1 - val_y) * np.log(1 - proba))

    if eval:
        model.fit(np.concatenate([train_X, val_X], axis=0), np.concatenate([train_y, val_y], axis=0))
        print('train + valid 정확도', accuracy_score(np.concatenate([train_y, val_y], axis=0), model.predict(np.concatenate([train_X, val_X], axis=0))))
        print('test          정확도', accuracy_score(test_y, model.predict(test_X)))
        return model, original_data, train_X, val_X, test_X, train_y, val_y, test_y
    return loss

def roc_objective(params, eval=False):
    # params에서 윈도우 크기 불러오기
    window_params = {
        'MA': int(params['MA']),
        'CCI': int(params['CCI']),
        'fast_k': int(params['fast_k']),
        'fast_d': int(params['fast_d']),
        'ROC': int(params['ROC']),
        'ks_ROC': int(params['ks_ROC']),
        'RSI': int(params['RSI']),
        'MFI': int(params['MFI']),
        'RSI_DP' : int(params['RSI_DP']),
        'DP' : int(params['DP'])
    }
    
    # 데이터셋 생성
    original_data, train_X, val_X, test_X, train_y, val_y, test_y = make_data(stock_dict, window_params, date_range, label_window, scaling_features=scaling_features, model_features=model_features)
    negative_rate = 1 - np.round(np.concatenate([train_y, val_y, test_y], axis=0).sum() / len(np.concatenate([train_y, val_y, test_y], axis=0)), 4)

    # 모델 생성
    model = xgb.XGBClassifier(
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        gamma=params['gamma'],
        reg_lambda=params['reg_lambda'],
        reg_alpha=params['reg_alpha'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        scale_pos_weight=negative_rate,
        random_state=0
    )
    model.fit(train_X, train_y)
    proba = model.predict_proba(val_X)[:, 1]
    test_fpr, test_tpr, _ = roc_curve(val_y, proba)

    test_auc = auc(test_fpr, test_tpr)

    if eval:
        model.fit(np.concatenate([train_X, val_X], axis=0), np.concatenate([train_y, val_y], axis=0))
        print('train + valid 정확도', accuracy_score(np.concatenate([train_y, val_y], axis=0), model.predict(np.concatenate([train_X, val_X], axis=0))))
        print('test          정확도', accuracy_score(test_y, model.predict(test_X)))
        return model, original_data, train_X, val_X, test_X, train_y, val_y, test_y
    return -test_auc

# trials에 사용된 parameter 값들의 분포 시각화
con_params = ['learning_rate', 'gamma', 'reg_lambda', 'reg_alpha', 'subsample', 'colsample_bytree', 'n_estimators', ]
int_params = ['max_depth', 'CCI', 'fast_k', 'fast_d', 'ROC', 'ks_ROC', 'RSI', 'MFI']
def param_distribute(trials):
    # 정수형 하이퍼파라미터의 분포를 막대 그래프로 표현
    # for parameter in int_params:
    #     parameter_values = trials.idxs_vals[1][parameter]
    #     value_range = range(int(min(parameter_values)), int(max(parameter_values))+1)
    #     parameter_counts = [parameter_values.count(i) for i in value_range]

    #     plt.bar(value_range, parameter_counts)
    #     plt.xlabel(parameter)
    #     plt.ylabel('Counts')
    #     plt.title('Distribution of '+parameter)
    #     plt.show()

    # 실수형 하이퍼파라미터의 분포를 히스토그램으로 표현
    for parameter in con_params:
        parameter_values = trials.idxs_vals[1][parameter]
        plt.hist(parameter_values, bins=40)
        plt.xlabel(parameter)
        plt.ylabel('Frequency')
        plt.title('Distribution of '+parameter)
        plt.show()
    return

# 모델의 auc-roc 시각화
def AUC_ROC(model, test_X, test_y):
    test_y_pred_proba = model.predict_proba(test_X)[:, 1]  # 클래스 1의 예측 확률만 선택

    test_fpr, test_tpr, _ = roc_curve(test_y, test_y_pred_proba)

    test_auc = auc(test_fpr, test_tpr)
    # print(train_auc, test_auc)
    plt.figure()
    plt.plot(test_fpr, test_tpr, color='navy', lw=2, label='Test ROC curve (AUC = %0.2f)' % test_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    return

def show_imp(xgb_model):
    fig, ax = plt.subplots(figsize = (10, 6))
    plot_importance(xgb_model, ax = ax)
    plt.show()
    return

def final_objective(params, eval=False):
    model = xgb.XGBClassifier(
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        gamma=params['gamma'],
        reg_lambda=params['reg_lambda'],
        reg_alpha=params['reg_alpha'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        scale_pos_weight=negative_rate,
        random_state=0
    )
    model.fit(train_X, train_y)
    y_pred = model.predict(val_X)
    accuracy = accuracy_score(val_y, y_pred)

    if eval:
        model.fit(np.concatenate([train_X, val_X], axis=0), np.concatenate([train_y, val_y], axis=0))
        print('train + valid 정확도', accuracy_score(np.concatenate([train_y, val_y], axis=0), model.predict(np.concatenate([train_X, val_X], axis=0))))
        print('test          정확도', accuracy_score(test_y, model.predict(test_X)))
        return model
    return -accuracy

# 시가총액 상위 35개 종목 데이터 사용
stocks = fdr.StockListing('KOSPI')
stock_name = stocks[['Name', 'Code']]
stock_dict = dict(stock_name.values[:35])
# 코스피 데이터
ks11 = fdr.DataReader('KS11', '2012-06-01')
ks11['ks_fast'] = ((ks11['Close'] - ks11['Low'].rolling(14).min()) / (ks11['High'].rolling(14).max() - ks11['Low'].rolling(14).min())) * 100
# 10년치 데이터 사용
date_range = ['2013-01-01', '2023-05-31']
# rbc만 사용
scaling_features = {'rbc' : ['ROC', 'ks_ROC', 'RSI_DP', 'DP', 'ks_DP', 'CCI', 'fast_k', 'fast_d', 'RSI', 'MFI', 'ks_fast'],
                    'mmc' : ['Close']}
model_features = ['ROC', 'ks_ROC', 'CCI', 'fast_k', 'fast_d', 'RSI', 'MFI', 'ks_fast', 'RSI_DP', 'DP', 'ks_DP']
# xgbclassifier paramer tuning 위한 space
space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 50, 1000, 1),
        'gamma': hp.uniform('gamma', 0, 10),
        'reg_lambda': hp.uniform('reg_lambda', 0, 10),
        'reg_alpha': hp.uniform('reg_alpha', 0, 10),
        'subsample': hp.uniform('subsample', 0, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0, 1)
}
# 상관관계 결과를 참고하여 window 임의 선택
window_params = {
    'MA': 20,
    'CCI': 45,
    'fast_k': 9,
    'fast_d': 5,
    'ROC': 45,
    'ks_ROC': 45,
    'RSI': 6,
    'MFI': 14,
    'RSI_DP' : 5,
    'DP' : 20
    }

label_window = (1, 60)
data, scalers, train_X, val_X, test_X, train_y, val_y, test_y = make_data(stock_dict, window_params, date_range, label_window, scaling_features=scaling_features, model_features=model_features, get_scaler=True)
negative_rate = 1 - np.round(train_y.sum() / len(train_y), 6)

trials = Trials()
best = fmin(final_objective, space, algo=anneal.suggest, max_evals=1000, trials=trials, rstate=np.random.default_rng(42))

print('Best Hyperparameters:')
print(space_eval(space, best))

final_model = final_objective(space_eval(space, best), eval=True)

# 스케일러 저장
with open('kjk/scalers_10.pkl', 'wb') as f:
    pickle.dump(scalers, f)

# 모델 저장
with open('kjk/xgb_model.pkl', 'wb') as f:
    pickle.dump(final_model10, f)

# FROC, label 생성 안하게 수정 후 데이터 생성
stock_dict_5 = {
    'POSCO홀딩스': '005490',
    '삼성전자': '005930',
    '현대차': '005380',
    '셀트리온': '068270',
    '삼성생명': '032830'
}
date_range = ['2013-01-01', '2023-06-25']
label_window = (1, 10)
final_data = make_data(stock_dict_5, window_params, date_range, label_window=None)

data = final_data[final_data.Date > '2017-12-31'].copy()

data.company_name.value_counts()

# 스케일러 로드
with open('scalers_10.pkl', 'rb') as f:
    loaded_scalers = pickle.load(f)

# 모델 로드
with open('xgb_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

data['Score'] = None
for name in stock_dict_5.keys():
    data.loc[data.company_name==name, ['Score']] = loaded_model.predict_proba(loaded_scalers[name].transform(data[data.company_name==name][scaling_features['rbc']]))[:, 1]

data.to_csv('stock_data_with_score.csv', index=False)

#processor.process_and_insert_stock_prediction_data('stock_data_with_score.csv',['company_id', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 'ma', 'std', 'upper',
#                                                                            'lower', 'obv', 'cci', 'fast_k', 'fast_d', 'roc', 'rsi', 'rsi_dp','mfi', 'dp', 'ks_roc', 'ks_dp', 'ks_fast', 'score'])