import FinanceDataReader as fdr
import numpy as np
import pickle
import sys

import xgboost as xgb
from sklearn.preprocessing import RobustScaler

import pymysql
import pandas as pd
from DB.insert_select_db import Database
from DB.realtime_insert_data_processor import RealTimeDataProcessor

# READ config.txt
with open("./DB/config.txt", "r") as file:
    exec(file.read())

db = Database(configs)
processor = RealTimeDataProcessor(db)

stock_ids = db.get_company_ids()
stock_dict = {
    '삼성전자':'005930',
    '현대차':'005380',
    '삼성생명':'032830',
    '셀트리온':'068270',
    '포스코':'005490',
}
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
model_features = ['roc', 'ks_roc', 'cci', 'fast_k', 'fast_d', 'rsi', 'mfi', 'ks_fast', 'rsi_dp', 'dp', 'ks_dp']
required_columns = ['company_id', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'stock_change', 'ma', 'std', 'upper',
                    'lower', 'obv', 'cci', 'fast_k', 'fast_d', 'roc', 'rsi', 'rsi_dp','mfi', 'dp', 'ks_roc', 'ks_dp', 'ks_fast', 'score']

dataframes = []
for i in range(1, 6):
    with db.DB.cursor() as cursor:
        query = f"SELECT * FROM stock_prediction WHERE company_id = {i} ORDER BY date DESC LIMIT 60"
        cursor.execute(query)
        column_names = [desc[0] for desc in cursor.description]
        result = cursor.fetchall()
    df = pd.DataFrame(result, columns=column_names)
    dataframes.append(df)

last = dataframes[0].date.max().strftime("%Y-%m-%d")
print('최근 데이터 생성날짜 :', last)
ks11 = fdr.DataReader('KS11', last)
if len(ks11.index) <= 1:
    print('추가 주가데이터가 존재하지 않습니다.')
    sys.exit()
date_list = [date.strftime("%Y-%m-%d") for date in ks11.index][1:]
print(date_list, '에 대해 데이터 생성')

# load scaler, model
with open('./csv/scalers_60_final.pkl', 'rb') as f:
    scalers = pickle.load(f)
scalers['포스코'] = scalers['POSCO홀딩스']
model = xgb.XGBClassifier()
model.load_model('./csv/model_final_60.bin')

# make kospi data
ks_data = fdr.DataReader('KS11', dataframes[0]['date'].min())
ks_data['ks_fast'] = ((ks_data['Close'] - ks_data['Low'].rolling(14).min()) / (ks_data['High'].rolling(14).max() - ks_data['Low'].rolling(14).min())) * 100
ks_data['ks_ROC'] = np.round(ks_data.Close.diff(window_params['ks_ROC']) / ks_data.Close.shift(window_params['ks_ROC']), 4)
ks_data['ks_DP'] = np.round(ks_data.Close / ks_data.Close.rolling(window=window_params['ks_ROC']).mean().values, 4) - 1


for stock_name, code in stock_dict.items():
    
    company_data = fdr.DataReader(code, date_list[0], date_list[-1])
    sorted_df = dataframes[stock_ids[stock_name]-1].sort_values(by='date', ascending=True)
    sorted_df.reset_index(drop=True, inplace=True)
    for date, row in company_data.iterrows():

        #-----calculate features for new row-----#
        company_id = stock_ids[stock_name]
        company_name = stock_name
        open_ = row.Open
        high = row.High
        low = row.Low
        close = row.Close
        volume = row.Volume
        stock_change = row.Change

        ma = np.append(sorted_df.close[-window_params['MA']+1:], close).mean()

        std = np.append(sorted_df.close[-19:], close).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)

        direction = 1 if sorted_df['close'].values[-1] < close else -1 if sorted_df['close'].values[-1] > close else 0
        obv = sorted_df.obv.values[-1] + (direction * volume)

        typical_price = (low + high + close) / 3
        typical_prices_cci = np.append(((sorted_df['high'] + sorted_df['low'] + sorted_df['close']) / 3)[-window_params['CCI']+1:], typical_price)
        cci = (typical_price - typical_prices_cci.mean()) / (0.015 * pd.Series(typical_prices_cci).sub(typical_prices_cci.mean()).abs().mean())

        fast_k = (close - np.append(sorted_df['low'][-window_params['fast_k']+1:], low).min()) / (np.append(sorted_df['high'][-window_params['fast_k']+1:], high).max() - np.append(sorted_df['low'][-4:], low).min()) * 100
        fast_d = np.append(sorted_df['fast_k'][-window_params['fast_d']+1:], fast_k).mean()

        roc = close / sorted_df['close'].values[-window_params['ROC']] - 1

        delta = sorted_df['close'].diff().append(pd.Series(close - sorted_df['close'].values[-1]))
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        _gain = up.ewm(com=(window_params['RSI'] - 1), min_periods=window_params['RSI']).mean()
        _loss = down.abs().ewm(com=(window_params['RSI'] - 1), min_periods=window_params['RSI']).mean()
        RS = _gain / _loss
        rsi_60 = pd.Series(100 - (100 / (1 + RS)), name="RSI")
        rsi = rsi_60.values[-1]
        
        rsi_dp = np.round(rsi / rsi_60[-window_params['RSI_DP']:].mean(), 4) - 1
        dp = np.round(close / ma, 4) - 1
        
        typical_prices = np.append(((sorted_df['high'] + sorted_df['low'] + sorted_df['close']) / 3), typical_price)
        money_flow = typical_prices * np.append(sorted_df['volume'], volume)
        positive_flow =[] 
        negative_flow = []
        for i in range(1, len(money_flow)):
            if typical_prices[i] > typical_prices[i-1]: 
                positive_flow.append(money_flow[i-1])
                negative_flow.append(0) 
            elif typical_prices[i] < typical_prices[i-1]:
                negative_flow.append(money_flow[i-1])
                positive_flow.append(0)
            else: 
                positive_flow.append(0)
                negative_flow.append(0)
        positive_mf = []
        negative_mf = [] 
        for i in range(window_params['MFI']-1, len(positive_flow)):
            positive_mf.append(sum(positive_flow[i+1-window_params['MFI'] : i+1]))
        for i in range(window_params['MFI']-1, len(negative_flow)):
            negative_mf.append(sum(negative_flow[i+1-window_params['MFI'] : i+1]))
        mfi_60 = list(100 * (np.array(positive_mf) / (np.array(positive_mf)  + np.array(negative_mf))))
        mfi_60 = list(np.repeat(np.nan,len(typical_prices)-len(mfi_60))) + mfi_60
        mfi = mfi_60[-1]
        
        model_input = np.array([roc, ks_data['ks_ROC'][date.strftime("%Y-%m-%d")], cci, fast_k, fast_d, rsi, mfi, ks_data['ks_fast'][date.strftime("%Y-%m-%d")], rsi_dp, dp, ks_data['ks_DP'][date.strftime("%Y-%m-%d")]]).reshape(1, -1)
        scaled_input = scalers[company_name].transform(model_input)
        score = model.predict_proba(scaled_input)[0, 1]
        #-----------------------------------------#

        new_row = {
            'stock_prediction_id':np.nan,
            'company_id':company_id,
            'company_name':company_name,
            'open':open_,
            'high':high,
            'low':low,
            'close':close,
            'volume':volume,
            'stock_change':stock_change,
            'date':date.strftime("%Y-%m-%d"),
            'ma':ma,
            'std':std,
            'upper':upper,
            'lower':lower,
            'obv':obv,
            'cci':cci, 
            'fast_k':fast_k, 
            'fast_d':fast_d,
            'roc':roc,
            'rsi':rsi,
            'rsi_dp':rsi_dp,
            'mfi':mfi,
            'dp':dp,
            'ks_roc':ks_data['ks_ROC'][date.strftime("%Y-%m-%d")],
            'ks_dp':ks_data['ks_DP'][date.strftime("%Y-%m-%d")],
            'ks_fast':ks_data['ks_fast'][date.strftime("%Y-%m-%d")],
            'score':score
        }
        sorted_df = sorted_df.append(new_row, ignore_index=True)
        print(company_name, date.strftime("%Y-%m-%d"), '모멘텀 점수 :', score)

    processor.process_and_insert_stock_prediction_data(sorted_df[required_columns][-len(company_data):], required_columns, table_name='stock_prediction')
