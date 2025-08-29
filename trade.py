import bitget.bitget_api as baseApi
from bitget.exceptions import BitgetAPIException

from stable_baselines3 import PPO

import pandas as pd
import datetime
import time
import numpy as np

from dotenv import load_dotenv
import os
        
def fetch_ohlcv(symbol, productType, granularity, limit):
    try:
        params={}
        params['symbol'] = symbol
        params['productType'] = productType
        params['granularity'] = granularity
        params['limit'] = limit
        data = baseApi.get('/api/v2/mix/market/candles', params)['data']
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_currency']).astype(np.float32)
        df.drop('volume_currency', axis=1, inplace=True)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=20).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=20).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['sma'] = df['close'].rolling(window=14).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df = df.replace(' ', pd.NA)
        df = df.dropna()
        return df
    except BitgetAPIException as e:
        print("error:" + e.message)

def set_position_mode(productType, posMode):
    try:
        params={}
        params['productType'] = productType
        params['posMode'] = posMode
        baseApi.post('/api/v2/mix/account/set-position-mode', params)
    except BitgetAPIException as e:
        print("error:" + e.message)

def set_leverage(symbol, productType, marginCoin, leverage):
    try:
        params={}
        params['symbol'] = symbol
        params['productType'] = productType
        params['marginCoin'] = marginCoin
        params['leverage'] = leverage
        baseApi.post('/api/v2/mix/account/set-leverage', params)
    except BitgetAPIException as e:
        print("error:" + e.message)

def place_order(symbol, productType, marginCoin, marginMode, size, side, orderType):
    try:
        params={}
        params['symbol'] = symbol
        params['productType'] = productType
        params['marginCoin'] = marginCoin
        params['marginMode'] = marginMode
        params['size'] = size
        params['side'] = side
        params['orderType'] = orderType
        baseApi.post('/api/v2/mix/order/place-order', params)
    except BitgetAPIException as e:
        print("error:" + e.message)

if __name__ == '__main__':
    load_dotenv()
    baseApi = baseApi.BitgetApi(os.environ.get('apiKey'), os.environ.get('secretKey'), os.environ.get('passphrase'))
    
    print('START TIME: ', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    set_position_mode('SUSDT-FUTURES', 'one_way_mode')
    set_leverage('SBTCSUSDT', 'SUSDT-FUTURES', 'SUSDT', '10')
    
    posInfo = 'empty'
    while True:
        df = fetch_ohlcv('SBTCSUSDT', 'SUSDT-FUTURES', '1m', '30')
        obs = df.iloc[-1][['open', 'high', 'low', 'close', 'volume', 'rsi', 'sma', 'ema_12', 'ema_26', 'macd', 'macd_signal']].values.astype(np.float32)
        action, _states = model.predict(obs)
        if action == 0: # HOLD
            pass
        elif action == 1:
            if posInfo == 'empty': # OPEN LONG
                place_order('SBTCSUSDT', 'SUSDT-FUTURES', 'SUSDT', 'isolated', '0.001', 'buy', 'market')
                posInfo = 'long'
        elif action == 2:
            if posInfo == 'empty': # OPEN SHORT
                place_order('SBTCSUSDT', 'SUSDT-FUTURES', 'SUSDT', 'isolated', '0.001', 'sell', 'market')
                posInfo = 'short'
        elif action == 3: # CLOSE
            if posInfo == 'long':
                place_order('SBTCSUSDT', 'SUSDT-FUTURES', 'SUSDT', 'isolated', '0.001', 'sell', 'market')
            elif posInfo == 'short':
                place_order('SBTCSUSDT', 'SUSDT-FUTURES', 'SUSDT', 'isolated', '0.001', 'buy', 'market')
            posInfo = 'empty'
        time.sleep(10)