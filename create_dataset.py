import bitget.bitget_api as baseApi
from bitget.exceptions import BitgetAPIException
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import os

def get_historical_candlestick(symbol, productType, endTime):
    try:
        params={}
        params['symbol'] = symbol
        params['productType'] = productType
        params['granularity'] = '1m'
        params['endTime'] = endTime
        params['limit'] = '200'
        data = baseApi.get('/api/v2/mix/market/history-candles', params)['data']
        for i in range(len(data)):
            del data[i][-1]
        return data
    except BitgetAPIException as e:
        print("error:" + e.message)

if __name__ == '__main__':
    load_dotenv()
    baseApi = baseApi.BitgetApi(os.environ.get('api_key'), os.environ.get('secret_key'), os.environ.get('passphrase'))
    
    start_time = str(int(datetime(2025, 6, 1).timestamp()*1000))
    end_time = str(int(datetime(2025, 7, 1).timestamp()*1000))
    time = end_time
    data = []
    while time > start_time:
        data = get_historical_candlestick('BTCUSDT', 'USDT-FUTURES', time) + data
        time = data[0][0]
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).astype(np.float32)

    filename_start_date = datetime.fromtimestamp(int(start_time)/1000).strftime('%Y%m%d')
    filename_end_date = datetime.fromtimestamp(int(end_time)/1000).strftime('%Y%m%d')
    df.to_csv(f'./dataset/data_{filename_start_date}_{filename_end_date}.csv')