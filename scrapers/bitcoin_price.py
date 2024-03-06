import requests
import pandas as pd


def get_price_history():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range'
    # Timestamps for 2021-01-01 to 2021-12-31
    params = {'vs_currency': 'usd', 'from': 1609459200, 'to': 1672444800}
    response = requests.get(url, params=params)
    data = response.json()
    df_price = pd.DataFrame(data['prices'], columns=['time', 'price'])
    df_mktcap = pd.DataFrame(
        data['market_caps'], columns=['time', 'market_cap']
    )
    df_volume = pd.DataFrame(
        data['total_volumes'], columns=['time', 'total_volume']
    )
    df = pd.merge(df_price, df_mktcap, on='time')
    df = pd.merge(df, df_volume, on='time')
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)

    print(df.head())

    return df
