import pandas as pd
import requests
import numpy as np

def get_historical_underlying_data(ticker):
    """
    Fetches daily adjusted historical data for the given ticker from Alpha Vantage.
    Returns a DataFrame with columns: Open, High, Low, Close, Adjusted Close, Volume.
    """
    api_key = '8CTF0D0WVXWCLBVF'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Invalid response: {data}")
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.drop(columns=['7. dividend amount', '8. split coefficient'])
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. adjusted close': 'Adjusted Close',
        '6. volume': 'Volume'
    })
    df.index = pd.to_datetime(df.index)
    underlying_df = df.sort_index()
    return underlying_df

def get_historical_options_data(ticker, api_key, begin_date, end_date):
    """
    Fetches historical options data for the given ticker (from Alpha Vantage)
    between begin_date and end_date. (Remember: this API is strictly historical.)
    Returns a combined DataFrame.
    """
    date_dict = {}
    all_dates = pd.date_range(begin_date, end_date, freq='D')
    for single_date in all_dates:
        date_str = single_date.strftime('%Y-%m-%d')
        url = (f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS"
               f"&symbol={ticker}&date={date_str}&apikey={api_key}")
        r = requests.get(url)
        data = r.json()
        # (For debugging, you might print the raw response here)
        if 'data' in data:
            df_daily = pd.DataFrame(data['data'])
            df_daily['data_date'] = date_str
            date_dict[date_str] = df_daily
        else:
            date_dict[date_str] = pd.DataFrame()
    if len(date_dict) > 0:
        options_df = pd.concat(date_dict.values(), ignore_index=True)
    else:
        options_df = pd.DataFrame()
    return options_df

def filter_options_by_volume(options_df, min_total_volume=250):
    """
    Filters out options contracts that have insufficient total volume.
    If volume is missing, attempts to rename 'Volume' to 'volume' or uses a default.
    """
    if options_df.empty or 'volume' not in options_df.columns:
        if 'Volume' in options_df.columns:
            options_df['volume'] = options_df['Volume']
        else:
            return options_df  # return as is if no volume info
    options_df['volume'] = pd.to_numeric(options_df['volume'], errors='coerce').fillna(0)
    if 'contractID' not in options_df.columns:
        if 'contractId' in options_df.columns:
            options_df = options_df.rename(columns={'contractId': 'contractID'})
        else:
            options_df['contractID'] = options_df.index
    options_df['volume_summed'] = options_df.groupby('contractID')['volume'].transform('sum')
    filtered_options_df = options_df[options_df['volume_summed'] > min_total_volume]
    return filtered_options_df

def create_expiration_dataframes(options_df):
    """
    Organizes options data into separate DataFrames keyed by expiration date.
    Returns a dictionary (expiration_dfs) and a list of expiration keys.
    """
    if 'expiration' in options_df.columns:
        options_df['expiration'] = pd.to_datetime(options_df['expiration'], errors='coerce')
    else:
        options_df['expiration'] = pd.NaT
    expiration_dfs = {}
    unique_exps = options_df['expiration'].dropna().unique()
    for exp in unique_exps:
        key_str = exp.strftime('%Y-%m-%d')
        expiration_dfs[key_str] = options_df[options_df['expiration'] == exp]
    expiration_keys = list(expiration_dfs.keys())
    return expiration_dfs, expiration_keys