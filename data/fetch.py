# data/fetch.py

import pandas as pd
import requests
import datetime

ALPHAVANTAGE_API_KEY = "8CTF0D0WVXWCLBVF"

def get_historical_underlying_data(ticker, start_date=None, end_date=None):
    """
    Fetches daily adjusted historical data for the given ticker from Alpha Vantage.
    Returns a DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume'].
    Optionally filter by start_date/end_date after retrieval.
    """
    url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
           f"&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}&outputsize=full")
    r = requests.get(url)
    data = r.json()
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Invalid response from Alpha Vantage: {data}")
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. adjusted close': 'Adjusted Close',
        '6. volume': 'Volume'
    })
    # Drop columns we don't need
    to_drop = [col for col in df.columns if col not in [
        'Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume']]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # filter by dates if provided
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(how='any', inplace=True)

    return df


def get_historical_options_data(ticker, start_date, end_date):
    """
    Placeholder function for historical options data.
    If your data source is Alpha Vantage's HISTORICAL_OPTIONS or something else,
    you can adapt it accordingly. Returns a DataFrame with many columns, including:
      ['contractID', 'expiration', 'strike', 'type', 'premium', 'volume', 'data_date']
    """
    # This is a stub to demonstrate approach:
    all_dates = pd.date_range(start_date, end_date, freq='D')
    frames = []
    for single_date in all_dates:
        date_str = single_date.strftime('%Y-%m-%d')
        url = (f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS"
               f"&symbol={ticker}&date={date_str}&apikey={ALPHAVANTAGE_API_KEY}")
        r = requests.get(url)
        data = r.json()
        if 'data' in data:
            df_daily = pd.DataFrame(data['data'])
            df_daily['data_date'] = date_str
            frames.append(df_daily)
    if frames:
        options_df = pd.concat(frames, ignore_index=True)
    else:
        options_df = pd.DataFrame()
    return options_df


def filter_options_by_volume(options_df, min_total_volume=100):
    """
    Filter out low-volume options. Summation by contractID.
    """
    if options_df.empty:
        return options_df

    # standardize column for volume
    if 'volume' not in options_df.columns:
        if 'Volume' in options_df.columns:
            options_df['volume'] = options_df['Volume']
        else:
            options_df['volume'] = 0

    # standardize contractID
    if 'contractID' not in options_df.columns:
        if 'contractId' in options_df.columns:
            options_df.rename(columns={'contractId': 'contractID'}, inplace=True)
        else:
            options_df['contractID'] = options_df.index

    options_df['volume'] = pd.to_numeric(options_df['volume'], errors='coerce').fillna(0)
    options_df['volume_summed'] = options_df.groupby('contractID')['volume'].transform('sum')
    filtered = options_df[options_df['volume_summed'] >= min_total_volume]
    return filtered


def create_expiration_dataframes(options_df):
    """
    Organize options data by unique expiration date. Returns dict keyed by 'YYYY-MM-DD' => DataFrame.
    """
    if 'expiration' not in options_df.columns:
        options_df['expiration'] = None
    else:
        options_df['expiration'] = pd.to_datetime(options_df['expiration'], errors='coerce')

    expiration_dfs = {}
    for exp in options_df['expiration'].dropna().unique():
        key_str = exp.strftime('%Y-%m-%d')
        expiration_dfs[key_str] = options_df[options_df['expiration'] == exp]

    return expiration_dfs
