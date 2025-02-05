import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_historical_options_data(ticker, api_key, begin_date, end_date):
    # Initialize an empty dictionary
    date_dict = {}

    # Create a list (or generator) of dates in 'YYYY-MM-DD' format
    all_dates = pd.date_range(begin_date, end_date, freq='D')
    
    # Loop through each date and fetch data
    for single_date in all_dates:
        date_str = single_date.strftime('%Y-%m-%d')
        
        url = f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS" \
              f"&symbol={ticker}&date={date_str}&apikey={api_key}"
        r = requests.get(url)
        data = r.json()

        # Some returns may be empty or have errors, so always check
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            # Store this date’s dataframe in the dictionary
            date_dict[date_str] = df
        else:
            # Optionally store an empty DataFrame or log the issue
            date_dict[date_str] = pd.DataFrame()

    master_df = pd.concat(date_dict.values(), ignore_index=True)

    return master_df

#Any option with less volume than 250 over our time period is not worth considering.

def volume_filter(df):
    # 1) Convert volume to numeric
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

# 2) Compute sums on the *unfiltered* DataFrame
    df['volume_summed'] = df.groupby('contractID')['volume'].transform('sum')

# 3) Now filter
    df = df[df['volume_summed'] > 250]
    return df


def create_expiration_dataframes(df):
    expiration_dfs = {}
    for expiration in df['expiration'].unique():
        expiration_dfs[expiration] = df[df['expiration'] == expiration]
    return expiration_dfs, expiration_dfs.keys()

def volatility_smile_function(expiration_date, expiration_dfs):
    df = expiration_dfs[expiration_date]

    df = df.sort_values(by = 'strike')

    strike_prices = df['strike'].astype(float).values

    iv = df['implied_volatility'].astype(float).values

    plt.figure(figsize=(12, 6))

    plt.plot(strike_prices, iv, marker='o', color='b', linestyle='-')
    plt.title(f"Volatility Smile for {expiration_date}")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")

    plt.show()

