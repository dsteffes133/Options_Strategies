
import pandas as pd
import matplotlib.pyplot as plt
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
        clean_key = expiration.strftime('%Y-%m-%d')  # Convert Timestamp to 'YYYY-MM-DD'
        expiration_dfs[clean_key] = df[df['expiration'] == expiration]
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





