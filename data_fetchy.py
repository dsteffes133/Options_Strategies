def get_historical_underlying_data(ticker):

    api_key = '8CTF0D0WVXWCLBVF'

    import requests
    import pandas as pd
    import requests


    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

     # Check if the expected key exists in the response
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Invalid response: {data}")  # Print the response for debugging

    # Convert the time series data to a DataFrame
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")

    df = df.drop(columns=['7. dividend amount', '8. split coefficient'])

    df = df.rename(columns={'1. open': 'Open', '2. high' : 'High', '3. low' : 'Low', '4. close' : 'Close', '5. adjusted close' : 'Adjusted Close', '6. volume' : 'Volume' })

    return df

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