##########################################################
# 1) IMPORTS & BASIC SETUP
##########################################################
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

##########################################################
# 2) DATA FETCH & CLEANING FUNCTIONS (FROM YOUR CODE)
##########################################################

import requests
import pandas as pd
import numpy as np
from sub_derivs import sub_derivs
from options_transformation import volume_filter


##########################################################
# UNDERLYING DATA FUNCTION
##########################################################
def get_historical_underlying_data(ticker):
    """
    Pull daily adjusted historical data for 'ticker' from Alpha Vantage.
    Returns a DataFrame with columns:
      [Open, High, Low, Close, Adjusted Close, Volume].
    Named 'underlying_df' internally for clarity.
    """
    api_key = '8CTF0D0WVXWCLBVF'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    # Check if the expected key exists in the response
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Invalid response: {data}")

    # Convert the time series data to a DataFrame
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

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    # Sort so earliest date is at top
    underlying_df = df.sort_index()

    return underlying_df


##########################################################
# OPTIONS DATA FUNCTIONS
##########################################################
def get_historical_options_data(ticker, api_key, begin_date, end_date):
    """
    Pull historical options data from Alpha Vantage (HISTORICAL_OPTIONS).
    Returns a combined DataFrame of all contract data
    between begin_date and end_date, named 'options_df'.
    """
    date_dict = {}
    all_dates = pd.date_range(begin_date, end_date, freq='D')

    for single_date in all_dates:
        date_str = single_date.strftime('%Y-%m-%d')
        url = (f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS"
               f"&symbol={ticker}&date={date_str}&apikey={api_key}")
        r = requests.get(url)
        data = r.json()

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
    Filter out options whose total volume (summed across the dataset)
    is less than 'min_total_volume'. Returns a filtered DataFrame
    named 'filtered_options_df'.
    """
    # Ensure the volume column is numeric
    options_df['volume'] = pd.to_numeric(options_df['volume'], errors='coerce').fillna(0)
    # Compute total volume by contractID
    options_df['volume_summed'] = options_df.groupby('contractID')['volume'].transform('sum')
    # Filter
    filtered_options_df = options_df[options_df['volume_summed'] > min_total_volume]
    return filtered_options_df


def create_expiration_dataframes(options_df):
    """
    Create a dict of DataFrames keyed by expiration date (string),
    and also return a list of expiration keys.
    """
    options_df['expiration'] = pd.to_datetime(options_df['expiration'], errors='coerce')
    
    expiration_dfs = {}
    unique_exps = options_df['expiration'].dropna().unique()
    for exp in unique_exps:
        key_str = exp.strftime('%Y-%m-%d')
        expiration_dfs[key_str] = options_df[options_df['expiration'] == exp]

    expiration_keys = list(expiration_dfs.keys())
    return expiration_dfs, expiration_keys


def compute_vanna_and_charm(options_df):
    """
    Compute vanna & charm for each row in the options DataFrame,
    assuming the columns vega, delta, implied_volatility, last, theta exist.
    Returns two Series (vanna, charm) which can be added to the DataFrame.
    """
    options_df['vanna'] = (
        options_df['vega'] * (options_df['delta'] / options_df['implied_volatility'].replace(0, np.nan))
    )
    options_df['charm'] = (
        -1 * (options_df['theta'] * options_df['delta']) / options_df['last'].replace(0, np.nan)
    )
    
    return options_df['vanna'], options_df['charm']


##########################################################
# 3) STRATEGY REPRESENTATION & PAYOFF/ANALYSIS
##########################################################

class OptionLeg:
    """
    Represents a single options leg: call/put, long/short, strike, expiry.
    Quantity and premium are set later (e.g., after filtering data or finalizing the thesis).
    """
    def __init__(self, is_call, is_long, strike, expiry):
        self.is_call = is_call
        self.is_long = is_long
        self.strike = strike
        self.expiry = expiry

        # We'll set these later once we know the contract details from the data or the thesis
        self.quantity = None
        self.premium = None

    def set_quantity_and_premium(self, quantity, premium):
        """
        Assign quantity (number of contracts) and premium (cost or credit per contract).
        """
        self.quantity = quantity
        self.premium = premium

    def payoff_at_expiration(self, underlying_price):
        """
        Intrinsic value minus premium, times the sign (long vs. short), times quantity.
        If quantity or premium is None, we assume 0 or error out—your code can handle that logic.
        """
        if self.quantity is None or self.premium is None:
            raise ValueError("Quantity and premium must be set before calculating payoff.")

        if self.is_call:
            intrinsic = max(0, underlying_price - self.strike)
        else:
            intrinsic = max(0, self.strike - underlying_price)

        sign = 1 if self.is_long else -1

        # For real equity options, typically each contract = 100 shares,
        # so you may want to multiply by 100 if needed:
        payoff = sign * (intrinsic - self.premium) * self.quantity
        return payoff


class Strategy:
    """
    A multi-leg strategy container. Summarizes payoff across all legs.
    """
    def __init__(self, name="Unnamed Strategy"):
        self.name = name
        self.legs = []

    def add_leg(self, leg: OptionLeg):
        self.legs.append(leg)

    def payoff_at_expiration(self, underlying_price):
        """
        Sum the payoff_at_expiration() of each leg.
        """
        return sum(leg.payoff_at_expiration(underlying_price) for leg in self.legs)

    def payoff_diagram(self, min_price, max_price, steps=50):
        """
        Return arrays of underlying_price, payoff for plotting.
        """
        import numpy as np

        prices = np.linspace(min_price, max_price, steps)
        payoffs = [self.payoff_at_expiration(p) for p in prices]
        return prices, payoffs

##########################################################
# 4) THE 25 STRATEGY "CONSTRUCTORS"
##########################################################
# We'll define short example constructors for each. In practice, you might
# want a more flexible approach (like passing a dictionary of strikes & premiums).
# Below are EXAMPLES for a few; you'd replicate for all 25.

def create_long_call(strike, expiry, premium, quantity=1):
    """
    Simple single-leg strategy: Long Call.
    """
    strat = Strategy(name="Long Call")
    leg = OptionLeg(is_call=True, is_long=True, strike=strike, expiry=expiry,
                    quantity=quantity, premium=premium)
    strat.add_leg(leg)
    return strat

def create_long_put(strike, expiry, premium, quantity=1):
    strat = Strategy(name="Long Put")
    leg = OptionLeg(is_call=False, is_long=True, strike=strike, expiry=expiry,
                    quantity=quantity, premium=premium)
    strat.add_leg(leg)
    return strat

def create_short_call(strike, expiry, premium, quantity=1):
    strat = Strategy(name="Short Call")
    leg = OptionLeg(is_call=True, is_long=False, strike=strike, expiry=expiry,
                    quantity=quantity, premium=premium)
    strat.add_leg(leg)
    return strat

def create_short_put(strike, expiry, premium, quantity=1):
    strat = Strategy(name="Short Put")
    leg = OptionLeg(is_call=False, is_long=False, strike=strike, expiry=expiry,
                    quantity=quantity, premium=premium)
    strat.add_leg(leg)
    return strat

def create_bull_call_spread(strike_long, strike_short, expiry, premium_long, premium_short, quantity=1):
    """
    Buy call at strike_long, sell call at strike_short.
    """
    strat = Strategy(name="Bull Call Spread")
    long_leg = OptionLeg(is_call=True, is_long=True, strike=strike_long, expiry=expiry,
                         quantity=quantity, premium=premium_long)
    short_leg = OptionLeg(is_call=True, is_long=False, strike=strike_short, expiry=expiry,
                          quantity=quantity, premium=premium_short)
    strat.add_leg(long_leg)
    strat.add_leg(short_leg)
    return strat

# ... Continue similarly for Bear Spread, Butterfly, Iron Condor, Risk Reversal, etc. ...
# For brevity, we won't write out all 25 here. But the pattern is the same:
# 1) Create Strategy object
# 2) Add appropriate OptionLeg objects
# 3) Return the Strategy

# You might store them in a dictionary for easy referencing:
STRATEGY_BUILDERS = {
    'LongCall': create_long_call,
    'LongPut': create_long_put,
    'ShortCall': create_short_call,
    'ShortPut': create_short_put,
    'BullCallSpread': create_bull_call_spread,
    # etc. for all 25...
}

##########################################################
# 5) HOW BOSSMAN INPUTS THE THESIS AND GETS HIS STRAT
##########################################################

def rank_strategies_for_thesis(ticker,
                               direction,           # e.g. "bullish", "bearish", "neutral"
                               target_price,        # e.g. 165
                               current_price,       # e.g. 150
                               time_horizon_days,   # e.g. 30
                               risk_tolerance,      # e.g. "low", "medium", "high"
                               etc_parameters=None):
    """
    Illustrative function: 
    1) Use the boss's thesis to pick some strikes/expirations
    2) Build candidate strategies
    3) Compute scenario payoffs
    4) Return the "best" or a sorted list

    This is a skeleton—actual logic can be as simple or advanced as you like.
    """
    if etc_parameters is None:
        etc_parameters = {}

    # Example: build 3 candidate strikes based on direction + target_price
    # This is simplistic logic, just to illustrate.
    expiry_str = (datetime.date.today() + datetime.timedelta(days=time_horizon_days)).strftime('%Y-%m-%d')

    # Possibly you get option chain and find actual premiums for chosen strikes
    # For demonstration, let's just guess some premium values:
    # (In real usage, you'd look up actual premium from your 'expiration_dfs' data)
    premium_est_long = 2.50
    premium_est_short = 1.00

    strategies = []

    if direction == "bullish":
        # Maybe a bull call spread from near-the-money to target
        s = create_bull_call_spread(
            strike_long=current_price,  # buy call at 150
            strike_short=target_price,  # sell call at 165
            expiry=expiry_str,
            premium_long=premium_est_long,
            premium_short=premium_est_short
        )
        strategies.append(s)

        # Another example: Long Call
        long_call = create_long_call(strike=current_price, expiry=expiry_str,
                                     premium=premium_est_long, quantity=1)
        long_call.name = "Simple Long Call"
        strategies.append(long_call)

    # If direction is "bearish," you might build a Bear Put Spread, etc.
    # If "neutral," maybe an Iron Condor or Short Straddle, etc.

    # Next, do scenario analysis for each strategy.
    # Let's define a scenario range around current_price +/- 20% for demonstration.
    scenario_prices = [current_price*0.8, current_price, target_price, current_price*1.2]

    ranked_results = []
    for strat in strategies:
        # We'll do a simple EV approach: payoff in each scenario with equal prob.
        payoffs = []
        for sp in scenario_prices:
            payoffs.append(strat.payoff_at_expiration(sp))
        ev = np.mean(payoffs)
        # We can incorporate risk_tolerance logic or advanced metrics (Sharpe, etc.)
        # For now, let's store the EV.
        ranked_results.append((strat, ev))

    # Sort by EV descending
    ranked_results.sort(key=lambda x: x[1], reverse=True)

    return ranked_results


##########################################################
# 6) EXAMPLE "MAIN" USAGE
##########################################################
if __name__ == "__main__":
    # 6.1. Get Underlying Data
    ticker = "AAPL"
    df_und = get_historical_underlying_data(ticker)

    # 6.2. Get Options Data
    api_key_av = "8CTF0D0WVXWCLBVF"
    begin_date = "2023-01-01"
    end_date   = "2023-02-01"
    df_opts = get_historical_options_data(ticker, api_key_av, begin_date, end_date)

    # 6.3. Clean & Filter
    df_opts = volume_filter(df_opts)
    expiration_dfs, expiration_keys = create_expiration_dataframes(df_opts)
    # If we want vanna/charm:
    if len(df_opts) > 0:
        df_opts['vanna'], df_opts['charm'] = sub_derivs(df_opts)

    # 6.4. Suppose Boss's Thesis:
    direction = "bullish"    # "bearish" or "neutral" or ...
    current_price = 150      # e.g. from the last row of df_und
    target_price = 165       # boss's predicted price in the time horizon
    time_horizon_days = 30
    risk_tolerance = "medium"

    # 6.5. Rank Strategies
    results = rank_strategies_for_thesis(
        ticker=ticker,
        direction=direction,
        target_price=target_price,
        current_price=current_price,
        time_horizon_days=time_horizon_days,
        risk_tolerance=risk_tolerance
    )

    # 6.6. Display top picks
    print("Ranked Strategies (top first):")
    for strat, ev in results:
        print(f"Strategy: {strat.name}, Expected Value: {ev:.2f}")
        # Optionally you can plot the payoff diagram here:
        # prices, payoffs = strat.payoff_diagram(min_price=120, max_price=180)
        # ... (plot using matplotlib if desired)
