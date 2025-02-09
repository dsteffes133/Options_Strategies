# full_script.py
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import QuantLib as ql

############################################################
# 1. DATA FETCHING FUNCTIONS
############################################################

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

############################################################
# 2. STRATEGY REPRESENTATION CLASSES & BUILDERS
############################################################

class UnderlyingLeg:
    """
    Represents a position in the underlying asset.
    """
    def __init__(self, is_long, entry_price, quantity=1):
        self.is_long = is_long
        self.entry_price = entry_price
        self.quantity = quantity

    def payoff_at_expiration(self, underlying_price):
        sign = 1 if self.is_long else -1
        return sign * (underlying_price - self.entry_price) * self.quantity

class OptionLeg:
    """
    Represents an options contract (with potential for extension to include Greeks).
    """
    def __init__(self, is_call, is_long, strike, expiry, quantity=1, premium=0.0):
        self.is_call = is_call
        self.is_long = is_long
        self.strike = strike
        self.expiry = expiry
        self.quantity = quantity
        self.premium = premium
        self.greeks = {}  # Placeholder for delta, gamma, etc.

    def payoff_at_expiration(self, underlying_price):
        if self.is_call:
            intrinsic = max(0, underlying_price - self.strike)
        else:
            intrinsic = max(0, self.strike - underlying_price)
        sign = 1 if self.is_long else -1
        payoff = sign * (intrinsic - self.premium) * self.quantity
        return payoff

class Strategy:
    """
    A multi-leg options strategy.
    """
    def __init__(self, name="Unnamed Strategy"):
        self.name = name
        self.legs = []
        self.attributes = {}  # Placeholder for aggregated Greeks, risk metrics, etc.

    def add_leg(self, leg):
        self.legs.append(leg)

    def payoff_at_expiration(self, underlying_price):
        return sum(leg.payoff_at_expiration(underlying_price) for leg in self.legs)

    def payoff_diagram(self, min_price, max_price, steps=50):
        prices = np.linspace(min_price, max_price, steps)
        payoffs = [self.payoff_at_expiration(p) for p in prices]
        return prices, payoffs

# Example strategy builders (expand as needed)
def create_long_call(strike, expiry, premium, quantity=1):
    strat = Strategy(name="Long Call")
    leg = OptionLeg(is_call=True, is_long=True, strike=strike, expiry=expiry, quantity=quantity, premium=premium)
    strat.add_leg(leg)
    return strat

def create_bull_call_spread(strike_long, strike_short, expiry, premium_long, premium_short, quantity=1):
    strat = Strategy(name="Bull Call Spread")
    long_leg = OptionLeg(is_call=True, is_long=True, strike=strike_long, expiry=expiry, quantity=quantity, premium=premium_long)
    short_leg = OptionLeg(is_call=True, is_long=False, strike=strike_short, expiry=expiry, quantity=quantity, premium=premium_short)
    strat.add_leg(long_leg)
    strat.add_leg(short_leg)
    return strat

STRATEGY_BUILDERS = {
    'LongCall': create_long_call,
    'BullCallSpread': create_bull_call_spread,
    # ... Add the other 23 strategy builders as needed.
}

############################################################
# 2.5 FUTURE PRICE FORECASTING
############################################################

def forecast_future_price(S0, current_date, future_entry_date, risk_free_rate, volatility, num_paths=1000):
    """
    Forecasts the underlying price at a future entry date using GBM simulation.
    Both current_date and future_entry_date are datetime.date objects.
    Returns the average simulated price.
    """
    T = (future_entry_date - current_date).days / 365.0
    simulated_prices = []
    for _ in range(num_paths):
        Z = np.random.normal()
        price = S0 * np.exp((risk_free_rate - 0.5 * volatility**2)*T + volatility*np.sqrt(T)*Z)
        simulated_prices.append(price)
    return np.mean(simulated_prices)

############################################################
# 3. SIMULATION & EVALUATION FUNCTIONS (USING QUANTLIB)
############################################################

def simulate_terminal_price(S0, T, risk_free_rate, volatility, thesis_scenarios, num_paths=10000, use_thesis=True):
    """
    Simulates terminal underlying prices over a horizon T (in years) using a mixture of GBM processes.
    If use_thesis is True, adjustments (drift changes) are applied based on the thesis scenarios.
    Otherwise, normal GBM (using risk_free_rate as drift) is used.
    """
    terminal_prices = []
    # For each simulated path:
    for _ in range(num_paths):
        # Choose drift according to thesis if required:
        if use_thesis:
            # Normalize scenario probabilities:
            probs = np.array([sc['prob'] for sc in thesis_scenarios])
            probs = probs / probs.sum()
            cum_probs = np.cumsum(probs)
            r_val = np.random.rand()
            scenario = None
            for sc, cp in zip(thesis_scenarios, cum_probs):
                if r_val <= cp:
                    scenario = sc
                    break
            # Process scenario types:
            if scenario['description'] == 'moderate':
                move_pct = scenario.get('move_pct', 10)
                adjusted_drift = np.log(1 + move_pct/100.0) / T
            elif scenario['description'] == 'big move':
                move_pct = scenario.get('move_pct', 30)  # absolute magnitude
                direction_prob = scenario.get('direction_prob', 0.5)
                # Use the direction probability to choose bullish or bearish:
                if np.random.rand() < direction_prob:
                    move_pct = move_pct  # bullish move
                else:
                    move_pct = -move_pct  # bearish move
                adjusted_drift = np.log(1 + move_pct/100.0) / T
            elif scenario['description'] == 'follow_ma':
                ma_value = scenario['ma_value']
                adjusted_drift = np.log(ma_value / S0) / T
            elif scenario['description'] == 'neutral':
                adjusted_drift = risk_free_rate
            else:
                adjusted_drift = risk_free_rate
        else:
            adjusted_drift = risk_free_rate

        # Set up QuantLib objects:
        today = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = today
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        expiry_date = calendar.advance(today, int(T * 365), ql.Days)
        underlying = ql.SimpleQuote(S0)
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
        vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, volatility, day_count))
        # Use BlackScholesMertonProcess to accept dividend_ts:
        process = ql.BlackScholesMertonProcess(ql.QuoteHandle(underlying), dividend_ts, flat_ts, vol_handle)
        # Set up a time grid for the simulation.
        time_grid = ql.TimeGrid(T, 100)
        dim = len(time_grid) - 1
    
        # Use the default constructor (without seed) for the Mersenne Twister RNG.
        mt_rng = ql.MersenneTwisterUniformRng()
    
        # Create the uniform sequence generator using the RNG.
        uniform_seq_gen = ql.UniformRandomSequenceGenerator(dim, mt_rng)
    
        # Wrap it in a Gaussian random sequence generator.
        rng = ql.GaussianRandomSequenceGenerator(uniform_seq_gen)
    
        # Generate a path using the Gaussian path generator.
        seq = ql.GaussianPathGenerator(process, T, dim, rng, False)
        sample_path = seq.next()
        path = sample_path.value()
        S_T = path.back()

        S_T_adjusted = S_T * np.exp(adjusted_drift * T)
        terminal_prices.append(S_T_adjusted)
    return terminal_prices

def evaluate_strategy_payoff(strategy, terminal_prices):
    """
    Computes the expected payoff and minimum payoff (downside) for a given strategy,
    given simulated terminal prices.
    """
    payoffs = [strategy.payoff_at_expiration(price) for price in terminal_prices]
    ev = np.mean(payoffs)
    downside = min(payoffs)
    return ev, downside

def optimize_strategy(candidates, bankroll):
    """
    Placeholder for an optimization routine.
    In the future, this could be replaced with linear programming to determine optimal contracts/quantities.
    For now, simply returns the top candidates sorted by score.
    """
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:3]

def rank_strategies_for_thesis(ticker, options_df, current_price, simulation_horizon_days, thesis_scenarios, risk_tolerance, future_entry_date=None):
    """
    Evaluates candidate strategies over a simulation horizon.
    If future_entry_date is provided and is in the future, forecast the underlying price at that date
    and use that as the starting price S0.
    Returns a list of candidate strategies (each with strategy object, score, contract details,
    expected value under thesis, and expected value under normal GBM).
    """
    today_date = datetime.date.today()
    risk_free_rate = 0.02
    volatility = 0.30
    if future_entry_date and future_entry_date > today_date:
        S0_future = forecast_future_price(current_price, today_date, future_entry_date, risk_free_rate, volatility, num_paths=1000)
    else:
        S0_future = current_price
    T = simulation_horizon_days / 365.0
    # Simulate terminal prices under thesis assumptions:
    terminal_prices_thesis = simulate_terminal_price(S0_future, T, risk_free_rate, volatility, thesis_scenarios, num_paths=1000, use_thesis=True)
    # Simulate terminal prices under normal GBM (without thesis adjustments):
    terminal_prices_normal = simulate_terminal_price(S0_future, T, risk_free_rate, volatility, thesis_scenarios, num_paths=1000, use_thesis=False)
    if risk_tolerance.lower() == "low":
        lambda_val = 1.5
    elif risk_tolerance.lower() == "high":
        lambda_val = 0.5
    else:
        lambda_val = 1.0
    strategy_results = []
    future_date = today_date + datetime.timedelta(days=simulation_horizon_days)
    def valid_expiration(exp):
        if pd.isnull(exp):
            return False
        if isinstance(exp, pd.Timestamp):
            exp_date = exp.date()
        else:
            exp_date = pd.to_datetime(exp).date()
        return exp_date <= future_date
    filtered_options = options_df[options_df['expiration'].apply(valid_expiration)]
    for idx, contract in filtered_options.iterrows():
        try:
            strike = float(contract['strike'])
            expiry = contract['expiration']
            if 'premium' in contract and not pd.isnull(contract['premium']):
                premium = float(contract['premium'])
            elif 'last' in contract and not pd.isnull(contract['last']):
                premium = float(contract['last'])
            else:
                continue
        except Exception:
            continue
        for strat_name, builder in STRATEGY_BUILDERS.items():
            try:
                if strat_name == 'BullCallSpread':
                    if strike <= S0_future:
                        continue
                    strike_long = S0_future
                    strike_short = strike
                    premium_long = premium * 1.2
                    premium_short = premium
                    candidate_strategy = builder(strike_long, strike_short, expiry, premium_long, premium_short, quantity=1)
                else:
                    candidate_strategy = builder(strike, expiry, premium, quantity=1)
            except Exception:
                continue
            ev_thesis, downside_thesis = evaluate_strategy_payoff(candidate_strategy, terminal_prices_thesis)
            ev_normal, downside_normal = evaluate_strategy_payoff(candidate_strategy, terminal_prices_normal)
            score = ev_thesis - lambda_val * abs(downside_thesis)
            strategy_results.append((candidate_strategy, score, contract, ev_thesis, ev_normal))
    optimized = optimize_strategy(strategy_results, bankroll=None)
    return optimized

############################################################
# 4. EXAMPLE MAIN USAGE (FOR TESTING PURPOSES)
############################################################

if __name__ == "__main__":
    ticker = "AAPL"
    underlying_df = get_historical_underlying_data(ticker)
    today_date = datetime.date.today()
    historical_start_date = today_date - datetime.timedelta(days=61)
    df_opts = get_historical_options_data(ticker, "8CTF0D0WVXWCLBVF", historical_start_date.isoformat(), today_date.isoformat())
    df_opts = filter_options_by_volume(df_opts)
    exp_dfs, exp_keys = create_expiration_dataframes(df_opts)
    underlying_recent = underlying_df[underlying_df.index >= pd.Timestamp(historical_start_date)].copy()
    underlying_recent.loc[:, '30MA'] = underlying_recent['Close'].rolling(window=30).mean()
    current_price = float(underlying_recent['Close'].iloc[-1])
    thesis_scenarios = [
        {'prob': 0.25, 'description': 'moderate', 'move_pct': 10},
        {'prob': 0.50, 'description': 'big move', 'move_pct': 30, 'direction_prob': 0.5},
        {'prob': 0.25, 'description': 'follow_ma', 'ma_value': underlying_recent['30MA'].iloc[-1]}
    ]
    risk_tolerance = "medium"
    simulation_horizon_days = 30
    future_entry_date = None  # Or e.g., datetime.date(2025, 3, 10)
    results = rank_strategies_for_thesis(ticker, df_opts, current_price, simulation_horizon_days, thesis_scenarios, risk_tolerance, future_entry_date=future_entry_date)
    print("Top Strategy Recommendations:")
    for candidate in results:
        strat, score, contract, ev_thesis, ev_normal = candidate
        print(f"Strategy: {strat.name}, Score: {score:.2f}, EV (Thesis): {ev_thesis:.2f}, EV (Normal): {ev_normal:.2f}")
        print(f"Contract Details: Strike={contract.get('strike', 'N/A')}, Expiry={contract.get('expiration', 'N/A')}, Premium={contract.get('premium', contract.get('last', 'N/A'))}")
