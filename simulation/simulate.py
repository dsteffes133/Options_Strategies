import pandas as pd
import numpy as np
import QuantLib as ql

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

def simulate_terminal_price_with_path(S0, T, risk_free_rate, volatility, thesis_scenarios, num_paths=10000, use_thesis=True):
    """
    Simulates underlying price paths over a horizon T (in years) using a mixture of GBM processes.
    Returns a tuple:
      (terminal_prices, paths)
    where:
      - terminal_prices is a list of the final prices (adjusted),
      - paths is a list of full simulated price paths (each path is a list of prices over time).
    """
    terminal_prices = []
    paths = []
    for _ in range(num_paths):
        # (Determine adjusted_drift based on thesis scenarios, as in your code...)
        if use_thesis:
            probs = np.array([sc['prob'] for sc in thesis_scenarios])
            probs = probs / probs.sum()
            cum_probs = np.cumsum(probs)
            r_val = np.random.rand()
            scenario = None
            for sc, cp in zip(thesis_scenarios, cum_probs):
                if r_val <= cp:
                    scenario = sc
                    break
            if scenario['description'] == 'moderate':
                move_pct = scenario.get('move_pct', 10)
                adjusted_drift = np.log(1 + move_pct/100.0) / T
            elif scenario['description'] == 'big move':
                move_pct = scenario.get('move_pct', 30)
                direction_prob = scenario.get('direction_prob', 0.5)
                if np.random.rand() < direction_prob:
                    move_pct = move_pct
                else:
                    move_pct = -move_pct
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

        today = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = today
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        expiry_date = calendar.advance(today, int(T * 365), ql.Days)
        underlying = ql.SimpleQuote(S0)
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
        vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, volatility, day_count))
        process = ql.BlackScholesMertonProcess(ql.QuoteHandle(underlying), dividend_ts, flat_ts, vol_handle)
        time_grid = ql.TimeGrid(T, 100)
        dim = len(time_grid) - 1
        mt_rng = ql.MersenneTwisterUniformRng()
        uniform_seq_gen = ql.UniformRandomSequenceGenerator(dim, mt_rng)
        rng = ql.GaussianRandomSequenceGenerator(uniform_seq_gen)
        seq = ql.GaussianPathGenerator(process, T, dim, rng, False)
        sample_path = seq.next()
        path = sample_path.value()  # The full price path.
        # Save the entire path:
        paths.append(list(path))
        S_T = path.back()
        S_T_adjusted = S_T * np.exp(adjusted_drift * T)
        terminal_prices.append(S_T_adjusted)
    return terminal_prices, paths


def evaluate_strategy_payoff(strategy, terminal_prices):
    """
    Computes the expected payoff and minimum payoff (downside) for a given strategy,
    given simulated terminal prices.
    """
    payoffs = [strategy.payoff_at_expiration(price) for price in terminal_prices]
    ev = np.mean(payoffs)
    downside = min(payoffs)
    return ev, downside