# simulation/simulate.py

import numpy as np
from thesis import Thesis, Scenario

def simulate_terminal_prices(S0, T, base_risk_free_rate, base_volatility, thesis: Thesis, num_paths=1000):
    """
    Simulate terminal prices at time T under multiple scenarios from the Thesis.
    Each simulation path does the following:
      1) Randomly pick a scenario (weighted by scenario probability).
      2) Adjust the drift using the scenario's expected price move percentage.
         - If the scenario specifies a move of, say, +10%, we use drift_adjustment = ln(1 + 0.10)/T.
      3) Adjust the volatility using the scenario's vol_adjust_pct.
         - For example, a vol_adjust_pct of +0.20 increases volatility by 20%.
      4) Use the geometric Brownian motion formula:
             S(T) = S0 * exp((r + drift_adjustment - 0.5 * sigma²)*T + sigma*sqrt(T)*Z)
         where Z ~ N(0,1).
    
    Parameters:
      S0: float
          The current underlying price.
      T: float
          Time to expiration in years.
      base_risk_free_rate: float
          The base risk-free rate (annualized).
      base_volatility: float
          The base volatility (annualized).
      thesis: Thesis
          A Thesis object that contains one or more scenarios.
      num_paths: int, optional
          The number of simulation paths to generate (default 1000).
    
    Returns:
      np.array of terminal prices.
    """
    # If no scenarios are provided in the thesis, default to a neutral scenario.
    scenarios = thesis.scenarios
    if not scenarios:
        scenarios = [Scenario(probability=1.0, price_move_pct=0.0, vol_adjust_pct=0.0)]
    
    # Build cumulative probabilities for weighted random selection.
    scenario_probs = [sc.probability for sc in scenarios]
    scenario_cumprobs = np.cumsum(scenario_probs)
    total_prob = scenario_cumprobs[-1]
    if abs(total_prob - 1.0) > 1e-6:
        scenario_cumprobs = scenario_cumprobs / total_prob

    terminal_prices = []

    for _ in range(num_paths):
        # Pick a scenario based on weighted probabilities.
        r_rand = np.random.rand()
        chosen = None
        for i, cp in enumerate(scenario_cumprobs):
            if r_rand <= cp:
                chosen = scenarios[i]
                break
        if chosen is None:
            chosen = scenarios[-1]

        # Calculate drift adjustment from the scenario.
        if chosen.price_move_pct != 0:
            drift_adjustment = np.log(1 + chosen.price_move_pct) / T
        else:
            drift_adjustment = 0.0
        
        # Adjust volatility based on the scenario.
        scenario_vol = base_volatility * (1 + chosen.vol_adjust_pct)

        # Calculate the drift and diffusion for the GBM process.
        drift = (base_risk_free_rate + drift_adjustment) - 0.5 * (scenario_vol ** 2)
        diffusion = scenario_vol * np.sqrt(T)
        
        # Draw a standard normal sample.
        Z = np.random.normal(0, 1)
        
        # Compute the terminal price using the GBM formula.
        S_T = S0 * np.exp(drift * T + diffusion * Z)
        terminal_prices.append(S_T)
    
    return np.array(terminal_prices)


def evaluate_strategy(strategy, terminal_prices):
    """
    Evaluate a given option strategy over simulated terminal prices.
    
    Parameters:
      strategy: an object that implements a method `payoff_at_expiration(price)`
                which returns the payoff for a given terminal underlying price.
      terminal_prices: np.array
                An array of simulated terminal prices.
    
    Returns:
      A tuple (ev, downside) where:
         ev: Expected value (average payoff) of the strategy.
         downside: Minimum payoff across the simulated paths.
    """
    payoffs = [strategy.payoff_at_expiration(price) for price in terminal_prices]
    ev = np.mean(payoffs)
    downside = min(payoffs)
    return ev, downside
