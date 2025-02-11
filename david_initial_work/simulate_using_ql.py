# simulation/simulate.py

import numpy as np
import QuantLib as ql
from thesis import Thesis

def simulate_terminal_prices(S0, T, base_risk_free_rate, base_volatility, thesis: Thesis,
                             num_paths=1000):
    """
    Simulate terminal prices at time T under multiple scenarios from the Thesis.
    Weighted by scenario probability.

    Steps:
      1) For each path, randomly pick a scenario (weighted by scenario probability).
      2) Adjust the drift (log(1 + move_pct)) if scenario indicates directional move.
      3) Adjust the volatility if scenario indicates a vol shift.
      4) Use QuantLib to generate a GBM path and take final price.
      5) Optionally multiply final by the scenario's drift factor or do it within the process.

    Return a list/array of terminal prices that includes all scenarios, with scenario weighting.
    """
    scenarios = thesis.scenarios
    if not scenarios:
        # If no scenarios, default to normal GBM with base_risk_free_rate and base_volatility
        scenarios = []
        from thesis import Scenario
        scenarios.append(Scenario(probability=1.0, price_move_pct=0.0, vol_adjust_pct=0.0))

    # Prepare the Weighted Sampler of scenarios
    scenario_probs = [sc.probability for sc in scenarios]
    scenario_cumprobs = np.cumsum(scenario_probs)
    if scenario_cumprobs[-1] < 0.9999:
        # If total prob < 1, there's some leftover scenario or error
        # We can renormalize or just proceed
        scenario_cumprobs = scenario_cumprobs / scenario_cumprobs[-1]

    terminal_prices = []

    # Set up for QuantLib
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

    for _ in range(num_paths):
        # pick a scenario
        r = np.random.rand()
        chosen = None
        for i, cp in enumerate(scenario_cumprobs):
            if r <= cp:
                chosen = scenarios[i]
                break
        if chosen is None:
            chosen = scenarios[-1]

        # scenario drift + vol
        # e.g. if user said +10% move, we might interpret that as an additive drift
        drift_scenario = np.log(1 + chosen.price_move_pct) / T if chosen.price_move_pct != 0 else 0.0
        scenario_vol = base_volatility * (1 + chosen.vol_adjust_pct)

        # build the process in QuantLib
        underlying = ql.SimpleQuote(S0)
        risk_free_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, base_risk_free_rate + drift_scenario, day_count)
        )
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, calendar, scenario_vol, day_count)
        )
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(underlying), dividend_ts, risk_free_ts, vol_ts
        )

        import copy

        dt = T
        steps = 1  # or more if you want path detail
        time_grid = ql.TimeGrid(dt, steps)
        dim = len(time_grid) - 1

        # Use SobolRsg instead of MersenneTwisterUniformRng
        sobol_rng = ql.SobolRsg(dim, False)  # note: 'False' means no scrambling
        uniform_seq = ql.UniformRandomSequenceGenerator(dim, sobol_rng)
        rng = ql.GaussianRandomSequenceGenerator(uniform_seq)
        path_generator = ql.GaussianPathGenerator(process, dt, dim, rng, False)
        sample_path = path_generator.next()
        path = sample_path.value()
        S_T = path.back()

        terminal_prices.append(S_T)


        


    return np.array(terminal_prices)


def evaluate_strategy(strategy, terminal_prices):
    """
    Compute average payoff and min payoff (or other risk metrics).
    """
    payoffs = [strategy.payoff_at_expiration(p) for p in terminal_prices]
    mean_payoff = np.mean(payoffs)
    min_payoff = np.min(payoffs)
    return mean_payoff, min_payoff