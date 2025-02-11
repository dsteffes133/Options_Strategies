# optimization/optimize.py

import pulp

def optimize_strategy_lp(candidates, bankroll):
    """
    Given a list of (strategy, score, cost, ev, min_payoff),
    pick a subset that maximizes total score subject to cost <= bankroll.
    """
    prob = pulp.LpProblem("StrategySelection", pulp.LpMaximize)
    x_vars = []
    for i in range(len(candidates)):
        x_vars.append(pulp.LpVariable(f"x_{i}", cat="Binary"))

    # Objective: sum of scores * x_i
    prob += pulp.lpSum([candidates[i][1] * x_vars[i] for i in range(len(candidates))]), "TotalScore"

    # Constraint: sum of cost * x_i <= bankroll
    prob += pulp.lpSum([candidates[i][2] * x_vars[i] for i in range(len(candidates))]) <= bankroll

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    chosen = []
    for i, var in enumerate(x_vars):
        if var.varValue == 1:
            chosen.append(candidates[i])
    return chosen


def pick_top_strategies_greedy(candidates, top_n=3):
    """
    A simpler approach: sort by 'score' descending and pick top_n.
    """
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return sorted_candidates[:top_n]
