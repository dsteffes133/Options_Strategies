import pulp

def optimize_strategy(candidates, bankroll):
    """
    Given a list of candidate strategies, select the optimal subset 
    subject to the total cost not exceeding the bankroll.
    
    Each candidate is a tuple: (strategy, score, contract, ev_thesis, ev_normal)
    For this example, we assume the cost is the premium of the contract.
    
    Returns a list of selected candidate strategies.
    """
    # Create a linear programming problem (maximize total score)
    prob = pulp.LpProblem("StrategyOptimization", pulp.LpMaximize)
    
    # Decision variables: binary variable for each candidate (0 or 1)
    num_candidates = len(candidates)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(num_candidates)]
    
    # Objective: maximize total score (you can adjust this to use expected value, risk-adjusted value, etc.)
    prob += pulp.lpSum([candidates[i][1] * x[i] for i in range(num_candidates)]), "TotalScore"
    
    # Constraint: total cost does not exceed the bankroll.
    # Here we assume cost is given by the premium of the contract.
    # (In practice, you might multiply by a contract multiplier, or use more refined cost data.)
    prob += pulp.lpSum([float(candidates[i][2].get('premium', 0)) * x[i] for i in range(num_candidates)]) <= bankroll, "BudgetConstraint"
    
    # Solve the problem
    prob.solve()
    
    selected = []
    for i, var in enumerate(x):
        if var.varValue == 1:
            selected.append(candidates[i])
    
    return selected
