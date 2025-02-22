# app.py

import streamlit as st
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data.fetch import (
    get_historical_underlying_data,
    get_historical_options_data,
    filter_options_by_volume,
    create_expiration_dataframes,
)
from models.strategy import STRATEGY_BUILDERS
from models.strategy import Strategy, OptionLeg, UnderlyingLeg
from thesis import Scenario, Thesis
from simulation.simulate import simulate_terminal_prices, evaluate_strategy
from optimization.optimize import pick_top_strategies_greedy  # or optimize_strategy_lp

# =============================================================================
# User Interface
# =============================================================================
st.title("Modularized Option Strategy Recommender")

st.sidebar.header("1. Input Ticker & Dates")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Historical Start Date", value=datetime.date.today() - datetime.timedelta(days=90))
end_date = st.sidebar.date_input("Historical End Date", value=datetime.date.today())

st.sidebar.header("2. Projection / Simulation")
projection_days = st.sidebar.number_input("Days to Project", min_value=1, value=30)
T = projection_days / 365.0  # Projection horizon for the underlying (in years)
# Note: Option premiums come directly from the API (current market prices) – they are not simulated.
num_paths = st.sidebar.number_input("Simulation Paths", min_value=100, value=1000, step=100)
base_risk_free_rate = st.sidebar.slider("Base Risk-Free Rate", min_value=0.0, max_value=0.10, value=0.02)
base_volatility = st.sidebar.slider("Base Volatility", min_value=0.0, max_value=1.0, value=0.30)

st.sidebar.header("3. Bankroll & Strategy Selection")
bankroll = st.sidebar.number_input("Bankroll ($)", value=100000)
risk_preference = st.sidebar.selectbox("Risk Preference", ["Low", "Medium", "High"])
pick_method = st.sidebar.selectbox("Strategy Pick Method", ["Greedy Top N", "Linear Program"])

st.write("## Define Your Thesis Scenarios")
st.markdown("""
Enter multiple scenarios. For each scenario:
- Probability (0-1)
- Expected final price move (%) from current
- Volatility adjustment (%): e.g., +20% => 0.2
""")

scenario_list = []
num_scenarios = st.number_input("Number of Scenarios", min_value=1, max_value=5, value=2)
for i in range(num_scenarios):
    st.subheader(f"Scenario {i+1}")
    prob = st.slider(f"Probability {i+1}", 0.0, 1.0, 0.5, 0.01, key=f"prob_{i}")
    move_pct = st.number_input(f"Price Move % (decimal form) {i+1}", value=0.1, key=f"move_{i}")
    vol_pct = st.number_input(f"Vol Shift % (decimal form) {i+1}", value=0.2, key=f"vol_{i}")
    scenario_dir_label = st.selectbox(f"Direction Label {i+1}", ["bullish", "bearish", "neutral", "other"], key=f"dir_{i}")
    scenario_name = st.text_input(f"Scenario Label {i+1}", value=f"Scenario_{i+1}", key=f"label_{i}")
    if prob > 0:
        sc = Scenario(probability=prob, price_move_pct=move_pct,
                      vol_adjust_pct=vol_pct,
                      direction_label=scenario_dir_label,
                      scenario_label=scenario_name)
        scenario_list.append(sc)

# =============================================================================
# Helper Function: Detailed Contract Description
# =============================================================================
def detailed_contract_description(strategy, expiration):
    """
    Returns a detailed, human-readable description of the contracts in a strategy.
    The expiration (as a string) is appended to each option leg.
    """
    descriptions = []
    for leg in strategy.legs:
        if hasattr(leg, 'is_call'):  # OptionLeg
            option_type = "Call" if leg.is_call else "Put"
            position = "Long" if leg.is_long else "Short"
            # Use the API premium value directly (assumed to be current market price)
            leg_desc = f"{position} {leg.quantity} {option_type} @ Strike {leg.strike}, Premium {leg.premium}, Expiring on {expiration}"
            descriptions.append(leg_desc)
        elif hasattr(leg, 'entry_price'):  # UnderlyingLeg
            position = "Long" if leg.is_long else "Short"
            leg_desc = f"{position} Underlying, Quantity {leg.quantity}, Entry Price {leg.entry_price}"
            descriptions.append(leg_desc)
        else:
            descriptions.append("Unknown leg type")
    return f"To perform the strategy '{strategy.name}', you must: " + "; ".join(descriptions)

# =============================================================================
# Run Analysis
# =============================================================================
if st.button("Run Analysis"):
    # Build Thesis object
    thesis_obj = Thesis(scenario_list)
    st.write(f"Total Probability: {thesis_obj.total_probability():.2f}")
    if len(thesis_obj.scenarios) < 1:
        st.warning("No scenarios provided. Please add at least one scenario.")
        st.stop()

    # Fetch historical underlying data
    st.write("Fetching historical data...")
    try:
        underlying_df = get_historical_underlying_data(
            ticker,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
    except Exception as e:
        st.error(f"Error fetching underlying: {e}")
        st.stop()

    if underlying_df.empty:
        st.warning("No underlying data returned.")
        st.stop()

    current_price = float(underlying_df['Close'].iloc[-1])
    st.write(f"Current Price: {current_price:.2f}")

    # Fetch and filter options data (current option premiums come from the API)
    st.write("Fetching & filtering options data...")
    options_df = get_historical_options_data(ticker, start_date=start_date, end_date=end_date)
    options_df = filter_options_by_volume(options_df, min_total_volume=250)
    exp_dict = create_expiration_dataframes(options_df)

    # =============================================================================
    # Candidate Strategy Generation (using current API premium values)
    # =============================================================================
    candidate_strategies = []
    if options_df.empty:
        st.warning("No options data found after filtering. Using placeholder candidates.")
    else:
        st.write("Processing candidate strategies across multiple expiration dates:")
        # Iterate over every expiration date available.
        for exp_key, df_exp in exp_dict.items():
            expiration_str = str(exp_key)  # Convert expiration key to string for display
            st.write(f"Considering expiration: {expiration_str}")
            for idx, row in df_exp.iterrows():
                try:
                    strike = float(row['strike'])
                    # Use the current premium from the API directly.
                    premium = float(row.get('premium', row.get('last', 0.0)))
                except Exception:
                    continue

                # Only consider strikes within ±10% of current price.
                if abs(strike - current_price) > current_price * 0.1:
                    continue

                # Determine directional bias from the first scenario in your thesis.
                if thesis_obj.scenarios:
                    direction = thesis_obj.scenarios[0].direction_label.lower()
                else:
                    direction = "bullish"

                # Build candidate strategies using the current premium values.
                if direction in ["bullish", "neutral"]:
                    # Example 1: Long Call
                    if "LongCall" in STRATEGY_BUILDERS:
                        strat = STRATEGY_BUILDERS["LongCall"](strike, premium, quantity=1)
                        cost = premium
                        candidate_strategies.append((strat, 0, cost, 0, 0, expiration_str))
                    # 2. Bull Call Spread
                    if "BullCallSpread" in STRATEGY_BUILDERS:
                        strike2 = strike + 5
                        premium2 = premium * 0.8
                        if strike2 > 0 and premium2 >= 0:
                            strat = STRATEGY_BUILDERS["BullCallSpread"](strike, premium, strike2, premium2, quantity=1)
                            cost = premium - premium2
                            candidate_strategies.append((strat, 0, cost, 0, 0, expiration_str))
                    # 3. Bull Put Spread
                    if "BullPutSpread" in STRATEGY_BUILDERS:
                        strike2 = strike - 5
                        premium2 = premium * 1.2
                        if strike2 > 0:
                            strat = STRATEGY_BUILDERS["BullPutSpread"](strike, premium, strike2, premium2, quantity=1)
                            cost = premium2 - premium
                            candidate_strategies.append((strat, 0, cost, 0, 0, expiration_str))
                    # 4. Long Risk Reversal (buy call at strike+5, sell put at strike-5)
                    if "LongRiskReversal" in STRATEGY_BUILDERS:
                        call_strike = strike + 5
                        call_premium = premium * 0.9
                        put_strike = strike - 5
                        put_premium = premium * 1.1
                        strat = STRATEGY_BUILDERS["LongRiskReversal"](put_strike, put_premium, call_strike, call_premium, quantity=1)
                        candidate_strategies.append((strat, 0, call_premium + put_premium, 0, 0, expiration_str))
                    # 5. Long Butterfly
                    if "LongButterfly" in STRATEGY_BUILDERS:
                        strike_A = strike
                        strike_B = strike + 5
                        strike_C = strike + 10
                        premium_A = premium
                        premium_B = premium * 0.9
                        premium_C = premium * 0.8
                        strat = STRATEGY_BUILDERS["LongButterfly"](strike_A, premium_A, strike_B, premium_B, strike_C, premium_C, quantity=1, is_call=True)
                        candidate_strategies.append((strat, 0, premium_A + premium_C - 2 * premium_B, 0, 0, expiration_str))
                    # 6. Long Straddle
                    if "LongStraddle" in STRATEGY_BUILDERS:
                        call_premium = premium
                        put_premium = premium
                        strat = STRATEGY_BUILDERS["LongStraddle"](strike, call_premium, put_premium, quantity=1)
                        candidate_strategies.append((strat, 0, call_premium + put_premium, 0, 0, expiration_str))
                    # 7. Long Strangle
                    if "LongStrangle" in STRATEGY_BUILDERS:
                        put_strike = strike - 5
                        call_strike = strike + 5
                        put_premium = premium * 1.1
                        call_premium = premium * 0.9
                        strat = STRATEGY_BUILDERS["LongStrangle"](put_strike, put_premium, call_strike, call_premium, quantity=1)
                        candidate_strategies.append((strat, 0, put_premium + call_premium, 0, 0, expiration_str))
                    # 8. Ratio Call Spread
                    if "RatioCallSpread" in STRATEGY_BUILDERS:
                        long_strike = strike
                        short_strike = strike + 5
                        long_premium = premium
                        short_premium = premium * 0.8
                        strat = STRATEGY_BUILDERS["RatioCallSpread"](long_strike, long_premium, short_strike, short_premium, ratio=2, quantity=1)
                        candidate_strategies.append((strat, 0, long_premium - 2 * short_premium, 0, 0, expiration_str))
                    # 9. Call Ratio Backspread
                    if "CallRatioBackspread" in STRATEGY_BUILDERS:
                        short_strike = strike
                        long_strike = strike + 5
                        short_premium = premium * 0.8
                        long_premium = premium * 0.9
                        strat = STRATEGY_BUILDERS["CallRatioBackspread"](short_strike, short_premium, long_strike, long_premium, ratio=2, quantity=1)
                        candidate_strategies.append((strat, 0, short_premium - 2 * long_premium, 0, 0, expiration_str))
                    # 10. Long Instrument Conversion
                    if "LongInstrumentConversion" in STRATEGY_BUILDERS:
                        strat = STRATEGY_BUILDERS["LongInstrumentConversion"](current_price, strike, premium * 1.1, premium * 0.9, quantity=1)
                        candidate_strategies.append((strat, 0, premium, 0, 0, expiration_str))
                # =============================
                # For bearish views:
                # =============================
                elif direction in ["bearish"]:
                    # 1. Long Put
                    if "LongPut" in STRATEGY_BUILDERS:
                        strat = STRATEGY_BUILDERS["LongPut"](strike, premium, quantity=1)
                        candidate_strategies.append((strat, 0, premium, 0, 0, expiration_str))
                    # 2. Bear Call Spread
                    if "BearCallSpread" in STRATEGY_BUILDERS:
                        strike2 = strike + 5
                        premium2 = premium * 0.8
                        strat = STRATEGY_BUILDERS["BearCallSpread"](strike, premium, strike2, premium2, quantity=1)
                        candidate_strategies.append((strat, 0, premium - premium2, 0, 0, expiration_str))
                    # 3. Bear Put Spread
                    if "BearPutSpread" in STRATEGY_BUILDERS:
                        strike2 = strike - 5
                        premium2 = premium * 1.2
                        strat = STRATEGY_BUILDERS["BearPutSpread"](strike, premium, strike2, premium2, quantity=1)
                        candidate_strategies.append((strat, 0, premium2 - premium, 0, 0, expiration_str))
                    # 4. Short Risk Reversal (sell call, buy put)
                    if "ShortRiskReversal" in STRATEGY_BUILDERS:
                        put_strike = strike - 5
                        call_strike = strike + 5
                        put_premium = premium * 1.1
                        call_premium = premium * 0.9
                        strat = STRATEGY_BUILDERS["ShortRiskReversal"](put_strike, put_premium, call_strike, call_premium, quantity=1)
                        candidate_strategies.append((strat, 0, put_premium + call_premium, 0, 0, expiration_str))
                    # 5. Short Straddle
                    if "ShortStraddle" in STRATEGY_BUILDERS:
                        call_premium = premium
                        put_premium = premium
                        strat = STRATEGY_BUILDERS["ShortStraddle"](strike, call_premium, put_premium, quantity=1)
                        candidate_strategies.append((strat, 0, call_premium + put_premium, 0, 0, expiration_str))
                    # 6. Short Strangle
                    if "ShortStrangle" in STRATEGY_BUILDERS:
                        put_strike = strike - 5
                        call_strike = strike + 5
                        put_premium = premium * 1.1
                        call_premium = premium * 0.9
                        strat = STRATEGY_BUILDERS["ShortStrangle"](put_strike, put_premium, call_strike, call_premium, quantity=1)
                        candidate_strategies.append((strat, 0, put_premium + call_premium, 0, 0, expiration_str))
                    # 7. Ratio Put Spread
                    if "RatioPutSpread" in STRATEGY_BUILDERS:
                        long_strike = strike
                        short_strike = strike - 5
                        long_premium = premium
                        short_premium = premium * 1.2
                        strat = STRATEGY_BUILDERS["RatioPutSpread"](long_strike, long_premium, short_strike, short_premium, ratio=2, quantity=1)
                        candidate_strategies.append((strat, 0, long_premium - 2 * short_premium, 0, 0, expiration_str))
                    # 8. Put Ratio Backspread
                    if "PutRatioBackspread" in STRATEGY_BUILDERS:
                        lower_strike = strike - 5
                        higher_strike = strike
                        lower_premium = premium * 1.2
                        higher_premium = premium * 0.9
                        strat = STRATEGY_BUILDERS["PutRatioBackspread"](lower_strike, lower_premium, higher_strike, higher_premium, ratio=2, quantity=1)
                        candidate_strategies.append((strat, 0, lower_premium - 2 * higher_premium, 0, 0, expiration_str))
                    # 9. Short Butterfly
                    if "ShortButterfly" in STRATEGY_BUILDERS:
                        strike_A = strike
                        strike_B = strike + 5
                        strike_C = strike + 10
                        premium_A = premium
                        premium_B = premium * 0.9
                        premium_C = premium * 0.8
                        strat = STRATEGY_BUILDERS["ShortButterfly"](strike_A, premium_A, strike_B, premium_B, strike_C, premium_C, quantity=1, is_call=True)
                        candidate_strategies.append((strat, 0, premium_A + premium_C - 2 * premium_B, 0, 0, expiration_str))
                    # 10. Short Iron Butterfly
                    if "ShortIronButterfly" in STRATEGY_BUILDERS:
                        strike_A = strike
                        strike_B = strike + 5
                        strike_C = strike + 10
                        premium_A = premium
                        premium_B_call = premium * 0.9
                        premium_B_put = premium * 1.1
                        premium_C = premium * 0.8
                        strat = STRATEGY_BUILDERS["ShortIronButterfly"](strike_A, premium_A, strike_B, premium_B_put, strike_B, premium_B_call, strike_C, premium_C, quantity=1)
                        candidate_strategies.append((strat, 0, premium_A + premium_C - (premium_B_call + premium_B_put), 0, 0, expiration_str))
                    # 11. Short Box
                    if "ShortBox" in STRATEGY_BUILDERS:
                        strike_A = strike
                        strike_B = strike + 5
                        call_premium_A = premium
                        put_premium_A = premium
                        call_premium_B = premium * 0.9
                        put_premium_B = premium * 0.9
                        strat = STRATEGY_BUILDERS["ShortBox"](strike_A, call_premium_A, put_premium_A, strike_B, call_premium_B, put_premium_B, quantity=1)
                        candidate_strategies.append((strat, 0, call_premium_A + put_premium_A - (call_premium_B + put_premium_B), 0, 0, expiration_str))
                    # 12. Short Instrument Conversion
                    if "ShortInstrumentConversion" in STRATEGY_BUILDERS:
                        strat = STRATEGY_BUILDERS["ShortInstrumentConversion"](current_price, strike, premium * 0.9, premium * 1.1, quantity=1)
                        candidate_strategies.append((strat, 0, premium, 0, 0, expiration_str))
                # Regardless of direction, add a baseline "Nothing" strategy.
                if "Nothing" in STRATEGY_BUILDERS:
                    strat = STRATEGY_BUILDERS["Nothing"](base_risk_free_rate, T, cash=1)
                    candidate_strategies.append((strat, 0, 0, 0, 0, expiration_str))
                    
    if not candidate_strategies:
        st.warning("No candidate strategies were generated.")
    else:
        st.write(f"Found {len(candidate_strategies)} candidate strategies.")

    # =============================================================================
    # Underlying Price Simulation (using GBM)
    # =============================================================================
    st.write("Simulating terminal underlying prices under your thesis scenarios...")
    terminal_prices = simulate_terminal_prices(
        S0=current_price,
        T=T,
        base_risk_free_rate=base_risk_free_rate,
        base_volatility=base_volatility,
        thesis=thesis_obj,
        num_paths=num_paths
    )

    # =============================================================================
    # Evaluate Candidate Strategies
    # =============================================================================
    results = []
    # Set risk tolerance factor based on user preference.
    if risk_preference == "Low":
        lambda_val = 1.5
    elif risk_preference == "High":
        lambda_val = 0.5
    else:
        lambda_val = 1.0

    for cand in candidate_strategies:
        # Each candidate is (strategy_obj, score, cost, ev, downside, expiration)
        strategy_obj = cand[0]
        cost = cand[2]
        ev, downside = evaluate_strategy(strategy_obj, terminal_prices)
        score = ev - lambda_val * abs(downside)
        results.append((strategy_obj, score, cost, ev, downside))

    # =============================================================================
    # Select Top 3 Candidates
    # =============================================================================
    if pick_method == "Linear Program":
        st.write("Using linear program approach (not fully implemented).")
        candidate_lp = []
        for item in results:
            strategy_obj, s_score, s_cost, s_ev, s_down = item
            candidate_lp.append((strategy_obj, s_score, s_cost, s_ev, s_down))
        from optimization.optimize import optimize_strategy_lp
        selected = optimize_strategy_lp(candidate_lp, bankroll=bankroll)
    else:
        st.write("Using greedy top 3 approach.")
        selected = pick_top_strategies_greedy(results, top_n=3)

    # =============================================================================
    # Display Only the Top 3 Strategies
    # =============================================================================
    if not selected:
        st.warning("No strategies selected.")
    else:
        st.subheader("Top 3 Strategies")
        for idx, sel in enumerate(selected, 1):
            strat_obj, score, cost, ev, downside = sel
            st.write(f"**{idx}. {strat_obj.name}**")
            # Retrieve expiration info for the candidate strategy.
            expiration = ""
            for cand in candidate_strategies:
                if cand[0] == strat_obj:
                    expiration = cand[5]
                    break
            description = detailed_contract_description(strat_obj, expiration)
            st.write(description)
            st.write(f"Score: {score:.2f}, Cost: {cost:.2f}, EV: {ev:.2f}, Min Payoff: {downside:.2f}")

            # Plot payoff diagram
            px_low = current_price * 0.5
            px_high = current_price * 1.5
            prices, payoffs = strat_obj.payoff_diagram(px_low, px_high, steps=50)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices, y=payoffs, mode='lines', name='Payoff at Expiration'))
            fig.update_layout(
                title=f"Payoff Diagram - {strat_obj.name}",
                xaxis_title="Underlying Price at Expiration",
                yaxis_title="Payoff"
            )
            st.plotly_chart(fig)

        # Optional: Plot distribution of terminal underlying prices.
        st.subheader("Distribution of Simulated Terminal Prices")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=terminal_prices, nbinsx=50))
        hist_fig.update_layout(
            title="Terminal Price Distribution",
            xaxis_title="Price",
            yaxis_title="Frequency"
        )
        st.plotly_chart(hist_fig)
