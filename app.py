# app.py
import numpy as np
import streamlit as st
import datetime
from simulation.simulate import simulate_terminal_price_with_path
import pandas as pd
import plotly.graph_objects as go
from full_script import (
    get_historical_underlying_data,
    get_historical_options_data,
    filter_options_by_volume,
    create_expiration_dataframes,
    rank_strategies_for_thesis
)

st.title("Option Strategy Recommender")
st.subheader("Thesis & Market View")

# --- User Inputs ---
ticker = st.text_input("Ticker Symbol", "AAPL")
risk_tolerance = st.selectbox("Risk Tolerance", options=["low", "medium", "high"], index=1)

st.write("Define up to three scenarios:")

def scenario_block(label):
    st.write(f"**{label}**")
    direction = st.selectbox(
        "Direction",
        ["Bullish", "Bearish", "Big Move", "Follow the 30MA", "Sideways/No Move", "None/Unused"],
        key=f"direction_{label}"
    )
    if direction == "Follow the 30MA":
        # For follow_ma, no magnitude is needed.
        magnitude = None
    else:
        magnitude = st.number_input("Expected Move (%)", value=0, step=1, key=f"magnitude_{label}")
    prob = st.slider("Probability (%)", min_value=0, max_value=100, value=0, key=f"prob_{label}")
    return direction, magnitude, prob

with st.expander("Scenario 1", expanded=True):
    direction1, magnitude1, prob1 = scenario_block("scenario1")
with st.expander("Scenario 2"):
    direction2, magnitude2, prob2 = scenario_block("scenario2")
with st.expander("Scenario 3"):
    direction3, magnitude3, prob3 = scenario_block("scenario3")

thesis_scenarios = []
def add_scenario(direction, magnitude, prob):
    if direction == "None/Unused" or prob == 0:
        return None
    if direction == "Bullish":
        return {'prob': prob/100.0, 'description': 'moderate', 'move_pct': magnitude if magnitude else 10}
    elif direction == "Bearish":
        return {'prob': prob/100.0, 'description': 'moderate', 'move_pct': -magnitude if magnitude else -10}
    elif direction == "Big Move":
        # For a big move, treat the magnitude as an absolute value.
        return {'prob': prob/100.0, 'description': 'big move', 'move_pct': abs(magnitude) if magnitude else 30, 'direction_prob': 0.5}
    elif direction == "Follow the 30MA":
        return {'prob': prob/100.0, 'description': 'follow_ma', 'ma_value': None}
    elif direction == "Sideways/No Move":
        return {'prob': prob/100.0, 'description': 'neutral'}
    else:
        return None

for (d, m, p) in [(direction1, magnitude1, prob1), (direction2, magnitude2, prob2), (direction3, magnitude3, prob3)]:
    sc = add_scenario(d, m, p)
    if sc is not None:
        thesis_scenarios.append(sc)

# Inform the user if directional scenarios don't sum to 100%
total_prob = sum(sc['prob'] for sc in thesis_scenarios) if thesis_scenarios else 0
if total_prob != 0 and abs(total_prob - 1.0) > 0.001:
    st.info(f"Note: The total scenario weight is {total_prob*100:.1f}%. For directional scenarios (bullish, bearish, sideways), weights should sum to 100%. 'Big Move' is expressed in absolute terms.")

bankroll = st.number_input("Bankroll ($)", value=5000000, step=10000)
projection_horizon_days = st.number_input("Projection Horizon (days)", value=30, min_value=1, step=1)
future_entry_date_input = st.date_input("Future Entry Date (optional)", value=datetime.date.today())
if future_entry_date_input <= datetime.date.today():
    future_entry_date = None
else:
    future_entry_date = future_entry_date_input

if st.button("Run Strategy Analysis"):
    st.write("Fetching historical data and analyzing strategies, please wait...")
    try:
        underlying_df = get_historical_underlying_data(ticker)
    except Exception as e:
        st.error(f"Error fetching underlying data: {e}")
        st.stop()
    historical_window = 61
    today_date = datetime.date.today()
    historical_start_date = today_date - datetime.timedelta(days=historical_window)
    underlying_recent = underlying_df[underlying_df.index >= pd.Timestamp(historical_start_date)].copy()
    underlying_recent.loc[:, '30MA'] = underlying_recent['Close'].rolling(window=30).mean()
    current_price = float(underlying_recent['Close'].iloc[-1])
    api_key = "8CTF0D0WVXWCLBVF"
    try:
        options_df = get_historical_options_data(ticker, api_key, historical_start_date.isoformat(), today_date.isoformat())
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        st.stop()
    options_df = filter_options_by_volume(options_df)
    exp_dfs, exp_keys = create_expiration_dataframes(options_df)
    # Update follow_ma scenarios with computed moving average:
    for sc in thesis_scenarios:
        if sc['description'] == 'follow_ma':
            sc['ma_value'] = underlying_recent['30MA'].iloc[-1]
    # Call the ranking function:
    results = rank_strategies_for_thesis(ticker, options_df, current_price, projection_horizon_days, thesis_scenarios, risk_tolerance, future_entry_date=future_entry_date)
    if not results:
        st.warning("No viable strategies found.")
    else:
        st.subheader("Top Strategy Recommendations")
        for idx, candidate in enumerate(results, start=1):
            strat, score, contract, ev_thesis, ev_normal = candidate
            st.write(f"**{idx}. {strat.name}**")
            st.write(f"Score: {score:.2f}")
            st.write(f"Expected Value (Thesis): {ev_thesis:.2f} | Expected Value (Normal GBM): {ev_normal:.2f}")
            st.write(f"Contract Details: Strike={contract.get('strike', 'N/A')}, Expiry={contract.get('expiration', 'N/A')}, Premium={contract.get('premium', contract.get('last', 'N/A'))}")
            # Create interactive payoff diagrams using Plotly:
            prices, payoffs = strat.payoff_diagram(current_price*0.8, current_price*1.2)
            fig_thesis = go.Figure()
            fig_thesis.add_trace(go.Scatter(x=prices, y=payoffs, mode='lines', name='Thesis'))
            fig_thesis.update_layout(title=f"Payoff Diagram (Thesis) for {strat.name}", xaxis_title="Underlying Price at Expiration", yaxis_title="Payoff")
            # For demonstration, we re-use the same diagram for normal GBM;
            # in a complete implementation, you would generate this from separate simulation data.
            fig_normal = go.Figure()
            fig_normal.add_trace(go.Scatter(x=prices, y=payoffs, mode='lines', name='Normal GBM'))
            fig_normal.update_layout(title=f"Payoff Diagram (Normal GBM) for {strat.name}", xaxis_title="Underlying Price at Expiration", yaxis_title="Payoff")
            st.plotly_chart(fig_thesis)
            st.plotly_chart(fig_normal)


        with st.expander("Show Sample Simulated Underlying Price Path"):
        # Determine the starting price for simulation:
            if future_entry_date_input <= datetime.date.today():
                future_entry_date = None
            else:
            # If a future entry date is specified, forecast the price:
            # (Assuming you have forecast_future_price in your module)
                from full_script import forecast_future_price
                S0_future = forecast_future_price(current_price, today_date, future_entry_date, risk_free_rate=0.02, volatility=0.30, num_paths=1000)

                T = projection_horizon_days / 365.0
                terminal_prices, simulated_paths = simulate_terminal_price_with_path(S0_future, T, risk_free_rate=0.02, volatility=0.30, thesis_scenarios=thesis_scenarios, num_paths=1000, use_thesis=True)
                sample_path = simulated_paths[0]
                time_points = np.linspace(0, T, len(sample_path))
                fig_path = go.Figure()
                fig_path.add_trace(go.Scatter(x=time_points, y=sample_path, mode='lines', name='Simulated Path'))
                fig_path.update_layout(title="Simulated Underlying Price Path", xaxis_title="Time (years)", yaxis_title="Price")
                st.plotly_chart(fig_path)

