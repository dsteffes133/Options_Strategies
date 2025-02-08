import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import datetime
from data_fetchy import (
    get_historical_underlying_data,
    get_historical_options_data)
from options_transformation import(
    create_expiration_dataframes,
)
from full_script import (
    # If you have Strategy, OptionLeg, etc. you can import them,
    # but here we mainly need rank_strategies_for_thesis
    rank_strategies_for_thesis,
    filter_options_by_volume
)


def main():
    st.title("Option Strategy Recommender")

    st.subheader("Thesis & Market View")

    ticker = st.text_input("Ticker Symbol", "AAPL")

    # ------------------------------------------------------------------
    # Collect 3 directional scenarios (or fewer, if user sets probability=0)
    # Each scenario: direction, magnitude, probability
    # ------------------------------------------------------------------
    scenarios = []

    def scenario_block(label):
        st.write(f"**{label}**")
        direction = st.selectbox(
            "Direction",
            ["Bullish", "Bearish", "Big Move (Straddle/Strangle)", "None/Unused"],
            key=f"direction_{label}"
        )
        magnitude = st.number_input(
            "Expected Move (%)",
            value=0, step=1,
            key=f"magnitude_{label}"
        )
        prob = st.slider(
            "Probability (%)",
            min_value=0,
            max_value=100,
            value=0,
            key=f"prob_{label}"
        )
        return direction, magnitude, prob

    st.write("Define up to three scenarios (set Probability=0 or pick 'None/Unused' if not using).")
    with st.expander("Scenario 1", expanded=True):
        direction_1, magnitude_1, prob_1 = scenario_block("scenario1")
    with st.expander("Scenario 2", expanded=False):
        direction_2, magnitude_2, prob_2 = scenario_block("scenario2")
    with st.expander("Scenario 3", expanded=False):
        direction_3, magnitude_3, prob_3 = scenario_block("scenario3")

    # Build a list of scenario dicts for whichever are valid
    def add_scenario_if_valid(direction, magnitude, prob):
        if direction != "None/Unused" and prob > 0:
            return {"direction": direction, 
                    "magnitude_pct": magnitude, 
                    "prob": prob / 100.0}
        else:
            return None

    scenario_list = []
    for (d, m, p) in [
        (direction_1, magnitude_1, prob_1),
        (direction_2, magnitude_2, prob_2),
        (direction_3, magnitude_3, prob_3)
    ]:
        sc = add_scenario_if_valid(d, m, p)
        if sc is not None:
            scenario_list.append(sc)

    # Check total probability
    total_prob = sum(sc["prob"] for sc in scenario_list)
    if abs(total_prob - 1.0) > 1e-8 and total_prob != 0:
        st.warning(f"Scenarios total probability = {total_prob*100:.1f}% (ideally 100%).")

    # ------------------------------------------------------------------
    # Bankroll & Dates
    # ------------------------------------------------------------------
    bankroll = st.number_input(
        "Bankroll ($)", 
        value=5000000, 
        step=10000, 
        help="Total capital available."
    )
    today = datetime.date.today()
    start_date = st.date_input("Earliest Entry Date", value=today)
    end_date = st.date_input("Latest Exit Date", value=today + datetime.timedelta(days=30))

    # ------------------------------------------------------------------
    # Button: Run Strategy Analysis
    # ------------------------------------------------------------------
    if st.button("Run Strategy Analysis"):
        st.write("Fetching data & analyzing strategies, please wait...")

        # 1) Get underlying data
        try:
            underlying_df = get_historical_underlying_data(ticker)
        except Exception as e:
            st.error(f"Error fetching underlying data: {e}")
            return

        # 2) Get options data
        api_key = "8CTF0D0WVXWCLBVF"
        try:
            options_df = get_historical_options_data(
                ticker,
                api_key,
                begin_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )
        except Exception as e:
            st.error(f"Error fetching options data: {e}")
            return

        # 3) Filter by volume, create expiration dfs
        options_df = filter_options_by_volume(options_df, min_total_volume=250)
        exp_dfs, exp_keys = create_expiration_dataframes(options_df)

        # If you just want the current underlying price:
        current_price = float(underlying_df["Close"].iloc[-1])

        # 4) Pass scenario_list to your rank function
        # We'll assume rank_strategies_for_thesis can handle a parameter "scenarios"
        results = rank_strategies_for_thesis(
            ticker=ticker,
            current_price=current_price,
            bankroll=bankroll,
            time_horizon_days=(end_date - start_date).days,
            scenarios=scenario_list
            # you can add more arguments if needed, like riskTolerance, etc.
        )

        # 5) Display results
        st.subheader("Top Strategy Recommendations")
        if not results:
            st.warning("No viable strategies found.")
        else:
            # Suppose results = [(StrategyObject, EV), ...] sorted descending by EV
            for idx, (strategy_obj, ev) in enumerate(results[:3], start=1):
                st.write(f"**{idx}.** {strategy_obj.name} - EV: {ev:,.2f}")
                st.write(f"Details: # legs = {len(strategy_obj.legs)} etc.")
                # If you want, generate a payoff diagram and show it
                # ...



if __name__ == "__main__":
    main()
