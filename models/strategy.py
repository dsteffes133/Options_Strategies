import pandas as pd
import numpy as np
import QuantLib as ql


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