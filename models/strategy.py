# models/strategy.py

import numpy as np

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
    Represents a single call or put option position (long or short).
    """
    def __init__(self, is_call, is_long, strike, premium, quantity=1):
        self.is_call = is_call
        self.is_long = is_long
        self.strike = strike
        self.premium = premium
        self.quantity = quantity

    def payoff_at_expiration(self, underlying_price):
        """
        Intrinsic payoff minus premium if long, plus premium if short.
        """
        if self.is_call:
            intrinsic = max(0, underlying_price - self.strike)
        else:
            intrinsic = max(0, self.strike - underlying_price)

        sign = 1 if self.is_long else -1
        return sign * (intrinsic - self.premium) * self.quantity


class Strategy:
    """
    A multi-leg strategy that can hold underlying and/or multiple options legs.
    """
    def __init__(self, name="Unnamed Strategy"):
        self.name = name
        self.legs = []

    def add_leg(self, leg):
        self.legs.append(leg)

    def payoff_at_expiration(self, underlying_price):
        return sum(leg.payoff_at_expiration(underlying_price) for leg in self.legs)

    def payoff_diagram(self, min_price, max_price, steps=50):
        prices = np.linspace(min_price, max_price, steps)
        payoffs = [self.payoff_at_expiration(p) for p in prices]
        return prices, payoffs


# Example strategy builders:
def create_long_underlying(entry_price, quantity=1):
    s = Strategy(name="Long Underlying")
    underlying_leg = UnderlyingLeg(is_long=True, entry_price=entry_price, quantity=quantity)
    s.add_leg(underlying_leg)
    return s

def create_short_underlying(entry_price, quantity=1):
    s = Strategy(name="Short Underlying")
    underlying_leg = UnderlyingLeg(is_long=False, entry_price=entry_price, quantity=quantity)
    s.add_leg(underlying_leg)
    return s

def create_long_call(strike, premium, quantity=1):
    s = Strategy(name="Long Call")
    call_leg = OptionLeg(is_call=True, is_long=True, strike=strike, premium=premium, quantity=quantity)
    s.add_leg(call_leg)
    return s

def create_short_call(strike, premium, quantity=1):
    s = Strategy(name="Short Call")
    call_leg = OptionLeg(is_call=True, is_long=False, strike=strike, premium=premium, quantity=quantity)
    s.add_leg(call_leg)
    return s

def create_long_put(strike, premium, quantity=1):
    s = Strategy(name="Long Put")
    # For a put, set is_call=False.
    put_leg = OptionLeg(is_call=False, is_long=True, strike=strike, premium=premium, quantity=quantity)
    s.add_leg(put_leg)
    return s

def create_short_put(strike, premium, quantity=1):
    s = Strategy(name="Short Put")
    # For a put, set is_call=False; for a short position, is_long=False.
    put_leg = OptionLeg(is_call=False, is_long=False, strike=strike, premium=premium, quantity=quantity)
    s.add_leg(put_leg)
    return s

def create_long_risk_reversal(put_strike, put_premium, call_strike, call_premium, quantity=1):
    s = Strategy(name="Long Risk Reversal")
    short_put_leg = OptionLeg(is_call=False, is_long=False, strike=put_strike, premium=put_premium, quantity=quantity)
    long_call_leg = OptionLeg(is_call=True, is_long=True, strike=call_strike, premium=call_premium, quantity=quantity)
    s.add_leg(short_put_leg)
    s.add_leg(long_call_leg)
    return s

def create_short_risk_reversal(put_strike, put_premium, call_strike, call_premium, quantity=1):
    s = Strategy(name="Short Risk Reversal")
    long_put_leg = OptionLeg(is_call=False, is_long=True, strike=put_strike, premium=put_premium, quantity=quantity)
    short_call_leg = OptionLeg(is_call=True, is_long=False, strike=call_strike, premium=call_premium, quantity=quantity)
    s.add_leg(long_put_leg)
    s.add_leg(short_call_leg)
    return s


def create_bull_call_spread(strike_long, premium_long, strike_short, premium_short, quantity=1):
    s = Strategy(name="Bull Call Spread")
    long_call = OptionLeg(is_call=True, is_long=True, strike=strike_long, premium=premium_long, quantity=quantity)
    short_call = OptionLeg(is_call=True, is_long=False, strike=strike_short, premium=premium_short, quantity=quantity)
    s.add_leg(long_call)
    s.add_leg(short_call)
    return s

def create_bull_put_spread(long_put_strike, long_put_premium, short_put_strike, short_put_premium, quantity=1):
    s = Strategy(name="Bull Put Spread")
    long_put_leg = OptionLeg(is_call=False, is_long=True, strike=long_put_strike, premium=long_put_premium, quantity=quantity)
    short_put_leg = OptionLeg(is_call=False, is_long=False, strike=short_put_strike, premium=short_put_premium, quantity=quantity)
    s.add_leg(long_put_leg)
    s.add_leg(short_put_leg)
    return s

def create_bear_call_spread(short_call_strike, short_call_premium, long_call_strike, long_call_premium, quantity=1):
    s = Strategy(name="Bear Call Spread")
    short_call_leg = OptionLeg(is_call=True, is_long=False, strike=short_call_strike, premium=short_call_premium, quantity=quantity)
    long_call_leg = OptionLeg(is_call=True, is_long=True, strike=long_call_strike, premium=long_call_premium, quantity=quantity)
    s.add_leg(short_call_leg)
    s.add_leg(long_call_leg)
    return s

def create_bear_put_spread(short_put_strike, short_put_premium, long_put_strike, long_put_premium, quantity=1):
    s = Strategy(name="Bear Put Spread")
    short_put_leg = OptionLeg(is_call=False, is_long=False, strike=short_put_strike, premium=short_put_premium, quantity=quantity)
    long_put_leg = OptionLeg(is_call=False, is_long=True, strike=long_put_strike, premium=long_put_premium, quantity=quantity)
    s.add_leg(short_put_leg)
    s.add_leg(long_put_leg)
    return s

def create_long_butterfly(strike_A, premium_A, strike_B, premium_B, strike_C, premium_C, quantity=1, is_call=True):
    s = Strategy(name="Long Butterfly")
    leg_A = OptionLeg(is_call=is_call, is_long=True, strike=strike_A, premium=premium_A, quantity=quantity)
    leg_B = OptionLeg(is_call=is_call, is_long=False, strike=strike_B, premium=premium_B, quantity=2 * quantity)
    leg_C = OptionLeg(is_call=is_call, is_long=True, strike=strike_C, premium=premium_C, quantity=quantity)
    s.add_leg(leg_A)
    s.add_leg(leg_B)
    s.add_leg(leg_C)
    return s

def create_short_butterfly(strike_A, premium_A, strike_B, premium_B, strike_C, premium_C, quantity=1, is_call=True):
    s = Strategy(name="Short Butterfly")
    leg_A = OptionLeg(is_call=is_call, is_long=False, strike=strike_A, premium=premium_A, quantity=quantity)
    leg_B = OptionLeg(is_call=is_call, is_long=True, strike=strike_B, premium=premium_B, quantity=2 * quantity)
    leg_C = OptionLeg(is_call=is_call, is_long=False, strike=strike_C, premium=premium_C, quantity=quantity)
    s.add_leg(leg_A)
    s.add_leg(leg_B)
    s.add_leg(leg_C)
    return s

def create_long_iron_butterfly(strike_A, premium_A, strike_B, premium_B_put, premium_B_call, strike_C, premium_C, quantity=1):
    s = Strategy(name="Long Iron Butterfly")
    leg1 = OptionLeg(is_call=False, is_long=False, strike=strike_A, premium=premium_A, quantity=quantity)
    leg2 = OptionLeg(is_call=False, is_long=True, strike=strike_B, premium=premium_B_put, quantity=quantity)
    leg3 = OptionLeg(is_call=True, is_long=True, strike=strike_B, premium=premium_B_call, quantity=quantity)
    leg4 = OptionLeg(is_call=True, is_long=False, strike=strike_C, premium=premium_C, quantity=quantity)
    s.add_leg(leg1)
    s.add_leg(leg2)
    s.add_leg(leg3)
    s.add_leg(leg4)
    return s

def create_short_iron_butterfly(strike_A, premium_A, strike_B, premium_B_put, premium_B_call, strike_C, premium_C, quantity=1):
    s = Strategy(name="Short Iron Butterfly")
    leg1 = OptionLeg(is_call=False, is_long=True, strike=strike_A, premium=premium_A, quantity=quantity)
    leg2 = OptionLeg(is_call=False, is_long=False, strike=strike_B, premium=premium_B_put, quantity=quantity)
    leg3 = OptionLeg(is_call=True, is_long=False, strike=strike_B, premium=premium_B_call, quantity=quantity)
    leg4 = OptionLeg(is_call=True, is_long=True, strike=strike_C, premium=premium_C, quantity=quantity)
    s.add_leg(leg1)
    s.add_leg(leg2)
    s.add_leg(leg3)
    s.add_leg(leg4)
    return s

def create_long_straddle(strike, call_premium, put_premium, quantity=1):
    s = Strategy(name="Long Straddle")
    long_call = OptionLeg(is_call=True, is_long=True, strike=strike, premium=call_premium, quantity=quantity)
    long_put = OptionLeg(is_call=False, is_long=True, strike=strike, premium=put_premium, quantity=quantity)
    s.add_leg(long_call)
    s.add_leg(long_put)
    return s

def create_short_straddle(strike, call_premium, put_premium, quantity=1):
    s = Strategy(name="Short Straddle")
    short_call = OptionLeg(is_call=True, is_long=False, strike=strike, premium=call_premium, quantity=quantity)
    short_put = OptionLeg(is_call=False, is_long=False, strike=strike, premium=put_premium, quantity=quantity)
    s.add_leg(short_call)
    s.add_leg(short_put)
    return s

def create_long_strangle(put_strike, put_premium, call_strike, call_premium, quantity=1):
    s = Strategy(name="Long Strangle")
    long_put_leg = OptionLeg(is_call=False, is_long=True, strike=put_strike, premium=put_premium, quantity=quantity)
    long_call_leg = OptionLeg(is_call=True, is_long=True, strike=call_strike, premium=call_premium, quantity=quantity)
    s.add_leg(long_put_leg)
    s.add_leg(long_call_leg)
    return s

def create_short_strangle(put_strike, put_premium, call_strike, call_premium, quantity=1):
    s = Strategy(name="Short Strangle")
    short_put_leg = OptionLeg(is_call=False, is_long=False, strike=put_strike, premium=put_premium, quantity=quantity)
    short_call_leg = OptionLeg(is_call=True, is_long=False, strike=call_strike, premium=call_premium, quantity=quantity)
    s.add_leg(short_put_leg)
    s.add_leg(short_call_leg)
    return s

def create_ratio_call_spread(long_strike, long_premium, short_strike, short_premium, ratio=2, quantity=1):
    s = Strategy(name="Ratio Call Spread")
    long_call_leg = OptionLeg(is_call=True, is_long=True, strike=long_strike, premium=long_premium, quantity=quantity)
    short_call_leg = OptionLeg(is_call=True, is_long=False, strike=short_strike, premium=short_premium, quantity=ratio * quantity)
    s.add_leg(long_call_leg)
    s.add_leg(short_call_leg)
    return s

def create_ratio_put_spread(long_strike, long_premium, short_strike, short_premium, ratio=2, quantity=1):
    s = Strategy(name="Ratio Put Spread")
    long_put_leg = OptionLeg(is_call=False, is_long=True, strike=long_strike, premium=long_premium, quantity=quantity)
    short_put_leg = OptionLeg(is_call=False, is_long=False, strike=short_strike, premium=short_premium, quantity=ratio * quantity)
    s.add_leg(long_put_leg)
    s.add_leg(short_put_leg)
    return s

def create_call_ratio_backspread(short_strike, short_premium, long_strike, long_premium, ratio=2, quantity=1):
    s = Strategy(name="Call Ratio Backspread")
    short_call = OptionLeg(is_call=True, is_long=False, strike=short_strike, premium=short_premium, quantity=quantity)
    long_call = OptionLeg(is_call=True, is_long=True, strike=long_strike, premium=long_premium, quantity=ratio * quantity)
    s.add_leg(short_call)
    s.add_leg(long_call)
    return s

def create_put_ratio_backspread(lower_strike, lower_premium, higher_strike, higher_premium, ratio=2, quantity=1):
    s = Strategy(name="Put Ratio Backspread")
    long_put = OptionLeg(is_call=False, is_long=True, strike=lower_strike, premium=lower_premium, quantity=ratio * quantity)
    short_put = OptionLeg(is_call=False, is_long=False, strike=higher_strike, premium=higher_premium, quantity=quantity)
    s.add_leg(long_put)
    s.add_leg(short_put)
    return s

def create_long_box(strike_A, call_premium_A, put_premium_A, strike_B, call_premium_B, put_premium_B, quantity=1):
    s = Strategy(name="Long Box")
    long_call_A = OptionLeg(is_call=True, is_long=True, strike=strike_A, premium=call_premium_A, quantity=quantity)
    short_call_B = OptionLeg(is_call=True, is_long=False, strike=strike_B, premium=call_premium_B, quantity=quantity)
    long_put_B = OptionLeg(is_call=False, is_long=True, strike=strike_B, premium=put_premium_B, quantity=quantity)
    short_put_A = OptionLeg(is_call=False, is_long=False, strike=strike_A, premium=put_premium_A, quantity=quantity)
    s.add_leg(long_call_A)
    s.add_leg(short_call_B)
    s.add_leg(long_put_B)
    s.add_leg(short_put_A)
    return s

def create_short_box(strike_A, call_premium_A, put_premium_A, strike_B, call_premium_B, put_premium_B, quantity=1):
    s = Strategy(name="Short Box")
    long_call_B = OptionLeg(is_call=True, is_long=True, strike=strike_B, premium=call_premium_B, quantity=quantity)
    short_call_A = OptionLeg(is_call=True, is_long=False, strike=strike_A, premium=call_premium_A, quantity=quantity)
    long_put_A = OptionLeg(is_call=False, is_long=True, strike=strike_A, premium=put_premium_A, quantity=quantity)
    short_put_B = OptionLeg(is_call=False, is_long=False, strike=strike_B, premium=put_premium_B, quantity=quantity)
    s.add_leg(long_call_B)
    s.add_leg(short_call_A)
    s.add_leg(long_put_A)
    s.add_leg(short_put_B)
    return s

def create_long_instrument_conversion(instrument_price, strike, put_premium, call_premium, quantity=1):
    s = Strategy(name="Long Instrument Conversion")
    underlying_leg = UnderlyingLeg(is_long=True, entry_price=instrument_price, quantity=quantity)
    long_put = OptionLeg(is_call=False, is_long=True, strike=strike, premium=put_premium, quantity=quantity)
    short_call = OptionLeg(is_call=True, is_long=False, strike=strike, premium=call_premium, quantity=quantity)
    s.add_leg(underlying_leg)
    s.add_leg(long_put)
    s.add_leg(short_call)
    return s

def create_short_instrument_conversion(instrument_price, strike, call_premium, put_premium, quantity=1):
    s = Strategy(name="Short Instrument Conversion")
    underlying_leg = UnderlyingLeg(is_long=False, entry_price=instrument_price, quantity=quantity)
    long_call = OptionLeg(is_call=True, is_long=True, strike=strike, premium=call_premium, quantity=quantity)
    short_put = OptionLeg(is_call=False, is_long=False, strike=strike, premium=put_premium, quantity=quantity)
    s.add_leg(underlying_leg)
    s.add_leg(long_call)
    s.add_leg(short_put)
    return s

def create_nothing_strategy(risk_free_rate, T, cash=1):
    s = Strategy(name="Nothing (Treasury Bonds)")
    class BondLeg:
        def __init__(self, payoff):
            self.payoff = payoff
        def payoff_at_expiration(self, underlying_price):
            return self.payoff
    payoff = cash * (np.exp(risk_free_rate * T) - 1)
    s.add_leg(BondLeg(payoff))
    return s


STRATEGY_BUILDERS = {
    "LongCall": create_long_call,
    "ShortCall": create_short_call,
    "BullCallSpread": create_bull_call_spread,
    "LongPut": create_long_put,
    "ShortPut": create_short_put,
    "LongUnderlying": create_long_underlying,
    "ShortUnderlying": create_short_underlying,
    "LongRiskReversal": create_long_risk_reversal,
    "ShortRiskReversal": create_short_risk_reversal,
    "BullPutSpread": create_bull_put_spread,
    "BearCallSpread": create_bear_call_spread,
    "BearPutSpread": create_bear_put_spread,
    "LongButterfly": create_long_butterfly,
    "ShortButterfly": create_short_butterfly,
    "LongIronButterfly": create_long_iron_butterfly,
    "ShortIronButterfly": create_short_iron_butterfly,
    "LongStraddle": create_long_straddle,
    "ShortStraddle": create_short_straddle,
    "LongStrangle": create_long_strangle,
    "ShortStrangle": create_short_strangle,
    "RatioCallSpread": create_ratio_call_spread,
    "RatioPutSpread": create_ratio_put_spread,
    "CallRatioBackspread": create_call_ratio_backspread,
    "PutRatioBackspread": create_put_ratio_backspread,
    "LongBox": create_long_box,
    "ShortBox": create_short_box,
    "LongInstrumentConversion": create_long_instrument_conversion,
    "ShortInstrumentConversion": create_short_instrument_conversion,
    "Nothing": create_nothing_strategy
    # etc. Add more as needed
}
