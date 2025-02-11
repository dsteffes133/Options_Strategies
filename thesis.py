# thesis.py
import math
import numpy as np

class Scenario:
    """
    Encapsulates a single scenario's assumptions:
      - probability: weight (0 to 1)
      - direction_label: user-friendly descriptor, e.g., "bullish", "bearish", "neutral"
      - price_move_pct: expected final move % relative to current price
      - vol_adjust_pct: expected relative shift in volatility (e.g., +0.20 => vol up 20%)
      - scenario_label: optional scenario name or short note
    """
    def __init__(self, probability, price_move_pct=0.0, vol_adjust_pct=0.0,
                 direction_label="neutral", scenario_label=""):
        self.probability = probability  # float 0..1
        self.price_move_pct = price_move_pct  # e.g. +0.10 => +10%
        self.vol_adjust_pct = vol_adjust_pct  # e.g. +0.20 => +20% to implied vol
        self.direction_label = direction_label
        self.scenario_label = scenario_label

    def __repr__(self):
        return (f"Scenario(prob={self.probability}, dir={self.direction_label}, "
                f"move={self.price_move_pct:.2f}, vol_adj={self.vol_adjust_pct:.2f})")

class Thesis:
    """
    Collects multiple scenarios (each with probability and different drift/vol outlook).
    Used by the simulation module to generate terminal or path-based prices.
    """
    def __init__(self, scenarios=None):
        if scenarios is None:
            scenarios = []
        self.scenarios = scenarios
        self.validate_scenarios()

    def validate_scenarios(self):
        """
        Check that the sum of scenario probabilities is <= 1 (or ~1).
        In many models, you want them to sum to exactly 1, but
        you might allow partial or 'residual' probability.
        """
        total_prob = sum(s.probability for s in self.scenarios)
        if abs(total_prob - 1.0) > 1e-6:
            # We might raise a warning or just continue.
            # For example, if user wants to indicate multiple 'branches' that total < or > 1.
            pass

    def add_scenario(self, scenario):
        self.scenarios.append(scenario)

    def total_probability(self):
        return sum(s.probability for s in self.scenarios)
