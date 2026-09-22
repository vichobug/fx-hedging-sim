"""
strategies.py

Turns simulated FX outcomes into USD received under a hedging strategy.

For each currency with notional N (foreign units), spot S0, forward F and
simulated horizon spot S_T, a strategy splits N into three slices:

  forward_ratio  -> sold forward today at F: receives N * F no matter what
  option_ratio   -> left open, but protected by a purchased put struck at
                    K = moneyness * F: receives N * max(S_T, K), minus the
                    premium (paid today, carried forward to the horizon at
                    the USD rate so every strategy is compared in
                    horizon dollars)
  the rest       -> unhedged: receives N * S_T

The forward slice removes both the downside AND the upside. The option
slice removes only the downside, and the premium is what that asymmetry
costs.
"""

from dataclasses import dataclass

import numpy as np

from pricing import forward_rate, garman_kohlhagen


@dataclass
class Market:
    """Everything known today, one entry per currency (arrays aligned by column)."""
    currencies: list
    notional: np.ndarray    # foreign units
    spot: np.ndarray        # USD per FCY
    r_usd: float
    r_foreign: np.ndarray
    vol: np.ndarray         # annualized, used for option pricing
    T: float

    @property
    def forward(self):
        return forward_rate(self.spot, self.r_usd, self.r_foreign, self.T)

    @property
    def budget_usd(self):
        """What the budget assumed: every receivable converted at today's spot."""
        return self.notional * self.spot

    def put_premium(self, moneyness):
        strike = moneyness * self.forward
        premium = garman_kohlhagen(self.spot, strike, self.T, self.r_usd, self.r_foreign, self.vol, "put")
        return strike, premium


def usd_received(market: Market, spot_ratio: np.ndarray, forward_ratio=0.0,
                 option_ratio=0.0, moneyness=1.0) -> np.ndarray:
    """USD received at the horizon, per path and currency: shape (n_paths, n_currencies)."""
    if forward_ratio < 0 or option_ratio < 0 or forward_ratio + option_ratio > 1 + 1e-9:
        raise ValueError("hedge ratios must be non-negative and sum to at most 1")

    N = market.notional
    S_T = market.spot * spot_ratio
    open_ratio = 1.0 - forward_ratio - option_ratio

    usd = forward_ratio * N * market.forward + open_ratio * N * S_T
    if option_ratio > 0:
        strike, premium = market.put_premium(moneyness)
        premium_at_horizon = premium * np.exp(market.r_usd * market.T)
        usd = usd + option_ratio * N * (np.maximum(S_T, strike) - premium_at_horizon)
    return usd
