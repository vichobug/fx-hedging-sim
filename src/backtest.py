"""
backtest.py

Historical backtest: what would each strategy actually have delivered if the
company had put the same hedge on at the start of every month since 2011?

For each start date t, everything is built from information available AT t:
  - spot S_0 on t, and the realized spot 126 trading days later
  - forwards from the interest rates known on t (see load_rate_history)
  - option premiums at the trailing 1-year realized vol up to t
The same usd_received() used by the simulation then scores each strategy on
the ONE path that actually happened.

This is the complement to the Monte Carlo: the simulation says what the
distribution of outcomes should look like; the backtest checks it against
the path history actually took, crises included.

Caveat: monthly starts with a 6-month horizon means consecutive windows
overlap by five months, so ~180 windows carry only ~30 independent
observations. Treat tail statistics accordingly.
"""

import numpy as np
import pandas as pd

import config
from fx_data import load_rate_history, load_spot_history, log_returns
from strategies import Market, usd_received

VOL_WINDOW = 252


def start_dates(spot: pd.DataFrame) -> list:
    """First trading day of each month with a full vol window behind it and a full horizon ahead."""
    idx = spot.index
    first_of_month = idx.to_series().groupby([idx.year, idx.month]).first()
    positions = [idx.get_loc(d) for d in first_of_month]
    return [p for p in positions if p >= VOL_WINDOW and p + config.HORIZON_DAYS < len(idx)]


def run_backtest(strategies=None) -> pd.DataFrame:
    """One row per start date; per-strategy P&L as a fraction of budget, plus context columns."""
    strategies = strategies or config.STRATEGIES
    ccys = list(config.EXPOSURES)
    # Pull extra history before HISTORY_START so the first window has a trailing vol
    spot = load_spot_history(ccys, start="2009-01-01")
    returns = log_returns(spot)
    trailing_vol = returns.rolling(VOL_WINDOW).std() * np.sqrt(252)
    rates = np.log1p(load_rate_history(spot.index))   # continuous compounding
    notional = np.array([config.EXPOSURES[c] for c in ccys], float)

    rows = []
    for p in start_dates(spot):
        t, t_end = spot.index[p], spot.index[p + config.HORIZON_DAYS]
        market = Market(
            currencies=ccys,
            notional=notional,
            spot=spot.iloc[p].to_numpy(),
            r_usd=float(rates["USD"].iloc[p]),
            r_foreign=rates[ccys].iloc[p].to_numpy(),
            vol=trailing_vol.iloc[p].to_numpy(),
            T=config.HORIZON_YEARS,
        )
        spot_ratio = (spot.iloc[p + config.HORIZON_DAYS] / spot.iloc[p]).to_numpy()[None, :]
        budget = market.budget_usd.sum()

        row = {"start": t, "end": t_end, "budget_usd": budget}
        for name, params in strategies.items():
            row[name] = usd_received(market, spot_ratio, **params).sum() / budget - 1
        for i, ccy in enumerate(ccys):
            row[f"move_{ccy}"] = spot_ratio[0, i] - 1
        rows.append(row)

    return pd.DataFrame(rows).set_index("start")


def summarize_backtest(bt: pd.DataFrame, strategies=None) -> pd.DataFrame:
    strategies = strategies or config.STRATEGIES
    out = {}
    for name in strategies:
        r = bt[name]
        worst_5pct = r[r <= r.quantile(0.05)]
        out[name] = {
            "mean": r.mean(),
            "std": r.std(),
            "CVaR": -worst_5pct.mean(),
            "worst": r.min(),
            "worst_start": r.idxmin().date(),
            "best": r.max(),
            "miss_2pct": (r < -0.02).mean(),
        }
    return pd.DataFrame(out).T
