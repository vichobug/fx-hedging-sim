"""
risk.py

Risk metrics on a distribution of P&L vs. budget (positive = beat budget).

VaR 95%:  the loss exceeded on only 5% of paths.
CVaR 95%: the AVERAGE loss across those worst 5% of paths (a.k.a. expected
          shortfall). Preferred over VaR because it looks at how bad the bad
          tail actually gets, not just where it starts -- two strategies can
          share a VaR while one has a far nastier tail beyond it.
"""

import numpy as np


def var(pnl, level=0.95):
    return -np.percentile(pnl, 100 * (1 - level))


def cvar(pnl, level=0.95):
    cutoff = np.percentile(pnl, 100 * (1 - level))
    return -pnl[pnl <= cutoff].mean()


def summarize(pnl, budget, level=0.95) -> dict:
    """Headline stats for one strategy's P&L vs. budget, in USD."""
    return {
        "mean": pnl.mean(),
        "std": pnl.std(),
        "VaR": var(pnl, level),
        "CVaR": cvar(pnl, level),
        "p95_upside": np.percentile(pnl, 95),
        "worst": pnl.min(),
        "prob_miss_2pct": np.mean(pnl < -0.02 * budget),
    }
