"""
simulate.py

Two ways to generate joint, correlated FX outcomes at the horizon. Both
return an (n_paths, n_currencies) array of S_T / S_0 -- the multiplier on
today's spot for each currency on each simulated path.

1. Correlated GBM (the textbook model): terminal log-returns drawn from a
   multivariate normal with the historical daily covariance scaled to the
   horizon. Captures volatility and correlation, but not fat tails -- a
   normal distribution puts almost no weight on 2015 SNB- or 2016 Brexit-
   sized days.

2. Block bootstrap (the historical model): each path is stitched together
   from randomly chosen ~1-month blocks of REAL consecutive historical days,
   with all five currencies taken from the same dates. That keeps the fat
   tails, the crisis-time correlation spikes, and volatility clustering the
   real data had, without assuming any distribution.

Drift: both models are centered so E[S_T] = S_0 -- no directional view on
any currency. The historical sample's trend (e.g. yen's long slide since
2012) is noise for a 6-month hedging decision, and baking it in would make
whichever strategy happens to match that trend look artificially good.
"""

import numpy as np
import pandas as pd


def _centered(returns: pd.DataFrame) -> np.ndarray:
    """Daily log-returns shifted to mean -sigma^2/2, so E[S_T / S_0] = 1."""
    r = returns.to_numpy()
    return r - r.mean(axis=0) - 0.5 * r.var(axis=0)


def simulate_gbm(returns: pd.DataFrame, n_paths: int, n_days: int, rng) -> np.ndarray:
    cov = np.cov(returns.to_numpy(), rowvar=False) * n_days
    mean = -0.5 * np.diag(cov)
    terminal_log = rng.multivariate_normal(mean, cov, size=n_paths)
    return np.exp(terminal_log)


def simulate_bootstrap(returns: pd.DataFrame, n_paths: int, n_days: int, rng,
                       block_length: int = 20) -> np.ndarray:
    r = _centered(returns)
    n_hist = len(r)
    n_blocks = int(np.ceil(n_days / block_length))

    starts = rng.integers(0, n_hist - block_length, size=(n_paths, n_blocks))
    day_idx = (starts[:, :, None] + np.arange(block_length)).reshape(n_paths, -1)[:, :n_days]

    terminal_log = r[day_idx].sum(axis=1)   # (n_paths, n_currencies)
    return np.exp(terminal_log)


ENGINES = {"gbm": simulate_gbm, "bootstrap": simulate_bootstrap}
