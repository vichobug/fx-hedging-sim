"""
scenario.py

Shared entry point: builds today's market from the FRED data, runs a
simulation engine, and evaluates every configured strategy. The notebook
scripts all go through run_scenario() so they're guaranteed to be looking
at the same market and the same simulated paths.
"""

import numpy as np
import pandas as pd

import config
from fx_data import load_spot_history, load_rates, log_returns
from risk import summarize
from simulate import ENGINES
from strategies import Market, usd_received


def build_market():
    ccys = list(config.EXPOSURES)
    spot_hist = load_spot_history(ccys)
    returns = log_returns(spot_hist)
    rates = load_rates()

    # Simple annual rates -> continuous compounding for the pricing formulas
    cont = np.log1p(rates["rate"])
    # Options are priced at the same historical vol the simulation is
    # calibrated to. Pricing at a different vol (e.g. a calmer trailing year)
    # would make options look systematically cheap or rich vs. the paths
    # they're evaluated on -- a model artifact, not a hedging insight.
    vol = returns.std().to_numpy() * np.sqrt(252)

    market = Market(
        currencies=ccys,
        notional=np.array([config.EXPOSURES[c] for c in ccys], float),
        spot=spot_hist.iloc[-1].to_numpy(),
        r_usd=float(cont["USD"]),
        r_foreign=cont[ccys].to_numpy(),
        vol=vol,
        T=config.HORIZON_YEARS,
    )
    return market, returns, spot_hist.index[-1].date(), rates


def simulate(market, returns, engine="bootstrap", n_paths=config.N_PATHS, seed=config.SEED,
             center=config.CENTER):
    """(n_paths, n_currencies) array of S_T / S_0 -- see config.CENTER for centering."""
    rng = np.random.default_rng(seed)
    kwargs = {"block_length": config.BLOCK_LENGTH} if engine == "bootstrap" else {}
    spot_ratio = ENGINES[engine](returns, n_paths, config.HORIZON_DAYS, rng, **kwargs)
    if center == "forward":
        spot_ratio = spot_ratio * (market.forward / market.spot)
    elif center != "spot":
        raise ValueError(f"center must be 'spot' or 'forward', got {center!r}")
    return spot_ratio


def evaluate(market, spot_ratio, strategies=None):
    """Portfolio P&L vs. budget per strategy -> (pnl dict, summary DataFrame)."""
    strategies = strategies or config.STRATEGIES
    budget = market.budget_usd.sum()
    pnl, rows = {}, {}
    for name, params in strategies.items():
        usd = usd_received(market, spot_ratio, **params).sum(axis=1)
        pnl[name] = usd - budget
        rows[name] = summarize(pnl[name], budget)
    return pnl, pd.DataFrame(rows).T


def run_scenario(engine="bootstrap", **kwargs):
    market, returns, as_of, rates = build_market()
    spot_ratio = simulate(market, returns, engine, **kwargs)
    pnl, summary = evaluate(market, spot_ratio)
    return market, spot_ratio, pnl, summary
