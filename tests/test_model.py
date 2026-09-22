"""Checks against known closed-form results. Run with: pytest"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from pricing import forward_rate, garman_kohlhagen
from risk import cvar, var
from simulate import simulate_bootstrap, simulate_gbm
from strategies import Market, usd_received

S, RD, RF, T, VOL = 1.15, 0.04, 0.02, 0.5, 0.08


def test_forward_follows_interest_rate_parity():
    # Higher USD rate -> EUR trades at a forward premium, and vice versa
    assert forward_rate(S, RD, RF, T) > S
    assert forward_rate(S, RF, RD, T) < S
    assert forward_rate(S, RD, RD, T) == pytest.approx(S)


@pytest.mark.parametrize("K", [1.05, 1.15, 1.25])
def test_put_call_parity(K):
    call = garman_kohlhagen(S, K, T, RD, RF, VOL, "call")
    put = garman_kohlhagen(S, K, T, RD, RF, VOL, "put")
    assert call - put == pytest.approx(S * np.exp(-RF * T) - K * np.exp(-RD * T))


def test_put_matches_monte_carlo():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(1_000_000)
    S_T = S * np.exp((RD - RF - 0.5 * VOL ** 2) * T + VOL * np.sqrt(T) * z)
    mc = np.exp(-RD * T) * np.maximum(1.15 - S_T, 0).mean()
    assert garman_kohlhagen(S, 1.15, T, RD, RF, VOL, "put") == pytest.approx(mc, rel=0.01)


def test_var_cvar_on_uniform_losses():
    pnl = -np.arange(1, 101, dtype=float)       # losses of 1..100
    assert var(pnl) == pytest.approx(95.05)
    assert cvar(pnl) == pytest.approx(98.0)     # mean of worst 5: 96..100
    assert cvar(pnl) >= var(pnl)


def _market():
    return Market(currencies=["EUR"], notional=np.array([1e6]), spot=np.array([S]),
                  r_usd=RD, r_foreign=np.array([RF]), vol=np.array([VOL]), T=T)


def test_full_forward_hedge_is_riskless():
    ratio = np.array([[0.8], [1.0], [1.3]])
    usd = usd_received(_market(), ratio, forward_ratio=1.0)
    assert np.allclose(usd, 1e6 * forward_rate(S, RD, RF, T))


def test_put_hedge_floors_the_downside_and_keeps_upside():
    m = _market()
    strike, premium = m.put_premium(1.0)
    floor = 1e6 * (strike[0] - premium[0] * np.exp(RD * T))
    usd = usd_received(m, np.array([[0.5], [2.0]]), option_ratio=1.0)
    assert usd[0, 0] == pytest.approx(floor)                       # crash -> floored
    assert usd[1, 0] == pytest.approx(1e6 * (2 * S - premium[0] * np.exp(RD * T)))  # rally -> kept


def test_hedge_ratios_must_be_valid():
    with pytest.raises(ValueError):
        usd_received(_market(), np.ones((1, 1)), forward_ratio=0.7, option_ratio=0.5)


@pytest.mark.parametrize("engine", [simulate_gbm, simulate_bootstrap])
def test_engines_are_centered_and_preserve_correlation(engine):
    rng = np.random.default_rng(1)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]]) * 0.006 ** 2
    returns = pd.DataFrame(rng.multivariate_normal([0.0003, -0.0002], cov, size=5000))
    ratio = engine(returns, 40_000, 126, np.random.default_rng(2))
    assert ratio.shape == (40_000, 2)
    assert np.allclose(ratio.mean(axis=0), 1.0, atol=0.003)       # no drift: E[S_T/S_0] = 1
    assert np.corrcoef(np.log(ratio).T)[0, 1] == pytest.approx(0.6, abs=0.05)


def test_rate_history_has_no_lookahead():
    from fx_data import load_rate_history
    # The Jan 2024 EUR monthly average must not be visible until Feb 1, 2024
    dates = pd.DatetimeIndex(["2024-01-15", "2024-01-31", "2024-02-01"])
    hist = load_rate_history(dates)
    assert hist.loc["2024-01-15", "EUR"] == hist.loc["2024-01-31", "EUR"]   # still Dec 2023's value
    assert hist.loc["2024-02-01", "EUR"] != hist.loc["2024-01-31", "EUR"]   # Jan value now known
