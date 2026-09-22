"""
sensitivity.py

Four checks on what actually drives the hedging decision:

1. Hedge-ratio frontier -- sweep the forward / option mix from 0% to 100%
   and trace tail risk (CVaR) against expected P&L and upside kept.
2. Strike choice -- how option cost and protection trade off as the put
   strike moves further out of the money.
3. Diversification -- sum of each currency's standalone CVaR vs. the
   portfolio's CVaR. The gap is risk that correlation already cancels, so
   hedging every currency to its standalone risk over-hedges.
4. Model risk -- GBM vs. block bootstrap in the far tail (99% CVaR, worst
   paths), to see whether normal returns understate crash risk here.

    python notebooks/sensitivity.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

import config
from risk import cvar
from scenario import build_market, simulate
from strategies import usd_received

M = 1e6
RATIOS = np.linspace(0, 1, 11)


def portfolio_pnl(market, spot_ratio, **params):
    return usd_received(market, spot_ratio, **params).sum(axis=1) - market.budget_usd.sum()


def hedge_ratio_frontier(market, spot_ratio):
    print("1. Hedge-ratio frontier (bootstrap)")
    print("-" * 78)
    print(f"{'Hedge %':>8} | {'Forwards: CVaR95':>17}{'Mean':>8}{'Upside95':>10} | "
          f"{'ATM puts: CVaR95':>17}{'Mean':>8}{'Upside95':>10}")
    curves = {"Forwards": [], "ATM-forward puts": []}
    for h in RATIOS:
        row = []
        for label, params in [("Forwards", dict(forward_ratio=h)), ("ATM-forward puts", dict(option_ratio=h))]:
            pnl = portfolio_pnl(market, spot_ratio, **params)
            stats = (cvar(pnl) / M, pnl.mean() / M, np.percentile(pnl, 95) / M)
            curves[label].append(stats)
            row.append(stats)
        (fc, fm, fu), (oc, om, ou) = row
        print(f"{h:>8.0%} | {fc:>17.2f}{fm:>8.2f}{fu:>10.2f} | {oc:>17.2f}{om:>8.2f}{ou:>10.2f}")
    print()
    return {k: np.array(v) for k, v in curves.items()}


def strike_sensitivity(market, spot_ratio):
    print("2. Put strike choice (100% hedged with puts, bootstrap)")
    print("-" * 78)
    print(f"{'Strike (x fwd)':>15}{'Premium $M':>12}{'CVaR95':>10}{'Worst':>9}{'Upside95':>10}{'Mean':>8}")
    for m in [1.00, 0.98, 0.96, 0.94, 0.92, 0.90]:
        _, premium = market.put_premium(m)
        pnl = portfolio_pnl(market, spot_ratio, option_ratio=1.0, moneyness=m)
        print(f"{m:>15.2f}{(premium * market.notional).sum() / M:>12.2f}{cvar(pnl) / M:>10.2f}"
              f"{pnl.min() / M:>9.2f}{np.percentile(pnl, 95) / M:>10.2f}{pnl.mean() / M:>8.2f}")
    print("Cheaper, further-OTM puts only pay out in large moves: you self-insure the first few %.\n")


def diversification(market, spot_ratio):
    print("3. Diversification (unhedged, bootstrap)")
    print("-" * 78)
    usd = usd_received(market, spot_ratio)
    standalone = [cvar(usd[:, i] - market.budget_usd[i]) for i in range(len(market.currencies))]
    for ccy, c in zip(market.currencies, standalone):
        print(f"  {ccy} standalone CVaR95: {c / M:>6.2f}M")
    portfolio = cvar(usd.sum(axis=1) - market.budget_usd.sum())
    total = sum(standalone)
    print(f"  Sum of standalone:      {total / M:>6.2f}M")
    print(f"  Portfolio CVaR95:       {portfolio / M:>6.2f}M  "
          f"({1 - portfolio / total:.0%} of standalone risk cancels out across currencies)\n")


def model_risk(market, returns):
    print("4. Model risk: normal (GBM) vs. historical (bootstrap) tails, unhedged")
    print("-" * 78)
    print(f"{'Engine':<12}{'Std':>8}{'CVaR95':>9}{'CVaR99':>9}{'Worst':>9}")
    for engine in ["gbm", "bootstrap"]:
        pnl = portfolio_pnl(market, simulate(market, returns, engine, n_paths=50_000))
        print(f"{engine:<12}{pnl.std() / M:>8.2f}{cvar(pnl, 0.95) / M:>9.2f}{cvar(pnl, 0.99) / M:>9.2f}{pnl.min() / M:>9.2f}")
    print("Over a 6-month horizon, 126 daily returns sum together and the fat tails of")
    print("individual days largely average out -- so the two engines should be close.\n")


def plot_frontier(curves, unhedged_cvar, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, arr in curves.items():
        ax.plot(arr[:, 0], arr[:, 2], marker="o", label=f"{label} (upside kept)")
        for h, (c, _, u) in zip(RATIOS[::2], arr[::2]):
            ax.annotate(f"{h:.0%}", (c, u), xytext=(5, -10), textcoords="offset points", fontsize=8)
    ax.axvline(unhedged_cvar, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Tail risk: CVaR 95% ($M)")
    ax.set_ylabel("Upside kept: 95th pct P&L ($M)")
    ax.set_title("Hedge-ratio frontier: forwards give up upside, puts keep it")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    market, returns, _, _ = build_market()
    spot_ratio = simulate(market, returns, "bootstrap")

    curves = hedge_ratio_frontier(market, spot_ratio)
    strike_sensitivity(market, spot_ratio)
    diversification(market, spot_ratio)
    model_risk(market, returns)

    config.FIGURES_DIR.mkdir(exist_ok=True)
    plot_frontier(curves, curves["Forwards"][0, 0], config.FIGURES_DIR / "hedge_frontier.png")
    print(f"Saved {(config.FIGURES_DIR / 'hedge_frontier.png').relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
