"""
run_simulation.py

Main run: prints today's market (spot, forwards, option premiums), then
compares every configured hedging strategy on 10,000 simulated six-month
outcomes, under both simulation engines, and saves the headline charts.

    python notebooks/run_simulation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

import config
from scenario import build_market, evaluate, simulate

M = 1e6


def print_market(market, as_of, rates):
    print(f"Market as of {as_of}  |  USD rate {rates.loc['USD', 'rate']:.2%}  |  horizon {config.HORIZON_DAYS} trading days\n")
    print(f"{'Ccy':<5}{'Notional':>16}{'Spot':>10}{'Forward':>10}{'Fwd pts':>9}{'Rate':>7}"
          f"{'Vol':>7}{'ATM put':>9}{'Budget $M':>11}")
    _, premium = market.put_premium(1.0)
    for i, ccy in enumerate(market.currencies):
        fwd_pts = market.forward[i] / market.spot[i] - 1
        print(f"{ccy:<5}{market.notional[i]:>16,.0f}{market.spot[i]:>10.5f}{market.forward[i]:>10.5f}"
              f"{fwd_pts:>+9.2%}{rates.loc[ccy, 'rate']:>7.2%}{market.vol[i]:>7.1%}"
              f"{premium[i] / market.spot[i]:>9.2%}{market.budget_usd[i] / M:>11.1f}")
    print(f"{'':<5}{'':>16}{'':>10}{'':>10}{'':>9}{'':>7}{'':>7}{'Total':>9}{market.budget_usd.sum() / M:>11.1f}")
    stale = rates[rates["as_of"] < as_of.replace(day=1)]
    if len(stale):
        print(f"\nNote: latest FRED rate for {', '.join(stale.index)} predates {as_of:%b %Y} (OECD series lag).")


def print_summary(title, summary):
    print(f"\n{title}  (P&L vs. budget, $M)")
    print("-" * 86)
    print(f"{'Strategy':<22}{'Mean':>8}{'Std':>8}{'VaR95':>9}{'CVaR95':>9}{'Upside95':>10}{'Worst':>9}{'P(miss>2%)':>12}")
    for name, row in summary.iterrows():
        print(f"{name:<22}{row['mean'] / M:>8.2f}{row['std'] / M:>8.2f}{row['VaR'] / M:>9.2f}"
              f"{row['CVaR'] / M:>9.2f}{row['p95_upside'] / M:>10.2f}{row['worst'] / M:>9.2f}"
              f"{row['prob_miss_2pct']:>12.1%}")


def plot_distributions(pnl, summary, path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    lo = min(np.percentile(p, 0.2) for p in pnl.values()) / M
    hi = max(np.percentile(p, 99.8) for p in pnl.values()) / M
    bins = np.linspace(lo, hi, 90)
    for name, p in pnl.items():
        ax.hist(p / M, bins=bins, histtype="step", linewidth=1.6, label=f"{name} (CVaR {summary.loc[name, 'CVaR'] / M:.1f})")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("P&L vs. budget at horizon ($M)")
    ax.set_ylabel("Simulated paths")
    ax.set_title("Six-month FX P&L by hedging strategy (block bootstrap, 10,000 paths)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_vs_cost(summary, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, row in summary.iterrows():
        ax.scatter(row["CVaR"] / M, row["mean"] / M, s=60)
        ax.annotate(name, (row["CVaR"] / M, row["mean"] / M), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Tail risk: CVaR 95% ($M, lower is better)")
    ax.set_ylabel("Expected P&L vs. budget ($M)")
    ax.set_title("What each strategy costs vs. how much tail risk it removes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    market, returns, as_of, rates = build_market()
    print_market(market, as_of, rates)

    results = {}
    for engine, label in [("gbm", "Correlated GBM (normal returns)"), ("bootstrap", "Block bootstrap (real historical returns)")]:
        pnl, summary = evaluate(market, simulate(market, returns, engine))
        results[engine] = (pnl, summary)
        print_summary(label, summary)

    pnl, summary = results["bootstrap"]
    unhedged_cvar = summary.loc["Unhedged", "CVaR"]
    print(f"\nVs. unhedged (bootstrap; unhedged CVaR95 = {unhedged_cvar / M:.2f}M, centered on {config.CENTER}):")
    for name, row in summary.drop("Unhedged").iterrows():
        print(f"  {name:<22} tail risk removed {(unhedged_cvar - row['CVaR']) / M:>6.2f}M   "
              f"expected P&L {row['mean'] / M:+.2f}M   upside95 {row['p95_upside'] / M:+.2f}M")

    config.FIGURES_DIR.mkdir(exist_ok=True)
    plot_distributions(pnl, summary, config.FIGURES_DIR / "pnl_distributions.png")
    plot_risk_vs_cost(summary, config.FIGURES_DIR / "risk_vs_cost.png")
    print(f"\nSaved figures to {config.FIGURES_DIR.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
