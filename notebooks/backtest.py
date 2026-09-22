"""
backtest.py (notebook)

Runs the historical backtest, compares it to what the Monte Carlo predicted,
walks through the worst historical windows, and plots the P&L timeline.

    python notebooks/backtest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

import config
from backtest import run_backtest, summarize_backtest
from scenario import build_market, evaluate, simulate

SHOWN = ["Unhedged", "100% forwards", "100% ATM puts", "Layered 50/30/20"]


def print_summary(bt, summary):
    print(f"Historical backtest: {len(bt)} monthly start dates, "
          f"{bt.index.min():%b %Y} to {bt.index.max():%b %Y} (6-month horizon each)")
    print(f"Windows overlap by 5 months -> roughly {len(bt) // 6} independent periods.\n")
    print("P&L vs. budget, % of budget")
    print("-" * 90)
    print(f"{'Strategy':<22}{'Mean':>8}{'Std':>8}{'CVaR95':>9}{'Worst':>8}{'  (window start)':<17}{'Best':>8}{'Miss>2%':>10}")
    for name, r in summary.iterrows():
        print(f"{name:<22}{r['mean']:>8.2%}{r['std']:>8.2%}{r['CVaR']:>9.2%}{r['worst']:>8.2%}"
              f"{'  (' + str(r['worst_start']) + ')':<17}{r['best']:>8.2%}{r['miss_2pct']:>10.1%}")


def compare_to_simulation(summary):
    market, returns, _, _ = build_market()
    budget = market.budget_usd.sum()
    _, sim = evaluate(market, simulate(market, returns, "bootstrap"))
    print("\nSimulation (bootstrap, today's market) vs. history, % of budget")
    print("-" * 70)
    print(f"{'Strategy':<22}{'Sim std':>9}{'Hist std':>10}{'Sim CVaR95':>12}{'Hist CVaR95':>13}")
    for name in config.STRATEGIES:
        print(f"{name:<22}{sim.loc[name, 'std'] / budget:>9.2%}{summary.loc[name, 'std']:>10.2%}"
              f"{sim.loc[name, 'CVaR'] / budget:>12.2%}{summary.loc[name, 'CVaR']:>13.2%}")
    print("Differences come from today's rates and forward points vs. each historical date's,")
    print("and from history being a single path with only ~30 independent windows.")


def worst_windows(bt, n=6):
    print(f"\nWorst {n} unhedged windows (non-overlapping) and what each hedge delivered")
    print("-" * 90)
    ccys = list(config.EXPOSURES)
    print(f"{'Window':<24}" + "".join(f"{c:>7}" for c in ccys) + "".join(f"{s.split()[-1] if s != 'Unhedged' else 'Unhedg':>10}" for s in SHOWN))
    picked = []
    for start in bt["Unhedged"].sort_values().index:
        if all(abs((start - p).days) > 182 for p in picked):
            picked.append(start)
        if len(picked) == n:
            break
    for start in picked:
        r = bt.loc[start]
        window = f"{start:%Y-%m} -> {r['end']:%Y-%m}"
        print(f"{window:<24}" + "".join(f"{r[f'move_{c}']:>7.1%}" for c in ccys)
              + "".join(f"{r[s]:>10.2%}" for s in SHOWN))
    print("(currency columns: move in USD value of each currency over the window)")


def plot_timeline(bt, path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in SHOWN:
        ax.plot(bt.index, bt[name] * 100, linewidth=1.5 if name == "Unhedged" else 1.3, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    for label, date in [("Eurozone crisis", "2011-08"), ("Dollar rally", "2014-07"), ("CNY deval.", "2015-08"),
                        ("Brexit vote", "2016-06"), ("COVID", "2020-03"), ("Fed hikes / USD peak", "2022-03")]:
        ax.axvline(np.datetime64(date), color="grey", linestyle=":", linewidth=0.8)
        ax.annotate(label, (np.datetime64(date), ax.get_ylim()[1]), rotation=90, va="top", ha="right", fontsize=8, color="grey")
    ax.set_ylabel("6-month P&L vs. budget (% of budget)")
    ax.set_xlabel("Hedge start date")
    ax.set_title("Backtest: realized 6-month outcome of each strategy, started every month since 2010")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    bt = run_backtest()
    summary = summarize_backtest(bt)
    print_summary(bt, summary)
    compare_to_simulation(summary)
    worst_windows(bt)

    config.FIGURES_DIR.mkdir(exist_ok=True)
    plot_timeline(bt, config.FIGURES_DIR / "backtest_timeline.png")
    print(f"\nSaved {(config.FIGURES_DIR / 'backtest_timeline.png').relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
