"""
config.py

The hypothetical company and the simulation settings, in one place.

Scenario: a US-based multinational (USD reporting currency) expects to
receive foreign-currency revenue from five regional subsidiaries in six
months' time. The CFO's budget translates that revenue at TODAY's spot
rates -- so any move in FX between now and then is a gain or loss vs.
budget. The question: how much of that should be hedged, and with what?
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

# Foreign-currency receivables due at the horizon (units of each currency).
# Sized so each is roughly $25-60M at current spot (~$175M total).
EXPOSURES = {
    "EUR": 50_000_000,
    "GBP": 25_000_000,
    "JPY": 4_000_000_000,
    "CNY": 200_000_000,
    "MXN": 500_000_000,
}

HORIZON_DAYS = 126               # ~6 months of trading days
HORIZON_YEARS = HORIZON_DAYS / 252

HISTORY_START = "2010-01-01"     # post-GFC history used for calibration

# Where simulated spot is centered at the horizon:
#   "spot"    -- random walk, E[S_T] = today's spot. Empirically the better
#                forecaster (the "forward premium puzzle"), so forward points
#                show up as a real expected gain/cost of hedging.
#   "forward" -- E[S_T] = the forward rate, i.e. the forward is an unbiased
#                forecast. Hedging then costs nothing on average and the
#                comparison is purely about risk.
CENTER = "spot"

N_PATHS = 10_000
SEED = 42
BLOCK_LENGTH = 20                # ~1 trading month per bootstrap block

# Each strategy hedges a fraction of every exposure with forwards and a
# fraction with purchased puts (strike = moneyness x forward rate); the rest
# is left open. Ratios are applied uniformly across currencies.
STRATEGIES = {
    "Unhedged":            dict(forward_ratio=0.0, option_ratio=0.0),
    "100% forwards":       dict(forward_ratio=1.0, option_ratio=0.0),
    "100% ATM puts":       dict(forward_ratio=0.0, option_ratio=1.0, moneyness=1.00),
    "100% 95% OTM puts":   dict(forward_ratio=0.0, option_ratio=1.0, moneyness=0.95),
    "Layered 50/30/20":    dict(forward_ratio=0.5, option_ratio=0.3, moneyness=0.97),
}
