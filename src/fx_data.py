"""
fx_data.py

Loads the FRED CSVs written by scripts/fetch_fx_data.py and puts every
currency on the same footing: USD per 1 unit of foreign currency. That's the
natural quote for a USD company converting foreign receivables -- a HIGHER
number is good news (each foreign unit buys more dollars).
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, HISTORY_START

# currency -> (FRED series id, how FRED quotes it)
FX_SERIES = {
    "EUR": ("DEXUSEU", "USD_per_FCY"),
    "GBP": ("DEXUSUK", "USD_per_FCY"),
    "JPY": ("DEXJPUS", "FCY_per_USD"),
    "CNY": ("DEXCHUS", "FCY_per_USD"),
    "MXN": ("DEXMXUS", "FCY_per_USD"),
}

# 3-month rates, % per annum. USD uses the 3-month T-bill (daily); the others
# use the OECD 3-month interbank series (monthly, and some lag by months).
# 3m rates stand in for the 6m tenor -- see README limitations.
RATE_SERIES = {
    "USD": "DTB3",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "CNY": "IR3TIB01CNM156N",
    "MXN": "IR3TIB01MXM156N",
}


def _read_fred(series_id: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / f"{series_id}.csv", parse_dates=["observation_date"])
    s = pd.to_numeric(df[series_id], errors="coerce")  # FRED marks holidays with "."
    s.index = df["observation_date"]
    return s.dropna()


def load_spot_history(currencies=None, start=HISTORY_START) -> pd.DataFrame:
    """Daily spot history, USD per 1 FCY, on dates where every currency traded."""
    currencies = currencies or list(FX_SERIES)
    cols = {}
    for ccy in currencies:
        series_id, quote = FX_SERIES[ccy]
        s = _read_fred(series_id)
        cols[ccy] = s if quote == "USD_per_FCY" else 1.0 / s
    spot = pd.DataFrame(cols).dropna()
    return spot.loc[spot.index >= start]


def log_returns(spot: pd.DataFrame) -> pd.DataFrame:
    return np.log(spot).diff().dropna()


def load_rates() -> pd.DataFrame:
    """Latest available rate per currency, as a decimal, with its as-of date."""
    rows = []
    for ccy, series_id in RATE_SERIES.items():
        s = _read_fred(series_id)
        rows.append({"currency": ccy, "rate": s.iloc[-1] / 100.0, "as_of": s.index[-1].date()})
    return pd.DataFrame(rows).set_index("currency")


if __name__ == "__main__":
    spot = load_spot_history()
    print(f"Spot history: {spot.index.min().date()} to {spot.index.max().date()}, {len(spot)} days")
    print(spot.tail(3).round(5))
    rets = log_returns(spot)
    print("\nAnnualized vol (full sample):")
    print((rets.std() * np.sqrt(252)).round(4))
    print("\nCorrelation of daily returns:")
    print(rets.corr().round(2))
    print("\nRates:")
    print(load_rates())
