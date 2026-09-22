"""
fetch_fx_data.py

Downloads daily FX rates and short-term interest rates from FRED (Federal
Reserve Bank of St. Louis) as CSVs into data/. No API key needed -- these
are FRED's public graph-download endpoints.

Run from the repo root:
    python scripts/fetch_fx_data.py
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DATA_DIR
from fx_data import FX_SERIES, RATE_SERIES

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def fetch(series_id: str) -> Path:
    out = DATA_DIR / f"{series_id}.csv"
    with urllib.request.urlopen(FRED_CSV_URL.format(series_id), timeout=60) as resp:
        out.write_bytes(resp.read())
    return out


def main():
    DATA_DIR.mkdir(exist_ok=True)
    series_ids = [s for s, _ in FX_SERIES.values()] + list(RATE_SERIES.values())
    for series_id in series_ids:
        path = fetch(series_id)
        lines = path.read_text().strip().splitlines()
        print(f"{series_id:<18} {len(lines) - 1:>6} rows, last: {lines[-1]}")


if __name__ == "__main__":
    main()
