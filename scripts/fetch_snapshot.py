"""Download a snapshot of REAL market data from Yahoo Finance for offline use.

Run on a computer with Internet access:

    python scripts/fetch_snapshot.py

Saves daily adjusted close, close, dividends and FX rates for every
instrument in data/instruments.json to data/snapshot/. The app's data source
"Migawka Yahoo Finance" then works without Internet and shows the snapshot date.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import catalog  # noqa: E402
from services.market_data import SNAPSHOT_DIR, download_yfinance  # noqa: E402


def main() -> None:
    tickers = [i.id for i in catalog.instruments("yfinance")]
    fx = sorted({catalog.fx_ticker(c, b) for c in ("USD", "EUR") for b in ("PLN", "USD", "EUR") if c != b} | {"PLNUSD=X", "PLNEUR=X"})
    print(f"Pobieranie {len(tickers)} instrumentów i {len(fx)} kursów walut…")
    raw = download_yfinance(tickers + fx, "1993-01-01", None)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    raw["Adj Close"][tickers].to_csv(SNAPSHOT_DIR / "adj_close.csv")
    raw["Close"][tickers].to_csv(SNAPSHOT_DIR / "close.csv")
    raw["Dividends"][tickers].fillna(0.0).to_csv(SNAPSHOT_DIR / "dividends.csv")
    raw["Close"][fx].to_csv(SNAPSHOT_DIR / "fx.csv")
    meta = {"fetched_at": datetime.now().replace(microsecond=0).isoformat(), "source": "Yahoo Finance via yfinance",
            "tickers": tickers, "fx": fx}
    (SNAPSHOT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Zapisano migawkę do {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
