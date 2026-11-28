"""Market data acquisition and preparation (no Streamlit imports).

Sources
-------
* ``demo``     – clearly labelled synthetic data shipped with the app (offline).
* ``yfinance`` – live download from Yahoo Finance (adjusted prices + dividends + FX).
* ``snapshot`` – real Yahoo Finance data saved earlier by ``scripts/fetch_snapshot.py`` (offline).
* ``csv``      – user-imported prices.

All sources end up as **monthly** total-return price indices in the base
currency. Adjusted close already contains reinvested dividends, so dividends are
never added to returns a second time; raw dividends are only used to estimate
the income yield for the "income only" withdrawal strategy.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from services import catalog

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"
SNAPSHOT_DIR = ROOT / "data" / "snapshot"
MIN_MONTHS = 36
FFILL_DAYS = 5


class DataUnavailableError(RuntimeError):
    """Raised when requested data cannot be obtained; message is user-facing (Polish)."""


@dataclass
class MarketData:
    prices: pd.DataFrame  # monthly total-return prices in base currency
    income: pd.DataFrame | None  # monthly income per unit in base currency (None = no data)
    source: str
    source_label: str
    is_synthetic: bool
    retrieved_at: datetime
    base_currency: str
    warnings: list[str] = field(default_factory=list)

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change(fill_method=None).iloc[1:]

    def availability(self) -> pd.DataFrame:
        rows = []
        for c in self.prices.columns:
            s = self.prices[c].dropna()
            rows.append(
                {
                    "asset": c,
                    "first": s.index.min() if not s.empty else pd.NaT,
                    "last": s.index.max() if not s.empty else pd.NaT,
                    "months": max(len(s) - 1, 0),
                }
            )
        return pd.DataFrame(rows).set_index("asset")

    def common_returns(self, start=None, end=None) -> pd.DataFrame:
        r = self.returns
        if start is not None:
            r = r.loc[pd.Timestamp(start) :]
        if end is not None:
            r = r.loc[: pd.Timestamp(end)]
        return r.dropna(how="any")

    def income_yield(self, assets: list[str], start=None, end=None) -> pd.Series | None:
        """Mean trailing-12M income yield per asset over the window (None if no income data)."""
        if self.income is None:
            return None
        inc = self.income.reindex(columns=assets)
        px = self.prices.reindex(columns=assets)
        trailing = inc.rolling(12).sum() / px.shift(12)
        if start is not None:
            trailing = trailing.loc[pd.Timestamp(start) :]
        if end is not None:
            trailing = trailing.loc[: pd.Timestamp(end)]
        y = trailing.mean()
        return None if y.isna().all() else y.fillna(0.0)


@dataclass
class DataCheck:
    ok: bool
    messages: list[str]
    suggested_start: pd.Timestamp | None = None
    substitutes: dict[str, str] = field(default_factory=dict)
    n_months: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


# ------------------------------------------------------------------ demo


def load_demo(assets: list[str]) -> MarketData:
    prices = pd.read_csv(DEMO_DIR / "synthetic_monthly_prices.csv", index_col=0, parse_dates=True)
    income = pd.read_csv(DEMO_DIR / "synthetic_monthly_income.csv", index_col=0, parse_dates=True)
    missing = [a for a in assets if a not in prices.columns]
    if missing:
        raise DataUnavailableError("Brak w danych demonstracyjnych: " + ", ".join(missing))
    stamp = datetime.fromtimestamp((DEMO_DIR / "synthetic_monthly_prices.csv").stat().st_mtime).replace(microsecond=0)
    return MarketData(
        prices=prices[assets],
        income=income[assets],
        source="demo",
        source_label="Dane syntetyczne (tryb demonstracyjny) – nie są notowaniami rynkowymi",
        is_synthetic=True,
        retrieved_at=stamp,
        base_currency="PLN",
    )


# ------------------------------------------------------------------ yfinance / snapshot


def _flatten_download(df: pd.DataFrame, field_name: str, tickers: list[str]) -> pd.DataFrame:
    """Extract one field from a yfinance download regardless of column layout."""
    if df is None or df.empty:
        return pd.DataFrame(columns=tickers)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if field_name in lvl0:
            out = df[field_name]
        else:  # group_by="ticker" layout
            out = df.xs(field_name, axis=1, level=1)
    else:
        out = df[[field_name]].rename(columns={field_name: tickers[0]})
    out = out.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    return out.reindex(columns=tickers)


def download_yfinance(tickers: list[str], start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
    """Raw daily data: adjusted close, close and dividends (local currencies)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailableError("Biblioteka yfinance nie jest zainstalowana.") from exc
    try:
        raw = yf.download(
            tickers, start=start or "1990-01-01", end=end, auto_adjust=False, actions=True,
            progress=False, group_by="column", threads=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise DataUnavailableError(f"Nie udało się pobrać danych z Yahoo Finance: {exc}") from exc
    if raw is None or raw.empty:
        raise DataUnavailableError(
            "Yahoo Finance nie zwróciło danych (brak połączenia z Internetem lub błędne tickery). "
            "Użyj trybu demonstracyjnego, migawki danych lub importu CSV."
        )
    out = {f: _flatten_download(raw, f, tickers) for f in ("Adj Close", "Close", "Dividends")}
    return out


def prepare_monthly(
    adj_close: pd.DataFrame,
    close: pd.DataFrame | None,
    dividends: pd.DataFrame | None,
    fx: pd.DataFrame,
    currencies: dict[str, str],
    base: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Convert daily local-currency data to monthly base-currency series.

    price_base(t) = price_local(t) × FX_{local→base}(t) using the FX rate of the same
    day (forward-filled over FX holidays). Gaps up to FFILL_DAYS trading days are
    forward-filled; longer gaps stay missing and are reported.
    """
    warnings: list[str] = []
    adj = adj_close.sort_index().ffill(limit=FFILL_DAYS)
    fx = fx.sort_index().reindex(adj.index.union(fx.index)).ffill().reindex(adj.index)
    conv = pd.DataFrame(index=adj.index)
    for c in adj.columns:
        cur = currencies.get(c, base)
        rate = 1.0 if cur == base else fx.get(catalog.fx_ticker(cur, base))
        if rate is None or (not np.isscalar(rate) and rate.isna().all()):
            warnings.append(f"Brak kursu {cur}/{base} – pominięto {c}.")
            continue
        conv[c] = adj[c] * rate
    monthly = conv.resample("ME").last()
    # drop months before the first valid observation, keep internal gaps visible
    income = None
    if dividends is not None and close is not None and not dividends.empty:
        div_b = pd.DataFrame(index=dividends.index)
        for c in conv.columns:
            cur = currencies.get(c, base)
            rate = 1.0 if cur == base else fx.reindex(dividends.index).ffill().get(catalog.fx_ticker(cur, base))
            div_b[c] = dividends[c].fillna(0.0) * rate
        income_m = div_b.resample("ME").sum()
        # income per unit of the *adjusted* series: scale by adj/close ratio at month end
        ratio = (adj_close / close).resample("ME").last().reindex(columns=conv.columns)
        income = (income_m * ratio).reindex(monthly.index)
        income = income.where(monthly.notna())
    for c in monthly.columns:
        s = monthly[c]
        first = s.first_valid_index()
        if first is not None and s.loc[first:].isna().any():
            warnings.append(f"{c}: luki w notowaniach dłuższe niż {FFILL_DAYS} dni – miesiące bez danych pozostawiono jako brakujące.")
    return monthly, income, warnings


def load_yfinance(assets: list[str], base: str, start=None, end=None) -> MarketData:
    currencies = catalog.currencies_for(assets, base)
    fx_needed = sorted({catalog.fx_ticker(c, base) for c in currencies.values() if c != base})
    raw = download_yfinance(assets + fx_needed, str(start) if start else None, str(end) if end else None)
    adj = raw["Adj Close"][assets]
    if adj.dropna(how="all").empty:
        raise DataUnavailableError("Brak notowań dla wybranych instrumentów.")
    fx = raw["Close"][fx_needed] if fx_needed else pd.DataFrame(index=adj.index)
    empty = [a for a in assets if adj[a].dropna().empty]
    prices, income, warns = prepare_monthly(adj, raw["Close"][assets], raw["Dividends"][assets], fx, currencies, base)
    if empty:
        warns.insert(0, "Brak danych dla: " + ", ".join(empty))
    return MarketData(prices, income, "yfinance", "Yahoo Finance (yfinance) – ceny skorygowane o dywidendy i splity",
                      False, datetime.now().replace(microsecond=0), base, warns)


def snapshot_available() -> bool:
    return (SNAPSHOT_DIR / "meta.json").exists()


def load_snapshot(assets: list[str], base: str) -> MarketData:
    if not snapshot_available():
        raise DataUnavailableError(
            "Brak migawki danych. Uruchom na komputerze z Internetem: python scripts/fetch_snapshot.py"
        )
    meta = json.loads((SNAPSHOT_DIR / "meta.json").read_text(encoding="utf-8"))
    read = lambda n: pd.read_csv(SNAPSHOT_DIR / n, index_col=0, parse_dates=True)  # noqa: E731
    adj, close, div, fx = read("adj_close.csv"), read("close.csv"), read("dividends.csv"), read("fx.csv")
    missing = [a for a in assets if a not in adj.columns]
    if missing:
        raise DataUnavailableError("Migawka nie zawiera: " + ", ".join(missing) + ". Odśwież migawkę.")
    currencies = catalog.currencies_for(assets, base)
    prices, income, warns = prepare_monthly(adj[assets], close[assets], div[assets], fx, currencies, base)
    return MarketData(prices, income, "snapshot", f"Yahoo Finance – migawka z {meta.get('fetched_at', '?')}",
                      False, datetime.fromisoformat(meta["fetched_at"]), base, warns)


# ------------------------------------------------------------------ CSV


def parse_price_csv(content: bytes) -> pd.DataFrame:
    """Parse a wide CSV: first column = date, remaining columns = prices (base currency).

    Accepts comma or semicolon separators and decimal comma. Returns monthly prices.
    """
    text = content.decode("utf-8-sig")
    sep = ";" if text.count(";") > text.count(",") else ","
    decimal = "," if sep == ";" else "."
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal)
    except Exception as exc:  # noqa: BLE001
        raise DataUnavailableError(f"Nie można odczytać pliku CSV: {exc}") from exc
    if df.shape[1] < 2:
        raise DataUnavailableError("Plik musi zawierać kolumnę daty i co najmniej jedną kolumnę cen.")
    date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce", format="ISO8601")
    if dates.isna().mean() > 0.05:  # e.g. 31.01.2020 (Polish convention)
        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True, format="mixed")
    if dates.isna().mean() > 0.05:
        raise DataUnavailableError(f"Kolumna „{date_col}” nie zawiera poprawnych dat.")
    df = df.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
    df.index = dates
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if (df <= 0).any().any():
        raise DataUnavailableError("Ceny muszą być dodatnie (plik zawiera wartości ≤ 0).")
    monthly = df.resample("ME").last()
    if len(monthly.dropna(how="any")) < MIN_MONTHS + 1:
        raise DataUnavailableError(f"Za krótka wspólna historia w pliku – wymagane co najmniej {MIN_MONTHS + 1} miesięcy.")
    monthly.columns = [str(c).strip() for c in monthly.columns]
    return monthly


def load_csv(prices: pd.DataFrame, assets: list[str], base: str, filename: str) -> MarketData:
    missing = [a for a in assets if a not in prices.columns]
    if missing:
        raise DataUnavailableError("Zaimportowany plik nie zawiera kolumn: " + ", ".join(missing))
    return MarketData(prices[assets], None, "csv", f"Import CSV użytkownika: {filename}", False,
                      datetime.now().replace(microsecond=0), base,
                      ["Dane CSV traktowane jako ceny całkowite (z dywidendami) w walucie bazowej – zweryfikuj ich źródło."])


# ------------------------------------------------------------------ validation


def check_coverage(md: MarketData, assets: list[str], start=None, end=None, min_months: int = MIN_MONTHS) -> DataCheck:
    """Validate that all assets have data over the requested period."""
    msgs: list[str] = []
    avail = md.availability()
    req_start = pd.Timestamp(start) if start else None
    req_end = pd.Timestamp(end) if end else None
    subs: dict[str, str] = {}
    late = []
    for a in assets:
        if a not in avail.index or pd.isna(avail.loc[a, "first"]):
            msgs.append(f"Brak jakichkolwiek danych dla {a}.")
            continue
        first = avail.loc[a, "first"]
        if req_start is not None and first > req_start + pd.offsets.MonthEnd(1):
            late.append((a, first))
    if late:
        for a, first in late:
            inst = catalog.get(a)
            sub = inst.substitute if inst else None
            hint = f" Możliwy, jawnie oznaczony zamiennik: {sub}." if sub else ""
            msgs.append(f"{a} ma dane dopiero od {first:%m.%Y}, a wybrano start {req_start:%m.%Y}.{hint}")
            if sub:
                subs[a] = sub
    r = md.common_returns(req_start, req_end)
    suggested = None
    firsts = [avail.loc[a, "first"] for a in assets if a in avail.index and pd.notna(avail.loc[a, "first"])]
    if firsts:
        suggested = max(firsts)
    ok = len(r) >= min_months and not any("Brak jakichkolwiek" in m for m in msgs)
    if len(r) < min_months:
        msgs.append(f"Wspólna historia wszystkich aktywów to {len(r)} mies. – wymagane co najmniej {min_months}.")
    return DataCheck(ok, msgs, suggested, subs, len(r), r.index.min() if len(r) else None, r.index.max() if len(r) else None)
