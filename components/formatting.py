"""Polish number, currency, percentage and date formatting."""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

NBSP = " "
CURRENCY_SYMBOLS = {"PLN": "zł", "USD": "USD", "EUR": "EUR"}
MONTHS_PL = ["sty", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrz", "paź", "lis", "gru"]


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def fmt_num(x, decimals: int = 0) -> str:
    if _is_missing(x):
        return "—"
    s = f"{x:,.{decimals}f}"
    return s.replace(",", NBSP).replace(".", ",")


def fmt_money(x, currency: str = "PLN", decimals: int = 0, compact: bool = False) -> str:
    if _is_missing(x):
        return "—"
    sym = CURRENCY_SYMBOLS.get(currency, currency)
    if compact:
        ax = abs(x)
        if ax >= 1e9:
            return f"{fmt_num(x / 1e9, 2)}{NBSP}mld{NBSP}{sym}"
        if ax >= 1e6:
            return f"{fmt_num(x / 1e6, 2)}{NBSP}mln{NBSP}{sym}"
        if ax >= 1e4:
            return f"{fmt_num(x / 1e3, 0)}{NBSP}tys.{NBSP}{sym}"
    return f"{fmt_num(x, decimals)}{NBSP}{sym}"


def fmt_pct(x, decimals: int = 1, signed: bool = False) -> str:
    if _is_missing(x):
        return "—"
    s = fmt_num(x * 100, decimals)
    if signed and x > 0:
        s = "+" + s
    return f"{s}%"


def fmt_pp(x, decimals: int = 1) -> str:
    """Percentage points."""
    if _is_missing(x):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{fmt_num(x * 100, decimals)}{NBSP}p.p."


def fmt_ratio(x, decimals: int = 2) -> str:
    return fmt_num(x, decimals)


def fmt_date(d) -> str:
    if d is None or (isinstance(d, float) and math.isnan(d)) or d is pd.NaT:
        return "—"
    if isinstance(d, (pd.Timestamp, datetime, date)):
        return d.strftime("%d.%m.%Y")
    return str(d)


def fmt_datetime(d) -> str:
    if d is None:
        return "—"
    return d.strftime("%d.%m.%Y, %H:%M")


def fmt_month(d) -> str:
    if d is None or d is pd.NaT:
        return "—"
    return f"{MONTHS_PL[d.month - 1]} {d.year}"


def fmt_age(x, decimals: int = 1) -> str:
    """Age in years with Polish grammar (e.g. '52,2 roku', '50 lat')."""
    if _is_missing(x):
        return "—"
    return years_pl(round(float(x), decimals))


def years_pl(n: float) -> str:
    """Polish plural of 'rok'."""
    if _is_missing(n):
        return "—"
    if abs(n - round(n)) > 1e-9:
        return f"{fmt_num(n, 1)}{NBSP}roku"
    n = int(round(n))
    if n == 1:
        word = "rok"
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        word = "lata"
    else:
        word = "lat"
    return f"{n}{NBSP}{word}"


def polish_frame(df: pd.DataFrame, money: list[str] = (), pct: list[str] = (), num: dict[str, int] | None = None,
                 age: list[str] = (), currency: str = "PLN", pct_is_fraction: bool = True) -> pd.DataFrame:
    """Return a copy with read-only numeric columns rendered in Polish format (for st.dataframe display)."""
    out = df.copy()
    for c in money:
        if c in out:
            out[c] = [fmt_money(v, currency) if v is not None and not pd.isna(v) else "—" for v in out[c]]
    for c in pct:
        if c in out:
            out[c] = [fmt_pct(v if pct_is_fraction else v / 100, 2) if v is not None and not pd.isna(v) else "—" for v in out[c]]
    for c, d in (num or {}).items():
        if c in out:
            out[c] = [fmt_num(v, d) if v is not None and not pd.isna(v) else "—" for v in out[c]]
    for c in age:
        if c in out:
            out[c] = [fmt_num(v, 1) if v is not None and not pd.isna(v) else "—" for v in out[c]]
    return out


def nearest(options, value):
    """Option closest to value (select_slider requires an exact member)."""
    return min(options, key=lambda o: abs(o - value))
