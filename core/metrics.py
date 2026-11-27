"""Performance and risk metrics for monthly return series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy import optimize

from core.returns import (
    MONTHS_PER_YEAR,
    annual_to_monthly,
    annualized_geometric_return,
    annualized_volatility,
    cumulative_return,
)


def wealth_index(returns) -> np.ndarray:
    """Growth of 1 unit, starting with 1.0 before the first return."""
    r = np.asarray(returns, dtype=float)
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def drawdown_series(returns) -> np.ndarray:
    idx = wealth_index(returns)
    peak = np.maximum.accumulate(idx)
    return idx / peak - 1.0


def max_drawdown(returns) -> float:
    """Maximum drawdown (negative number, e.g. -0.35) of a TWR index."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    return float(drawdown_series(r).min())


def sharpe_ratio(returns, rf_annual: float = 0.0, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """Annualised Sharpe: mean periodic excess return * k / (std * sqrt(k))."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return float("nan")
    excess = r - ((1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0)
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns, rf_annual: float = 0.0, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """Annualised Sortino with MAR = risk-free rate; downside deviation over all periods."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return float("nan")
    excess = r - ((1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0)
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return float("nan")
    return float(np.mean(excess) / dd * np.sqrt(periods_per_year))


@dataclass
class RelativeStats:
    beta: float
    alpha: float  # annualised Jensen's alpha
    tracking_error: float
    information_ratio: float


def relative_stats(returns, benchmark, rf_annual: float = 0.0) -> RelativeStats:
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    if r.shape != b.shape or r.size < 3:
        return RelativeStats(*(float("nan"),) * 4)
    rf_m = float(annual_to_monthly(rf_annual))
    ex_r, ex_b = r - rf_m, b - rf_m
    var_b = np.var(ex_b, ddof=1)
    beta = float(np.cov(ex_r, ex_b, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
    alpha = float((np.mean(ex_r) - beta * np.mean(ex_b)) * MONTHS_PER_YEAR) if np.isfinite(beta) else float("nan")
    active = r - b
    te = float(np.std(active, ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    ir = float(np.mean(active) * MONTHS_PER_YEAR / te) if te > 0 else float("nan")
    return RelativeStats(beta, alpha, te, ir)


def _year_fraction(d0: date, d1: date) -> float:
    return (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.0


def xirr(cash_flows: list[float] | np.ndarray, dates: list[date | datetime | pd.Timestamp]) -> float:
    """Money-weighted annual return. Investor convention: deposits negative, final value positive."""
    cf = np.asarray(cash_flows, dtype=float)
    if cf.size < 2 or not (np.any(cf > 0) and np.any(cf < 0)):
        return float("nan")
    t = np.array([_year_fraction(dates[0], d) for d in dates])

    def npv(rate: float) -> float:
        return float(np.sum(cf / np.power(1.0 + rate, t)))

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    return float(optimize.brentq(npv, lo, hi, xtol=1e-10, maxiter=500))


def rolling_volatility(returns: pd.Series, window: int = 12) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)


def rolling_sharpe(returns: pd.Series, rf_annual: float, window: int = 36) -> pd.Series:
    ex = returns - float(annual_to_monthly(rf_annual))
    return ex.rolling(window).mean() / ex.rolling(window).std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Year x month matrix of monthly returns (for heatmaps)."""
    df = pd.DataFrame({"r": returns.values}, index=pd.DatetimeIndex(returns.index))
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df.pivot_table(index="year", columns="month", values="r", aggfunc="first").reindex(columns=range(1, 13))


def performance_summary(
    twr_returns: pd.Series,
    rf_annual: float,
    benchmark_returns: pd.Series | None = None,
    cash_flows: list[float] | None = None,
    flow_dates: list | None = None,
) -> dict[str, float]:
    r = twr_returns.to_numpy(dtype=float)
    out = {
        "cumulative_return": cumulative_return(r),
        "cagr": annualized_geometric_return(r),
        "twr": cumulative_return(r),
        "volatility": annualized_volatility(r),
        "sharpe": sharpe_ratio(r, rf_annual),
        "sortino": sortino_ratio(r, rf_annual),
        "max_drawdown": max_drawdown(r),
    }
    out["xirr"] = xirr(cash_flows, flow_dates) if cash_flows is not None and flow_dates is not None else float("nan")
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(twr_returns.index).to_numpy(dtype=float)
        rel = relative_stats(r, b, rf_annual)
        out.update(beta=rel.beta, alpha=rel.alpha, tracking_error=rel.tracking_error, information_ratio=rel.information_ratio)
    return out
