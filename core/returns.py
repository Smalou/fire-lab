"""Return conventions: compounding, annualisation, nominal/real conversion.

All functions accept scalars or NumPy arrays and never import Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def annual_to_monthly(rate):
    """Geometric de-annualisation: (1 + r)^(1/12) - 1."""
    return np.power(1.0 + np.asarray(rate, dtype=float), 1.0 / MONTHS_PER_YEAR) - 1.0


def monthly_to_annual(rate):
    """Geometric annualisation: (1 + r_m)^12 - 1."""
    return np.power(1.0 + np.asarray(rate, dtype=float), MONTHS_PER_YEAR) - 1.0


def nominal_to_real(nominal, inflation):
    """Fisher relation (exact): (1 + n) / (1 + pi) - 1."""
    return (1.0 + np.asarray(nominal, dtype=float)) / (1.0 + np.asarray(inflation, dtype=float)) - 1.0


def real_to_nominal(real, inflation):
    return (1.0 + np.asarray(real, dtype=float)) * (1.0 + np.asarray(inflation, dtype=float)) - 1.0


def deflator(n_periods: int, inflation_per_period: float) -> np.ndarray:
    """Price index I_t = (1 + pi)^t for t = 0..n_periods."""
    return np.power(1.0 + inflation_per_period, np.arange(n_periods + 1, dtype=float))


def simple_to_log(r):
    r = np.asarray(r, dtype=float)
    if np.any(r <= -1.0):
        raise ValueError("Stopa zwrotu ≤ -100% nie ma logarytmu.")
    return np.log1p(r)


def log_to_simple(x):
    return np.expm1(np.asarray(x, dtype=float))


def cumulative_return(returns) -> float:
    r = np.asarray(returns, dtype=float)
    return float(np.prod(1.0 + r) - 1.0)


def annualized_geometric_return(returns, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """CAGR implied by a series of periodic simple returns."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n) - 1.0)


def annualized_arithmetic_return(returns, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=float)
    return float(np.mean(r) * periods_per_year) if r.size else float("nan")


def annualized_volatility(returns, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def arithmetic_to_geometric(mean_simple: float, vol_simple: float) -> float:
    """Median (geometric) growth rate of a lognormal return with given simple mean and volatility.

    If 1 + R ~ LogNormal(m, s^2): E[R] = exp(m + s^2/2) - 1 and the geometric
    rate is exp(m) - 1. Works for any period length provided both inputs refer
    to the same period.
    """
    one_plus = 1.0 + mean_simple
    if one_plus <= 0:
        return -1.0
    s2 = np.log(1.0 + (vol_simple**2) / (one_plus**2))
    m = np.log(one_plus) - 0.5 * s2
    return float(np.expm1(m))


def prices_to_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple returns from (adjusted) prices; first row dropped, no fill-forward of returns."""
    return prices.pct_change(fill_method=None).iloc[1:]
