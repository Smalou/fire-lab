"""Sequence of Returns Risk experiment.

With no cash flows the terminal value is invariant to the order of returns
(multiplication is commutative). With regular withdrawals it is not: losses
early in retirement are "locked in" because capital is withdrawn at low prices.
Convention: withdrawal at the **start** of each year, then the year's return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequencePath:
    label: str
    returns: np.ndarray
    wealth: np.ndarray  # length n+1, start-of-year wealth (index 0 = initial)
    depletion_year: int | None
    total_withdrawn: float


def simulate_sequence(returns, capital: float, withdrawal: float, label: str = "") -> SequencePath:
    r = np.asarray(returns, dtype=float)
    w = np.empty(r.size + 1)
    w[0] = capital
    v = capital
    paid = 0.0
    depleted = None
    for t, rt in enumerate(r):
        take = min(withdrawal, v)
        paid += take
        v -= take
        if v <= 1e-9 and depleted is None and withdrawal > 0:
            depleted = t + 1 if take < withdrawal else t + 1
        v *= 1.0 + rt
        v = max(v, 0.0)
        w[t + 1] = v
    return SequencePath(label, r, w, depleted, paid)


def crash_sequence(n_years: int, crash_year: int, crash_return: float, cagr: float) -> np.ndarray:
    """Constant returns with a single crash at ``crash_year`` (1-based); geometric mean fixed at ``cagr``."""
    if not 1 <= crash_year <= n_years:
        raise ValueError("Rok krachu poza horyzontem.")
    other = ((1 + cagr) ** n_years / (1 + crash_return)) ** (1 / (n_years - 1)) - 1 if n_years > 1 else crash_return
    r = np.full(n_years, other)
    r[crash_year - 1] = crash_return
    return r


def orderings(returns) -> dict[str, np.ndarray]:
    r = np.asarray(returns, dtype=float)
    return {
        "Kolejność oryginalna": r,
        "Najgorsze lata na początku": np.sort(r),
        "Najlepsze lata na początku": np.sort(r)[::-1],
        "Kolejność odwrócona": r[::-1],
    }


def geometric_mean(returns) -> float:
    r = np.asarray(returns, dtype=float)
    return float(np.prod(1 + r) ** (1 / r.size) - 1)
