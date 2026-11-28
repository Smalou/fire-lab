"""Monte Carlo generators of monthly portfolio returns.

Both models work on **asset-level** returns so that the correlation structure
between assets is preserved, and the portfolio is rebalanced monthly to the
target weights (portfolio return = wᵀ r_t).

* Parametric: multivariate normal (or Student-t with the same covariance) of
  monthly **log** returns, estimated on the estimation window.
* Historical circular block bootstrap of the vectors of monthly simple returns
  (keeps cross-sectional correlation and short-range autocorrelation).

Reproducibility: a single ``numpy.random.Generator`` seeded with ``seed``;
paths are produced in fixed-size chunks so results do not depend on the chunk
callback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.models import SimulationModel

CHUNK = 500


@dataclass
class ReturnModelSpec:
    model: SimulationModel
    n_paths: int
    n_months: int
    seed: int
    block_months: int = 12
    fat_tails: bool = False
    t_dof: float = 5.0
    return_haircut: float = 0.0  # annual, subtracted from log returns
    shrinkage: bool = False  # Ledoit–Wolf covariance of log returns (parametric model)


def _parametric_chunk(rng, n, months, mean_log, chol, fat_tails, dof):
    k = mean_log.size
    z = rng.standard_normal((n, months, k))
    if fat_tails:
        # Multivariate t with covariance equal to Σ: scale by sqrt((ν-2)/ν) and divide by sqrt(χ²/ν).
        g = rng.chisquare(dof, size=(n, months, 1)) / dof
        z = z * np.sqrt((dof - 2.0) / dof) / np.sqrt(g)
    return mean_log + z @ chol.T


def simulate_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    spec: ReturnModelSpec,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Return array (n_paths, n_months) of nominal monthly simple portfolio returns."""
    data = asset_returns.dropna(how="any")
    w = np.asarray(weights, dtype=float)
    if data.shape[1] != w.size:
        raise ValueError("Liczba wag nie odpowiada liczbie aktywów.")
    if len(data) < 24:
        raise ValueError("Do symulacji potrzeba co najmniej 24 miesięcy wspólnej historii aktywów.")
    rng = np.random.default_rng(spec.seed)
    haircut_m = spec.return_haircut / 12.0
    out = np.empty((spec.n_paths, spec.n_months))

    if spec.model == SimulationModel.PARAMETRIC:
        logs = np.log1p(data.to_numpy(dtype=float))
        mean_log = logs.mean(axis=0) - haircut_m
        if spec.shrinkage:
            from sklearn.covariance import LedoitWolf

            cov = LedoitWolf().fit(logs).covariance_ * len(logs) / (len(logs) - 1)
        else:
            cov = np.cov(logs, rowvar=False, ddof=1)
        cov = np.atleast_2d(cov)
        vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
        vals = np.clip(vals, 1e-14, None)  # PSD safeguard
        chol = vecs @ np.diag(np.sqrt(vals))
        for start in range(0, spec.n_paths, CHUNK):
            n = min(CHUNK, spec.n_paths - start)
            x = _parametric_chunk(rng, n, spec.n_months, mean_log, chol, spec.fat_tails, spec.t_dof)
            out[start : start + n] = np.expm1(x) @ w
            if progress:
                progress((start + n) / spec.n_paths)
    elif spec.model == SimulationModel.BOOTSTRAP:
        hist = data.to_numpy(dtype=float) @ w  # monthly-rebalanced historical portfolio returns
        hist = np.expm1(np.log1p(hist) - haircut_m)
        t = hist.size
        b = max(1, min(spec.block_months, t))
        n_blocks = int(np.ceil(spec.n_months / b))
        for start in range(0, spec.n_paths, CHUNK):
            n = min(CHUNK, spec.n_paths - start)
            starts = rng.integers(0, t, size=(n, n_blocks))
            idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % t  # circular blocks
            out[start : start + n] = hist[idx.reshape(n, -1)[:, : spec.n_months]]
            if progress:
                progress((start + n) / spec.n_paths)
    else:
        raise ValueError(spec.model)
    return out


def horizon_returns(port_returns: np.ndarray, months: int) -> np.ndarray:
    """Cumulative simple return over the first ``months`` months of each path (no cash flows)."""
    months = min(months, port_returns.shape[1])
    return np.prod(1.0 + port_returns[:, :months], axis=1) - 1.0
