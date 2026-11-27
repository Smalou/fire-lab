"""Markowitz mean-variance toolkit, risk parity and CAPM.

Inputs are monthly simple returns; outputs are annualised
(arithmetic mean x 12, covariance x 12) unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from core.models import OptimizationMethod
from core.returns import MONTHS_PER_YEAR, arithmetic_to_geometric

WEIGHT_TOL = 1e-6


class OptimizationError(RuntimeError):
    """Raised when an optimiser fails to converge to a feasible solution."""


@dataclass
class Moments:
    assets: list[str]
    mu: np.ndarray  # annual arithmetic mean of simple returns
    cov: np.ndarray  # annual covariance
    n_obs: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    method: str
    psd_repaired: bool = False
    min_eigenvalue: float = 0.0
    shrinkage_intensity: float | None = None

    @property
    def vol(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def corr(self) -> np.ndarray:
        v = self.vol
        with np.errstate(invalid="ignore", divide="ignore"):
            c = self.cov / np.outer(v, v)
        np.fill_diagonal(c, 1.0)
        return np.nan_to_num(c)


@dataclass
class OptimizationResult:
    method: str
    weights: np.ndarray
    success: bool
    message: str
    iterations: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- estimation


def is_psd(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    sym = (matrix + matrix.T) / 2
    return bool(np.linalg.eigvalsh(sym).min() >= -tol)


def nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Eigenvalue clipping (Higham-style projection onto the PSD cone)."""
    sym = (matrix + matrix.T) / 2
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, floor, None)
    out = vecs @ np.diag(vals) @ vecs.T
    return (out + out.T) / 2


def estimate_moments(returns: pd.DataFrame, method: str = "sample") -> Moments:
    """Annualised mean vector and covariance with PSD control.

    ``method``: "sample" (unbiased sample covariance) or "ledoit_wolf"
    (shrinkage towards scaled identity, scikit-learn).
    """
    clean = returns.dropna(how="any")
    if len(clean) < max(12, clean.shape[1] + 2):
        raise ValueError(
            f"Za mało obserwacji do estymacji ({len(clean)} mies.) dla {clean.shape[1]} aktywów. Wydłuż okres danych."
        )
    x = clean.to_numpy(dtype=float)
    mu_m = x.mean(axis=0)
    shrink = None
    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(x)
        # sklearn uses the MLE (1/n); rescale to the unbiased estimator for consistency.
        cov_m = lw.covariance_ * len(x) / (len(x) - 1)
        shrink = float(lw.shrinkage_)
    else:
        cov_m = np.cov(x, rowvar=False, ddof=1)
    cov_m = np.atleast_2d(cov_m)
    min_eig = float(np.linalg.eigvalsh((cov_m + cov_m.T) / 2).min())
    repaired = False
    if min_eig < -1e-12:
        cov_m = nearest_psd(cov_m)
        repaired = True
    return Moments(
        assets=list(clean.columns),
        mu=mu_m * MONTHS_PER_YEAR,
        cov=cov_m * MONTHS_PER_YEAR,
        n_obs=len(clean),
        start=clean.index[0],
        end=clean.index[-1],
        method=method,
        psd_repaired=repaired,
        min_eigenvalue=min_eig * MONTHS_PER_YEAR,
        shrinkage_intensity=shrink,
    )


# ---------------------------------------------------------------- portfolio stats


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(w, mu))


def portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def portfolio_geometric_return(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Expected annual geometric (median) return from monthly lognormal approximation."""
    m_month = portfolio_return(w, mu) / MONTHS_PER_YEAR
    s_month = portfolio_volatility(w, cov) / np.sqrt(MONTHS_PER_YEAR)
    g_month = arithmetic_to_geometric(m_month, s_month)
    return float((1 + g_month) ** MONTHS_PER_YEAR - 1)


def sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    vol = portfolio_volatility(w, cov)
    return (portfolio_return(w, mu) - rf) / vol if vol > 0 else float("nan")


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Euler decomposition: RC_i = w_i (Σw)_i / σ_p; sums to σ_p."""
    vol = portfolio_volatility(w, cov)
    if vol == 0:
        return np.zeros_like(w)
    return w * (cov @ w) / vol


def check_weights(w: np.ndarray, allow_short: bool = False, tol: float = WEIGHT_TOL) -> None:
    if not np.isfinite(w).all():
        raise OptimizationError("Wagi zawierają wartości nieskończone lub NaN.")
    if abs(w.sum() - 1.0) > 1e-4:
        raise OptimizationError(f"Suma wag = {w.sum():.6f} ≠ 1.")
    if not allow_short and (w < -1e-4).any():
        raise OptimizationError("Wykryto ujemne wagi przy zakazie krótkiej sprzedaży.")


def clean_weights(w: np.ndarray, allow_short: bool = False) -> np.ndarray:
    w = np.where(np.abs(w) < 1e-8, 0.0, w)
    if not allow_short:
        w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s != 0 else w


# ---------------------------------------------------------------- optimisers


def _bounds(n: int, min_w: float, max_w: float, allow_short: bool) -> list[tuple[float, float]]:
    lo = min_w if allow_short else max(min_w, 0.0)
    return [(lo, max_w)] * n


def _feasible(n: int, min_w: float, max_w: float, allow_short: bool) -> bool:
    lo = min_w if allow_short else max(min_w, 0.0)
    return n * lo <= 1.0 + 1e-12 and n * max_w >= 1.0 - 1e-12


def _solve(objective, n, bounds, constraints, x0=None, name="") -> OptimizationResult:
    starts = [np.full(n, 1.0 / n)] if x0 is None else [x0]
    rng = np.random.default_rng(0)
    starts += [rng.dirichlet(np.ones(n)) for _ in range(3)]
    best = None
    for s in starts:
        s = np.clip(s, [b[0] for b in bounds], [b[1] for b in bounds])
        res = optimize.minimize(
            objective, s, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12}
        )
        feasible = res.success and all(
            (c["fun"](res.x) >= -1e-6) if c["type"] == "ineq" else abs(c["fun"](res.x)) < 1e-6 for c in constraints
        )
        if feasible and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return OptimizationResult(name, np.full(n, np.nan), False, "Optymalizator nie znalazł rozwiązania dopuszczalnego (SLSQP nie osiągnął zbieżności).")
    return OptimizationResult(name, best.x, True, str(best.message), int(best.nit))


def equal_weight(n: int) -> OptimizationResult:
    return OptimizationResult("equal_weight", np.full(n, 1.0 / n), True, "Wagi równe 1/N.")


def min_variance(cov: np.ndarray, min_w=0.0, max_w=1.0, allow_short=False) -> OptimizationResult:
    n = cov.shape[0]
    if not _feasible(n, min_w, max_w, allow_short):
        return OptimizationResult("min_variance", np.full(n, np.nan), False, "Ograniczenia wag są sprzeczne (nie da się uzyskać sumy 100%).")
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = _solve(lambda w: w @ cov @ w, n, _bounds(n, min_w, max_w, allow_short), cons, name="min_variance")
    if res.success:
        res.weights = clean_weights(res.weights, allow_short)
    return res


def max_sharpe(mu, cov, rf, min_w=0.0, max_w=1.0, allow_short=False) -> OptimizationResult:
    n = len(mu)
    if not _feasible(n, min_w, max_w, allow_short):
        return OptimizationResult("max_sharpe", np.full(n, np.nan), False, "Ograniczenia wag są sprzeczne.")
    if np.all(mu <= rf):
        return OptimizationResult(
            "max_sharpe", np.full(n, np.nan), False,
            "Żadne aktywo nie ma oczekiwanej stopy wyższej od stopy wolnej od ryzyka – portfel styczny nie istnieje.",
        )

    def neg_sharpe(w):
        v = np.sqrt(max(w @ cov @ w, 1e-16))
        return -(w @ mu - rf) / v

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = _solve(neg_sharpe, n, _bounds(n, min_w, max_w, allow_short), cons, name="max_sharpe")
    if res.success:
        res.weights = clean_weights(res.weights, allow_short)
        if sharpe(res.weights, mu, cov, rf) <= 0:
            res.success = False
            res.message = "Maksymalny Sharpe ≤ 0 – portfel styczny niezdefiniowany przy tej stopie wolnej od ryzyka."
    return res


def risk_parity(cov: np.ndarray, min_w=0.0, max_w=1.0) -> OptimizationResult:
    """Equal risk contribution (long-only).

    Step 1: convex formulation of Spinu (2013): min ½wᵀΣw − Σ b_i ln w_i, then normalise.
    Step 2: if bounds bind, refine with SLSQP on squared deviations of risk shares.
    """
    n = cov.shape[0]
    b = np.full(n, 1.0 / n)

    def f(y):
        return 0.5 * y @ cov @ y - b @ np.log(y)

    def g(y):
        return cov @ y - b / y

    res = optimize.minimize(f, np.full(n, 1.0), jac=g, method="L-BFGS-B", bounds=[(1e-10, None)] * n)
    w = res.x / res.x.sum()
    ok = res.success
    lo = max(min_w, 0.0)
    if (w < lo - 1e-9).any() or (w > max_w + 1e-9).any():
        if not _feasible(n, lo, max_w, False):
            return OptimizationResult("risk_parity", np.full(n, np.nan), False, "Ograniczenia wag są sprzeczne.")

        def obj(x):
            rc = x * (cov @ x)
            share = rc / max(rc.sum(), 1e-16)
            return float(np.sum((share - b) ** 2)) * 1e4

        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0}]
        sol = _solve(obj, n, _bounds(n, lo, max_w, False), cons, x0=np.clip(w, lo, max_w), name="risk_parity")
        if not sol.success:
            return sol
        w, ok = sol.weights, True
    msg = "Równy udział aktywów w ryzyku portfela." if ok else str(res.message)
    return OptimizationResult("risk_parity", clean_weights(w), ok, msg, int(getattr(res, "nit", 0)))


def max_return_feasible(mu, min_w=0.0, max_w=1.0, allow_short=False) -> float:
    n = len(mu)
    lo = min_w if allow_short else max(min_w, 0.0)
    res = optimize.linprog(-mu, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(lo, max_w)] * n, method="highs")
    return float(-res.fun) if res.success else float(np.max(mu))


def min_return_feasible(mu, min_w=0.0, max_w=1.0, allow_short=False) -> float:
    n = len(mu)
    lo = min_w if allow_short else max(min_w, 0.0)
    res = optimize.linprog(mu, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(lo, max_w)] * n, method="highs")
    return float(res.fun) if res.success else float(np.min(mu))


def target_return_portfolio(mu, cov, target, min_w=0.0, max_w=1.0, allow_short=False) -> OptimizationResult:
    n = len(mu)
    hi = max_return_feasible(mu, min_w, max_w, allow_short)
    lo = min_return_feasible(mu, min_w, max_w, allow_short)
    if target > hi + 1e-9 or target < lo - 1e-9:
        return OptimizationResult(
            "target_return", np.full(n, np.nan), False,
            f"Zadana stopa {target:.2%} jest poza zakresem osiągalnym przy tych ograniczeniach ({lo:.2%} – {hi:.2%}).",
        )
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu - target},
    ]
    res = _solve(lambda w: w @ cov @ w, n, _bounds(n, min_w, max_w, allow_short), cons, name="target_return")
    if res.success:
        res.weights = clean_weights(res.weights, allow_short)
    return res


def optimize_portfolio(
    method: OptimizationMethod | str,
    moments: Moments,
    rf: float,
    min_w: float = 0.0,
    max_w: float = 1.0,
    allow_short: bool = False,
    target_return: float | None = None,
    manual_weights: np.ndarray | None = None,
) -> OptimizationResult:
    method = OptimizationMethod(method)
    mu, cov, n = moments.mu, moments.cov, len(moments.mu)
    if method == OptimizationMethod.MANUAL:
        w = np.asarray(manual_weights if manual_weights is not None else np.full(n, 1 / n), dtype=float)
        return OptimizationResult("manual", w, True, "Alokacja ręczna.")
    if method == OptimizationMethod.EQUAL_WEIGHT:
        return equal_weight(n)
    if method == OptimizationMethod.MIN_VARIANCE:
        return min_variance(cov, min_w, max_w, allow_short)
    if method == OptimizationMethod.MAX_SHARPE:
        return max_sharpe(mu, cov, rf, min_w, max_w, allow_short)
    if method == OptimizationMethod.RISK_PARITY:
        return risk_parity(cov, min_w, max_w)
    if method == OptimizationMethod.TARGET_RETURN:
        if target_return is None:
            return OptimizationResult("target_return", np.full(n, np.nan), False, "Nie podano docelowej stopy zwrotu.")
        return target_return_portfolio(mu, cov, target_return, min_w, max_w, allow_short)
    raise ValueError(method)


# ---------------------------------------------------------------- frontier


@dataclass
class Frontier:
    returns: np.ndarray
    vols: np.ndarray
    weights: np.ndarray  # (k, n)
    failures: int


def efficient_frontier(mu, cov, min_w=0.0, max_w=1.0, allow_short=False, n_points: int = 30) -> Frontier:
    mv = min_variance(cov, min_w, max_w, allow_short)
    if not mv.success:
        raise OptimizationError(mv.message)
    r_lo = portfolio_return(mv.weights, mu)
    r_hi = max_return_feasible(mu, min_w, max_w, allow_short)
    targets = np.linspace(r_lo, r_hi, n_points)
    rets, vols, ws, fails = [], [], [], 0
    for t in targets:
        res = target_return_portfolio(mu, cov, t, min_w, max_w, allow_short) if t > r_lo + 1e-9 else mv
        if not res.success:
            fails += 1
            continue
        rets.append(portfolio_return(res.weights, mu))
        vols.append(portfolio_volatility(res.weights, cov))
        ws.append(res.weights)
    return Frontier(np.array(rets), np.array(vols), np.array(ws), fails)


def random_portfolios(mu, cov, n: int = 3000, seed: int = 7, min_w=0.0, max_w=1.0):
    """Long-only Dirichlet sample respecting the weight bounds (rejection)."""
    rng = np.random.default_rng(seed)
    k = len(mu)
    w = rng.dirichlet(np.full(k, 0.8), size=n * 3)
    mask = (w >= max(min_w, 0.0) - 1e-12).all(axis=1) & (w <= max_w + 1e-12).all(axis=1)
    w = w[mask][:n]
    r = w @ mu
    v = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    return w, r, v


# ---------------------------------------------------------------- CAPM


@dataclass
class CapmResult:
    assets: list[str]
    betas: np.ndarray
    expected: np.ndarray  # historical arithmetic annual
    sml: np.ndarray  # rf + beta * (E[Rm] - rf)
    alphas: np.ndarray
    market_return: float


def capm_analysis(returns: pd.DataFrame, market: pd.Series, rf: float) -> CapmResult:
    data = returns.join(market.rename("__mkt__"), how="inner").dropna()
    m = data.pop("__mkt__").to_numpy()
    x = data.to_numpy()
    var_m = np.var(m, ddof=1)
    betas = np.array([np.cov(x[:, i], m, ddof=1)[0, 1] / var_m for i in range(x.shape[1])])
    exp = x.mean(axis=0) * MONTHS_PER_YEAR
    rm = m.mean() * MONTHS_PER_YEAR
    sml = rf + betas * (rm - rf)
    return CapmResult(list(data.columns), betas, exp, sml, exp - sml, float(rm))
