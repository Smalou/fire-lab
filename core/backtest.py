"""Walk-forward historical backtest with contributions, costs and rebalancing.

Timing (identical to the life-cycle engine): at the **start** of month t the
cash flow arrives and, on rebalancing dates, the portfolio is rebalanced to
weights computed **only from returns of months < t**; then the return of month
t is applied and the management fee charged.

Look-ahead protection: the decision function receives ``returns.iloc[:t]``
(a strictly historical slice). ``tests/test_backtest.py`` verifies that
modifying future data does not change past decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.metrics import performance_summary
from core.models import BacktestSettings, OptimizationMethod
from core.portfolio import estimate_moments, optimize_portfolio
from core.returns import annual_to_monthly

STRATEGY_LABELS: dict[str, str] = {
    "buy_hold": "Kup i trzymaj (Buy & Hold)",
    "scenario": "Portfel scenariusza (rebalancing)",
    "equal_weight": "Equal Weight (1/N)",
    "sixty_forty": "Portfel 60/40",
    "min_variance": "Minimum Variance (walk-forward)",
    "max_sharpe": "Maximum Sharpe (walk-forward)",
    "risk_parity": "Risk Parity (walk-forward)",
    "glide_path": "Glide Path (ścieżka schodząca)",
}
ESTIMATED = {"min_variance", "max_sharpe", "risk_parity"}
REBALANCE_STEP = {"none": None, "monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
EQUITY_CLASSES = {"global_equity", "us_equity", "europe_equity", "em_equity"}


@dataclass
class BacktestResult:
    strategy: str
    values: pd.Series  # end-of-month portfolio value
    twr_returns: pd.Series  # monthly time-weighted returns (net of costs and fees)
    invested: pd.Series  # cumulative contributions incl. initial investment
    weights: pd.DataFrame  # target weights decided at the start of each month
    costs: pd.Series  # transaction costs paid each month
    turnover: pd.Series  # traded value / portfolio value
    metrics: dict[str, float] = field(default_factory=dict)
    optimization_failures: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def investment_gain(self) -> pd.Series:
        return self.values - self.invested


def _norm(w: np.ndarray) -> np.ndarray:
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / w.size)


def split_equity(assets: list[str], classes: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    eq = np.array([classes.get(a, "") in EQUITY_CLASSES for a in assets])
    return eq, ~eq


class DecisionRule:
    """Maps a strictly historical return slice to target weights."""

    def __init__(self, strategy, assets, classes, scenario_weights, settings: BacktestSettings, rf,
                 min_w, max_w, allow_short, cov_method, n_months_total, t0):
        self.strategy = strategy
        self.assets = assets
        self.scn = _norm(np.asarray(scenario_weights, dtype=float)) if not allow_short else np.asarray(scenario_weights, float)
        self.settings = settings
        self.rf = rf
        self.bounds = (min_w, max_w, allow_short)
        self.cov_method = cov_method
        self.T = n_months_total
        self.t0 = t0
        self.prev: np.ndarray | None = None
        self.failures = 0
        self.eq_mask, self.other_mask = split_equity(assets, classes)
        if strategy == "sixty_forty":
            bonds = np.array([classes.get(a) == "bonds" for a in assets])
            if not self.eq_mask.any() or not bonds.any():
                raise ValueError("Strategia 60/40 wymaga co najmniej jednego aktywa akcyjnego i jednego obligacyjnego.")
            self.mask_bonds = bonds
        if strategy == "glide_path" and (not self.eq_mask.any() or not self.other_mask.any()):
            raise ValueError("Glide Path wymaga aktywów akcyjnych i nieakcyjnych w portfelu.")

    def _within(self, mask: np.ndarray) -> np.ndarray:
        sub = np.where(mask, np.clip(self.scn, 0, None), 0.0)
        if sub.sum() <= 0:
            sub = mask.astype(float)
        return sub / sub.sum()

    def __call__(self, history: pd.DataFrame, t: int) -> np.ndarray:
        n = len(self.assets)
        s = self.strategy
        if s in ("buy_hold", "scenario"):
            return self.scn.copy()
        if s == "equal_weight":
            return np.full(n, 1.0 / n)
        if s == "sixty_forty":
            return 0.6 * self._within(self.eq_mask) + 0.4 * self._within(self.mask_bonds)
        if s == "glide_path":
            span = max(self.T - 1 - self.t0, 1)
            prog = (t - self.t0) / span
            eq = self.settings.glide_start_equity + (self.settings.glide_end_equity - self.settings.glide_start_equity) * prog
            return eq * self._within(self.eq_mask) + (1 - eq) * self._within(self.other_mask)
        if s in ESTIMATED:
            window = history.iloc[-self.settings.estimation_months :]
            try:
                mom = estimate_moments(window, self.cov_method)
                res = optimize_portfolio(OptimizationMethod(s), mom, self.rf, *self.bounds)
            except Exception:  # noqa: BLE001 - reported as failure, never hidden
                res = None
            if res is None or not res.success:
                self.failures += 1
                return self.prev.copy() if self.prev is not None else np.full(n, 1.0 / n)
            self.prev = res.weights
            return res.weights
        raise ValueError(f"Nieznana strategia: {s}")


def run_backtest(
    returns: pd.DataFrame,
    strategy: str,
    classes: dict[str, str],
    scenario_weights,
    settings: BacktestSettings,
    rf: float,
    management_fee: float = 0.0,
    min_w: float = 0.0,
    max_w: float = 1.0,
    allow_short: bool = False,
    cov_method: str = "sample",
    benchmark: pd.Series | None = None,
    start_index: int | None = None,
) -> BacktestResult:
    data = returns.dropna(how="any")
    assets = list(data.columns)
    T = len(data)
    t0 = settings.estimation_months if start_index is None else start_index
    if T - t0 < 12:
        raise ValueError(
            f"Za krótka historia: {T} mies. danych, z czego {t0} mies. to okres estymacji. Potrzeba co najmniej 12 mies. testu."
        )
    rule = DecisionRule(strategy, assets, classes, scenario_weights, settings, rf, min_w, max_w, allow_short, cov_method, T, t0)
    step = REBALANCE_STEP[settings.rebalance]
    if strategy == "buy_hold":
        step = None
    cost_rate = settings.commission_rate + settings.spread_bps / 10_000
    fee_m = float(annual_to_monthly(management_fee))
    x = data.to_numpy(dtype=float)
    n = len(assets)

    h = np.zeros(n)
    target = None
    v_prev = 0.0
    values, twr, invested, costs, turnover, w_hist = [], [], [], [], [], []
    cum_in = 0.0
    for t in range(t0, T):
        flow = settings.monthly_contribution + (settings.initial_investment if t == t0 else 0.0)
        cum_in += flow
        rebalance = t == t0 or (step is not None and (t - t0) % step == 0)
        if rebalance:
            target = rule(data.iloc[:t], t)  # strictly past data
            pre = h.sum() + flow
            trades = target * pre - h
            # cost solves V' = pre - c·|w·V' - h|₁ approximately (first-order)
            c = cost_rate * np.abs(trades).sum()
            h = target * (pre - c)
            traded = np.abs(trades).sum()
        else:
            c = cost_rate * flow
            h = h + target * (flow - c)
            traded = flow
        denom = v_prev + flow
        h = h * (1.0 + x[t]) * (1.0 - fee_m)
        v_end = float(h.sum())
        twr.append(v_end / denom - 1.0 if denom > 0 else 0.0)
        values.append(v_end)
        invested.append(cum_in)
        costs.append(c)
        turnover.append(traded / denom if denom > 0 else 0.0)
        w_hist.append(target.copy())
        v_prev = v_end

    idx = data.index[t0:]
    res = BacktestResult(
        strategy=strategy,
        values=pd.Series(values, index=idx),
        twr_returns=pd.Series(twr, index=idx),
        invested=pd.Series(invested, index=idx),
        weights=pd.DataFrame(w_hist, index=idx, columns=assets),
        costs=pd.Series(costs, index=idx),
        turnover=pd.Series(turnover, index=idx),
        optimization_failures=rule.failures,
    )
    # money-weighted return: deposits at month starts, terminal value at last month end
    month_start = [d - pd.offsets.MonthEnd(1) for d in idx]
    flows = [-(settings.monthly_contribution + (settings.initial_investment if i == 0 else 0.0)) for i in range(len(idx))]
    cf_dates = month_start + [idx[-1]]
    cf = flows + [values[-1]]
    bench = benchmark.reindex(idx) if benchmark is not None else None
    res.metrics = performance_summary(res.twr_returns, rf, bench, cf, cf_dates)
    res.metrics["total_costs"] = float(np.sum(costs))
    res.metrics["avg_turnover_annual"] = float(np.mean(turnover[1:]) * 12) if len(turnover) > 1 else 0.0
    res.metrics["final_value"] = values[-1]
    res.metrics["total_invested"] = cum_in
    res.metrics["investment_gain"] = values[-1] - cum_in
    if rule.failures:
        res.notes.append(f"Optymalizator nie osiągnął zbieżności {rule.failures} raz(y) – zachowano poprzednie wagi.")
    return res
