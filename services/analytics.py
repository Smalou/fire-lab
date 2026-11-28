"""Glue between scenario, data and the computational core, with Streamlit caching.

Everything numerical is delegated to ``core``; this module only decides which
data window to use, caches expensive calls and packages results for views.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from core import portfolio as pf
from core.backtest import STRATEGY_LABELS, run_backtest
from core.fire import build_plan, deterministic_projection, required_capital
from core.lifecycle import deflate_returns, run_lifecycle, summarize
from core.models import BacktestSettings, RetirementRule, Scenario, WithdrawalStrategy
from core.returns import nominal_to_real
from core.risk import var_es
from core.simulation import ReturnModelSpec, horizon_returns, simulate_portfolio_returns
from core.withdrawal import run_decumulation, strategy_variants
from services import catalog
from services.market_data import (
    DataCheck,
    DataUnavailableError,
    MarketData,
    check_coverage,
    load_csv,
    load_demo,
    load_snapshot,
    load_yfinance,
)

SENSITIVITY_PATHS = 1000


@st.cache_data(show_spinner=False, ttl="12h", max_entries=24)
def _load_market(source: str, assets: tuple[str, ...], base: str, csv_prices: pd.DataFrame | None, csv_name: str) -> MarketData:
    if source == "demo":
        return load_demo(list(assets))
    if source == "yfinance":
        return load_yfinance(list(assets), base)
    if source == "snapshot":
        return load_snapshot(list(assets), base)
    if source == "csv":
        if csv_prices is None:
            raise DataUnavailableError("Najpierw zaimportuj plik CSV (Ustawienia danych → Import CSV).")
        return load_csv(csv_prices, list(assets), base, csv_name)
    raise DataUnavailableError(f"Nieznane źródło danych: {source}")


def clear_market_cache() -> None:
    _load_market.clear()


@st.cache_data(show_spinner=False, max_entries=64)
def _moments(est_returns: pd.DataFrame, method: str) -> pf.Moments:
    return pf.estimate_moments(est_returns, method)


@dataclass
class Context:
    scenario: Scenario
    md: MarketData | None = None
    error: str | None = None
    check: DataCheck | None = None
    returns: pd.DataFrame | None = None  # common returns in the analysis period
    est_returns: pd.DataFrame | None = None  # estimation window (last N years of the period)
    moments: pf.Moments | None = None
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    classes: dict[str, str] = field(default_factory=dict)
    currencies: dict[str, str] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.moments is not None

    @property
    def assets(self) -> list[str]:
        return list(self.scenario.portfolio.assets)


def custom_classes() -> dict[str, str] | None:
    ds = st.session_state.get("csv_dataset")
    return ds.get("classes") if ds else None


def get_context(s: Scenario) -> Context:
    ctx = Context(scenario=s)
    assets = s.portfolio.assets
    if not assets:
        ctx.error = "Portfel nie zawiera aktywów. Dodaj instrumenty w Konstruktorze portfela."
        return ctx
    m = s.market
    ds = st.session_state.get("csv_dataset")
    try:
        ctx.md = _load_market(m.data_source, tuple(assets), m.base_currency,
                              ds["prices"] if (m.data_source == "csv" and ds) else None,
                              ds["name"] if (m.data_source == "csv" and ds) else "")
    except DataUnavailableError as exc:
        ctx.error = str(exc)
        return ctx
    ctx.classes = catalog.classes_for(assets, custom_classes())
    ctx.currencies = catalog.currencies_for(assets, m.base_currency)
    ctx.check = check_coverage(ctx.md, assets, m.start_date, m.end_date)
    ctx.returns = ctx.md.common_returns(m.start_date, m.end_date)
    if not ctx.check.ok:
        ctx.error = " ".join(ctx.check.messages)
        return ctx
    ctx.est_returns = ctx.returns.iloc[-m.estimation_years * 12 :]
    method = m.covariance_method
    try:
        ctx.moments = _moments(ctx.est_returns, method)
    except ValueError as exc:
        ctx.error = str(exc)
        return ctx
    ctx.weights = np.asarray(s.portfolio.weights, dtype=float)
    ctx.stats = portfolio_stats(ctx)
    return ctx


def portfolio_stats(ctx: Context) -> dict[str, float]:
    s, mom, w = ctx.scenario, ctx.moments, ctx.weights
    exp = pf.portfolio_return(w, mom.mu)
    vol = pf.portfolio_volatility(w, mom.cov)
    geo = pf.portfolio_geometric_return(w, mom.mu, mom.cov)
    real = float(nominal_to_real(geo, s.profile.inflation))
    real_net = (1 + real) * (1 - s.portfolio.management_fee) - 1
    y = ctx.md.income_yield(ctx.assets, ctx.est_returns.index[0], ctx.est_returns.index[-1])
    income = float(np.dot(w, y.reindex(ctx.assets).fillna(0).to_numpy())) if y is not None else None
    hist = ctx.est_returns.to_numpy() @ w
    return {
        "expected_arithmetic": exp,
        "volatility": vol,
        "geometric_nominal": geo,
        "geometric_real": real,
        "geometric_real_net": real_net,
        "sharpe": pf.sharpe(w, mom.mu, mom.cov, s.market.risk_free_rate),
        "income_yield": income if income is not None else float("nan"),
        "income_available": income is not None,
        "hist_monthly": hist,
    }


def income_yield_or_none(ctx: Context) -> float | None:
    return ctx.stats["income_yield"] if ctx.stats.get("income_available") else None


# ------------------------------------------------------------------ deterministic


def deterministic(ctx: Context):
    return deterministic_projection(ctx.scenario, ctx.stats["geometric_nominal"], income_yield=income_yield_or_none(ctx))


def fire_target(ctx: Context):
    return required_capital(ctx.scenario, ctx.stats.get("geometric_real_net"))


# ------------------------------------------------------------------ portfolio builder


@st.cache_data(show_spinner=False, max_entries=32)
def frontier(mu: np.ndarray, cov: np.ndarray, min_w: float, max_w: float, allow_short: bool, n_points: int = 30):
    return pf.efficient_frontier(mu, cov, min_w, max_w, allow_short, n_points)


@st.cache_data(show_spinner=False, max_entries=32)
def random_cloud(mu: np.ndarray, cov: np.ndarray, min_w: float, max_w: float, n: int = 2500):
    return pf.random_portfolios(mu, cov, n=n, min_w=min_w, max_w=max_w)


# ------------------------------------------------------------------ backtest


@st.cache_data(show_spinner=False, max_entries=16)
def backtests(returns: pd.DataFrame, strategies: tuple[str, ...], classes: tuple[tuple[str, str], ...],
              weights: tuple[float, ...], settings_json: str, rf: float, fee: float, min_w: float, max_w: float,
              allow_short: bool, cov_method: str, benchmark: str | None) -> dict:
    settings = BacktestSettings.model_validate_json(settings_json)
    cls = dict(classes)
    bench = returns[benchmark] if benchmark in returns.columns else None
    out, errors = {}, {}
    for strat in strategies:
        try:
            out[strat] = run_backtest(returns, strat, cls, np.array(weights), settings, rf, fee, min_w, max_w,
                                      allow_short, cov_method, bench)
        except ValueError as exc:
            errors[strat] = str(exc)
    return {"results": out, "errors": errors}


def benchmark_series(ctx: Context, bench_id: str | None) -> pd.Series | None:
    if bench_id and ctx.md is not None and bench_id in ctx.md.returns.columns:
        return ctx.md.returns[bench_id]
    return None


# ------------------------------------------------------------------ Monte Carlo


def return_spec(s: Scenario, n_months: int, n_paths: int | None = None) -> ReturnModelSpec:
    sim = s.simulation
    return ReturnModelSpec(sim.model, n_paths or sim.n_paths, n_months, sim.seed, sim.block_months,
                           sim.fat_tails, sim.t_dof, sim.return_haircut,
                           shrinkage=sim.shrinkage or s.market.covariance_method == "ledoit_wolf")


def run_monte_carlo(ctx: Context, progress: Callable[[float, str], None] | None = None) -> dict:
    """Full life-cycle Monte Carlo. Returns a compact, serialisable payload."""
    s = ctx.scenario
    plan = build_plan(s, ctx.stats["geometric_real_net"])
    spec = return_spec(s, plan.n_months)
    cb = (lambda f: progress(0.85 * f, f"Generowanie ścieżek stóp zwrotu: {int(f * spec.n_paths)} / {spec.n_paths}")) if progress else None
    nominal = simulate_portfolio_returns(ctx.est_returns, ctx.weights, spec, cb)
    if progress:
        progress(0.88, "Symulacja przepływów: akumulacja i dekumulacja")
    real = deflate_returns(nominal, s.profile.inflation)
    ws = s.withdrawal
    income = income_yield_or_none(ctx)
    note = None
    if ws.strategy == WithdrawalStrategy.INCOME_ONLY and income is None:
        ws = ws.model_copy(update={"strategy": WithdrawalStrategy.CONSTANT_REAL})
        note = "Brak danych o dochodzie portfela – zastosowano stałą realną wypłatę."
    res = run_lifecycle(real, plan, ws, income)
    # accumulation-only run: distribution of the age at which the target is reached
    acc_plan = build_plan(s.model_copy(update={"fire": s.fire.model_copy(update={"retirement_rule": RetirementRule.FIXED_AGE})}),
                          ctx.stats["geometric_real_net"])
    acc_plan.fire_month = acc_plan.max_work_month
    acc = run_lifecycle(real, acc_plan, ws.model_copy(update={"strategy": WithdrawalStrategy.CONSTANT_REAL}))
    if progress:
        progress(0.95, "Miary ryzyka: VaR, ES, percentyle")
    summ = summarize(res)
    fh = acc.first_hit_month
    summ["p_fire_by_target_age"] = float(np.mean((fh >= 0) & (fh <= plan.fire_month)))
    summ["median_fire_age"] = float(s.profile.current_age + np.median(fh[fh >= 0]) / 12) if (fh >= 0).any() else float("nan")
    summ["p_fire_ever"] = float(np.mean(fh >= 0))
    h = s.simulation.var_horizon_years * 12
    var = var_es(horizon_returns(nominal, h), s.simulation.var_confidence, s.profile.initial_capital, s.simulation.var_horizon_years)
    ages = res.ages()
    pct = np.percentile(res.wealth_start_year, [5, 25, 50, 75, 95], axis=0)
    fire_ages = np.where(fh >= 0, s.profile.current_age + fh / 12.0, np.nan)
    depl = np.where(res.depletion_month >= 0, s.profile.current_age + res.depletion_month / 12.0, np.nan)
    if progress:
        progress(1.0, "Zakończono")
    return {
        "summary": summ,
        "ages": ages,
        "wealth_percentiles": pct,
        "wealth_at_fire": res.wealth_start_year[:, min(plan.fire_month // 12, res.wealth_start_year.shape[1] - 1)],
        "terminal_wealth": res.terminal_wealth,
        "fire_ages": fire_ages,
        "depletion_ages": depl,
        "var": var,
        "horizon_returns": horizon_returns(nominal, h),
        "nominal_subset": nominal[:SENSITIVITY_PATHS],
        "target": plan.fire_target,
        "fire_month": plan.fire_month,
        "n_paths": spec.n_paths,
        "model": s.simulation.model.value,
        "seed": s.simulation.seed,
        "note": note,
        "sample_paths": res.wealth_start_year[:25],
    }


# ------------------------------------------------------------------ decumulation


def run_decumulation_lab(ctx: Context, start_capital: float, progress: Callable[[float, str], None] | None = None) -> dict:
    s = ctx.scenario
    n_months = (s.profile.life_horizon_age - s.profile.fire_age) * 12
    spec = return_spec(s, n_months)
    cb = (lambda f: progress(0.7 * f, f"Generowanie ścieżek: {int(f * spec.n_paths)} / {spec.n_paths}")) if progress else None
    nominal = simulate_portfolio_returns(ctx.est_returns, ctx.weights, spec, cb)
    real = deflate_returns(nominal, s.profile.inflation)
    income = income_yield_or_none(ctx)
    variants = strategy_variants(s.withdrawal, income is not None)
    out = {}
    for i, (label, ws) in enumerate(variants.items()):
        res = run_decumulation(real, s, start_capital, ws, ctx.stats["geometric_real_net"], income)
        summ = summarize(res)
        depl = np.where(res.depletion_month >= 0, s.profile.fire_age + res.depletion_month / 12.0, np.nan)
        out[label] = {
            "summary": summ,
            "wealth_percentiles": np.percentile(res.wealth_start_year, [5, 25, 50, 75, 95], axis=0),
            "consumption_median": np.nanmedian(res.consumption_year, axis=0) if np.isfinite(res.consumption_year).any() else None,
            "consumption_p10": np.nanpercentile(res.consumption_year, 10, axis=0) if np.isfinite(res.consumption_year).any() else None,
            "depletion_ages": depl,
            "ages": res.ages(),
            "strategy": ws.strategy.value,
        }
        if progress:
            progress(0.7 + 0.3 * (i + 1) / len(variants), f"Strategia: {label}")
    return {"variants": out, "start_capital": start_capital, "required_spending": required_capital(s).monthly_spending,
            "income_available": income is not None, "n_paths": spec.n_paths}


def label_strategy(code: str) -> str:
    return STRATEGY_LABELS.get(code, code)
