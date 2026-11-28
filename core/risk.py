"""Risk measures, stress testing and sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.fire import build_plan
from core.lifecycle import deflate_returns, run_lifecycle, summarize
from core.models import RetirementRule, Scenario

PERCENTILES = (5, 25, 50, 75, 95)


@dataclass
class VarResult:
    confidence: float
    horizon_years: int
    var: float  # loss as positive fraction of initial value
    es: float
    var_amount: float
    es_amount: float


def var_es(horizon_returns: np.ndarray, confidence: float, value: float, horizon_years: int) -> VarResult:
    """Historical-simulation style VaR/ES on the simulated distribution.

    VaR_α = −q_{1−α}(R_H);  ES_α = −E[R_H | R_H ≤ q_{1−α}(R_H)].
    R_H is the cumulative portfolio return over H years **without** cash flows.
    """
    r = np.asarray(horizon_returns, dtype=float)
    q = np.quantile(r, 1.0 - confidence)
    tail = r[r <= q]
    es = -float(tail.mean()) if tail.size else -float(q)
    var = -float(q)
    return VarResult(confidence, horizon_years, var, es, var * value, es * value)


def percentile_table(values: np.ndarray, axis: int = 0) -> dict[int, np.ndarray]:
    return {p: np.percentile(values, p, axis=axis) for p in PERCENTILES}


# ------------------------------------------------------------------ stress scenarios

HYPOTHETICAL_SCENARIOS: dict[str, dict] = {
    "Globalny krach akcyjny": {
        "months": 18,
        "shocks": {"global_equity": -0.45, "us_equity": -0.45, "europe_equity": -0.50, "em_equity": -0.55,
                   "bonds": 0.08, "gold": 0.15, "cash": 0.02},
        "description": "Spadek akcji o 45–55% w ciągu 18 miesięcy, ucieczka do obligacji i złota.",
    },
    "Stagflacja": {
        "months": 24,
        "shocks": {"global_equity": -0.25, "us_equity": -0.25, "europe_equity": -0.28, "em_equity": -0.30,
                   "bonds": -0.18, "gold": 0.25, "cash": 0.04},
        "description": "Równoczesne spadki akcji i obligacji przy wysokiej inflacji (jak w 2022 r., ale dłużej).",
    },
    "Umocnienie PLN o 20%": {
        "months": 12,
        "shocks": {"global_equity": -0.20, "us_equity": -0.20, "europe_equity": -0.20, "em_equity": -0.20,
                   "bonds": -0.20, "gold": -0.20, "cash": -0.20},
        "foreign_only": True,
        "description": "Spadek wartości aktywów zagranicznych w PLN wyłącznie z powodu kursu walutowego.",
    },
    "Dekada stracona": {
        "months": 120,
        "shocks": {"global_equity": -0.10, "us_equity": -0.10, "europe_equity": -0.15, "em_equity": -0.10,
                   "bonds": 0.20, "gold": 0.30, "cash": 0.15},
        "description": "Dziesięć lat stagnacji rynku akcji (por. 2000–2009).",
    },
}

HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "Bańka internetowa 2000–2002": ("2000-04-01", "2002-09-30"),
    "Globalny kryzys finansowy 2007–2009": ("2007-11-01", "2009-02-28"),
    "Kryzys strefy euro 2011": ("2011-05-01", "2011-09-30"),
    "Pandemia COVID-19 2020": ("2020-02-01", "2020-03-31"),
    "Szok inflacyjny 2022": ("2022-01-01", "2022-10-31"),
}


def hypothetical_stress_path(scenario: dict, assets: list[str], classes: dict[str, str], currencies: dict[str, str],
                             weights: np.ndarray, base_currency: str = "PLN") -> np.ndarray:
    months = scenario["months"]
    asset_paths = np.zeros((months, len(assets)))
    for i, a in enumerate(assets):
        shock = scenario["shocks"].get(classes.get(a, ""), 0.0)
        if scenario.get("foreign_only") and currencies.get(a, base_currency) == base_currency:
            shock = 0.0
        asset_paths[:, i] = (1.0 + shock) ** (1.0 / months) - 1.0
    return asset_paths @ np.asarray(weights, dtype=float)


def historical_stress_path(asset_returns: pd.DataFrame, start: str, end: str, weights: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Portfolio monthly returns during a historical window, only if every asset has data there."""
    window = asset_returns.loc[start:end]
    expected_months = len(pd.period_range(start, end, freq="M"))
    if len(window) < expected_months or window.isna().any().any():
        missing = [c for c in asset_returns.columns if asset_returns[c].loc[start:end].dropna().size < expected_months]
        return None, "Brak pełnych danych dla: " + ", ".join(missing) if missing else "Brak danych w tym okresie."
    return window.to_numpy(dtype=float) @ np.asarray(weights, dtype=float), "ok"


def apply_stress(nominal_returns: np.ndarray, stress_path: np.ndarray, start_month: int) -> np.ndarray:
    out = nominal_returns.copy()
    end = min(start_month + stress_path.size, out.shape[1])
    out[:, start_month:end] = stress_path[: end - start_month]
    return out


# ------------------------------------------------------------------ sensitivity


def _p(nominal, s: Scenario, real_net, income_yield, gold_nominal=None):
    plan = build_plan(s, real_net)
    real = deflate_returns(nominal[:, : plan.n_months], s.profile.inflation)
    ws = s.withdrawal
    if ws.strategy == "income_only" and income_yield is None:
        ws = ws.model_copy(update={"strategy": "constant_real"})
    gold = deflate_returns(gold_nominal[:, : plan.n_months], s.profile.inflation) if gold_nominal is not None else None
    return summarize(run_lifecycle(real, plan, ws, income_yield, gold))


def sensitivity_analysis(nominal_returns: np.ndarray, s: Scenario, real_net: float, income_yield: float | None = None,
                         metric: str = "p_success", gold_nominal: np.ndarray | None = None) -> pd.DataFrame:
    """One-at-a-time sensitivity with common random numbers (same paths for every variant)."""
    base = _p(nominal_returns, s, real_net, income_yield, gold_nominal)[metric]
    prof = s.profile

    def with_profile(**kw):
        return s.model_copy(update={"profile": prof.model_copy(update=kw)})

    def with_fire(**kw):
        return s.model_copy(update={"fire": s.fire.model_copy(update=kw)})

    def with_port(**kw):
        return s.model_copy(update={"portfolio": s.portfolio.model_copy(update=kw)})

    variants = [
        ("Wydatki po FIRE ±10%", with_profile(retirement_monthly_expenses=prof.retirement_monthly_expenses * 0.9),
         with_profile(retirement_monthly_expenses=prof.retirement_monthly_expenses * 1.1), None),
        ("Miesięczna wpłata ±20%", with_profile(monthly_contribution=prof.monthly_contribution * 1.2),
         with_profile(monthly_contribution=prof.monthly_contribution * 0.8), None),
        ("Inflacja ∓1 p.p.", with_profile(inflation=prof.inflation - 0.01), with_profile(inflation=prof.inflation + 0.01), None),
        ("Koszty portfela ∓0,5 p.p.", with_port(management_fee=max(s.portfolio.management_fee - 0.005, 0.0)),
         with_port(management_fee=s.portfolio.management_fee + 0.005), None),
        ("Stopa zwrotu ±1 p.p. rocznie", s, s, 0.01),
    ]
    if prof.fire_age + 2 < prof.life_horizon_age and prof.fire_age - 2 > prof.current_age:
        variants.append(("Wiek FIRE ±2 lata", with_profile(fire_age=prof.fire_age + 2), with_profile(fire_age=prof.fire_age - 2), None))
    if s.fire.swr - 0.005 > 0:
        variants.append(("SWR (cel) ∓0,5 p.p.", with_fire(swr=s.fire.swr - 0.005), with_fire(swr=s.fire.swr + 0.005), None))

    rows = []
    for name, better, worse, shift in variants:
        if shift is not None:
            up = np.expm1(np.log1p(nominal_returns) + shift / 12)
            dn = np.expm1(np.log1p(nominal_returns) - shift / 12)
            hi = _p(up, s, real_net + shift, income_yield, gold_nominal)[metric]
            lo = _p(dn, s, real_net - shift, income_yield, gold_nominal)[metric]
        else:
            try:
                hi = _p(nominal_returns, better, real_net, income_yield, gold_nominal)[metric]
                lo = _p(nominal_returns, worse, real_net, income_yield, gold_nominal)[metric]
            except ValueError:
                continue
        rows.append({"parametr": name, "korzystny": hi, "niekorzystny": lo, "bazowy": base})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rozpiętość"] = (df["korzystny"] - df["niekorzystny"]).abs()
        df = df.sort_values("rozpiętość")
    return df


def success_heatmap(nominal_returns: np.ndarray, s: Scenario, real_net: float, fire_ages: list[int],
                    spending_multipliers: list[float], income_yield: float | None = None,
                    gold_nominal: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """P(success) and P(not depleted) on a grid FIRE age × post-FIRE spending (fixed retirement age)."""
    succ = np.full((len(spending_multipliers), len(fire_ages)), np.nan)
    keep = np.full_like(succ, np.nan)
    for j, age in enumerate(fire_ages):
        for i, mult in enumerate(spending_multipliers):
            try:
                variant = s.model_copy(update={
                    "profile": s.profile.model_copy(update={
                        "fire_age": age,
                        "retirement_monthly_expenses": s.profile.retirement_monthly_expenses * mult}),
                    "fire": s.fire.model_copy(update={"retirement_rule": RetirementRule.FIXED_AGE,
                                                      "max_work_age": max(s.fire.max_work_age, age)}),
                })
                out = _p(nominal_returns, variant, real_net, income_yield, gold_nominal)
            except ValueError:
                continue
            succ[i, j] = out["p_success"]
            keep[i, j] = out["p_not_depleted"]
    return {"p_success": succ, "p_not_depleted": keep}
