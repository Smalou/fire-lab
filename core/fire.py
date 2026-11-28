"""FIRE targets (five modes with distinct logic) and deterministic projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.accounts import build_accounts, rental_schedule
from core.lifecycle import LifecyclePlan, LifecycleResult, run_lifecycle
from core.models import FireMode, RetirementRule, Scenario, WithdrawalSettings, WithdrawalStrategy
from core.returns import annual_to_monthly, nominal_to_real

MODE_LABELS = {
    FireMode.TRADITIONAL: "Traditional FIRE",
    FireMode.LEAN: "Lean FIRE",
    FireMode.FAT: "Fat FIRE",
    FireMode.COAST: "Coast FIRE",
    FireMode.BARISTA: "Barista FIRE",
}


@dataclass
class FireTarget:
    mode: FireMode
    capital: float  # real capital required at FIRE
    monthly_spending: float  # real required spending after FIRE
    annual_need_from_portfolio: float  # what the SWR is applied to
    cash_buffer: float = 0.0
    coast_number_today: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


def required_capital(s: Scenario, expected_real_net_return: float | None = None) -> FireTarget:
    """Capital required for financial independence, mode-specific.

    * Traditional: C = E / SWR
    * Lean:        C = (share · E) / SWR – only essential spending is financed
    * Fat:         C = E(1+m) / SWR + k · E(1+m) – higher spending plus k-year cash reserve
    * Coast:       same C as Traditional; additionally the Coast number
                   C / (1+g)^(T−t) after which contributions may stop
    * Barista:     C = max(E − 12·I_pt, 0) / SWR – part-time income covers part of spending
    """
    prof, f = s.profile, s.fire
    e_month = prof.retirement_monthly_expenses
    mode = f.mode
    buffer = 0.0
    coast_today = None
    comps: dict[str, float] = {}
    if mode == FireMode.LEAN:
        spending = e_month * f.lean_essential_share
        need = 12 * spending
        capital = need / f.swr
        expl = "Kapitał pokrywa wyłącznie wydatki podstawowe (udział wydatków niezbędnych)."
    elif mode == FireMode.FAT:
        spending = e_month * (1 + f.fat_extra_share)
        need = 12 * spending
        buffer = f.fat_cash_buffer_years * need
        capital = need / f.swr + buffer
        comps["Bufor gotówkowy"] = buffer
        expl = "Wyższy standard wydatków oraz rezerwa płynnościowa na lata złej koniunktury."
    elif mode == FireMode.BARISTA:
        spending = e_month
        need = max(12 * (e_month - f.barista_monthly_income), 0.0)
        capital = need / f.swr
        comps["Pokrywane pracą częściową (rocznie)"] = 12 * min(f.barista_monthly_income, e_month)
        expl = "Część wydatków finansuje praca w niepełnym wymiarze do wskazanego wieku."
    else:  # TRADITIONAL, COAST
        spending = e_month
        need = 12 * spending
        capital = need / f.swr
        expl = "Kapitał = roczne wydatki / bezpieczna stopa wypłat (SWR)."
    if mode == FireMode.COAST:
        years = prof.fire_age - prof.current_age
        g = expected_real_net_return if expected_real_net_return is not None else 0.0
        coast_today = capital / (1 + g) ** years if g > -1 else capital
        comps["Kwota Coast FIRE dziś"] = coast_today
        expl = "Po zgromadzeniu kwoty Coast wpłaty ustają, a portfel ma sam urosnąć do celu do wieku FIRE."
    comps["Kapitał z reguły SWR"] = need / f.swr
    return FireTarget(mode, capital, spending, need, buffer, coast_today, comps, expl)


def contribution_schedule(s: Scenario, n_months: int) -> tuple[np.ndarray, np.ndarray]:
    """Real monthly contributions with real indexation, capped by savings capacity.

    Returns (contributions, capped_flag) arrays of length n_months.
    """
    prof = s.profile
    years = np.arange(n_months) // 12
    planned = prof.monthly_contribution * (1 + prof.contribution_growth_real) ** years
    capacity = prof.monthly_net_income * (1 + prof.income_growth_real) ** years - prof.monthly_expenses
    capacity = np.maximum(capacity, 0.0)
    contrib = np.minimum(planned, capacity)
    return contrib, planned > capacity + 1e-9


def events_schedule(s: Scenario, n_months: int) -> np.ndarray:
    ev = np.zeros(n_months)
    age0 = s.profile.current_age
    for e in s.profile.events:
        if not e.enabled or e.amount == 0:
            continue
        ages = range(e.age, (e.end_age if e.end_age is not None else s.profile.life_horizon_age - 1) + 1) if e.recurring else [e.age]
        for a in ages:
            m = (a - age0) * 12
            if 0 <= m < n_months:
                ev[m] += e.signed_amount
    return ev


def income_schedule(s: Scenario, n_months: int) -> tuple[np.ndarray, np.ndarray]:
    """(other_income, barista_income) real monthly arrays; counted only while retired."""
    ages = s.profile.current_age + np.arange(n_months) / 12.0
    other = np.zeros(n_months)
    for inc in s.fire.incomes:
        if not inc.enabled:
            continue
        end = inc.end_age if inc.end_age is not None else 10_000
        other += np.where((ages >= inc.start_age) & (ages < end), inc.monthly_amount, 0.0)
    barista = np.zeros(n_months)
    if s.fire.mode == FireMode.BARISTA:
        barista = np.where(ages < s.fire.barista_end_age, s.fire.barista_monthly_income, 0.0)
    return other, barista


def build_plan(s: Scenario, expected_real_net_return: float, cash_nominal_rate: float | None = None) -> LifecyclePlan:
    prof = s.profile
    n_months = (prof.life_horizon_age - prof.current_age) * 12
    target = required_capital(s, expected_real_net_return)
    contrib, _ = contribution_schedule(s, n_months)
    other, barista = income_schedule(s, n_months)
    other = other + rental_schedule(s, n_months)  # rent counts as income once retired
    fire_month = (prof.fire_age - prof.current_age) * 12
    max_work_month = (max(s.fire.max_work_age, prof.fire_age) - prof.current_age) * 12
    coast = None
    if s.fire.mode == FireMode.COAST:
        months_left = np.maximum(fire_month - np.arange(n_months), 0)
        g_m = (1 + expected_real_net_return) ** (1 / 12) - 1 if expected_real_net_return > -1 else 0.0
        coast = target.capital / (1 + g_m) ** months_left
    rf = s.market.risk_free_rate if cash_nominal_rate is None else cash_nominal_rate
    cash_real_m = float(annual_to_monthly(nominal_to_real(rf, prof.inflation)))
    return LifecyclePlan(
        current_age=prof.current_age,
        n_months=n_months,
        initial_capital=prof.initial_capital,
        contributions=contrib,
        events=events_schedule(s, n_months),
        spending=target.monthly_spending,
        other_income=other,
        barista_income=barista,
        fire_target=target.capital,
        retirement_rule=s.fire.retirement_rule,
        fire_month=fire_month,
        max_work_month=max_work_month,
        coast_targets=coast,
        cash_buffer=target.cash_buffer,
        cash_real_monthly=cash_real_m,
        fee_monthly=float(annual_to_monthly(s.portfolio.management_fee)),
        inflation_annual=prof.inflation,
        accounts=build_accounts(s, n_months),
    )


def pv_required_capital(s: Scenario, real_return: float) -> float:
    """Finite-horizon alternative: PV at FIRE of real spending gaps (spending − income) until horizon."""
    prof = s.profile
    target = required_capital(s, real_return)
    n = (prof.life_horizon_age - prof.current_age) * 12
    other, barista = income_schedule(s, n)
    start = (prof.fire_age - prof.current_age) * 12
    gaps = np.maximum(target.monthly_spending - other[start:] - barista[start:], 0.0)
    r_m = (1 + real_return) ** (1 / 12) - 1
    disc = (1 + r_m) ** -np.arange(gaps.size)  # payments at the start of each month
    return float(np.sum(gaps * disc) + target.cash_buffer)


@dataclass
class DeterministicProjection:
    table: pd.DataFrame  # yearly: age, wealth_real, wealth_nominal, contributions, withdrawals, phase
    fire_age: float | None  # age at which real wealth first reaches the target (accumulation run)
    target: FireTarget
    result: LifecycleResult
    real_return: float
    capped_contributions: bool


def deterministic_projection(
    s: Scenario,
    expected_nominal_geometric: float,
    ws: WithdrawalSettings | None = None,
    income_yield: float | None = None,
    gold_nominal_geometric: float | None = None,
) -> DeterministicProjection:
    """Single path with constant (geometric, i.e. median) return; fees applied by the engine."""
    real = float(nominal_to_real(expected_nominal_geometric, s.profile.inflation))
    fee = s.portfolio.management_fee
    real_net = (1 + real) * (1 - fee) - 1
    plan = build_plan(s, real_net)
    r_m = float(annual_to_monthly(real))
    rets = np.full((1, plan.n_months), r_m)
    gold = None
    if gold_nominal_geometric is not None:
        gold = np.full((1, plan.n_months), float(annual_to_monthly(nominal_to_real(gold_nominal_geometric, s.profile.inflation))))
    elif plan.accounts is not None and plan.accounts.uses_gold.any():
        gold = np.full((1, plan.n_months), float(annual_to_monthly(s.household.metals.manual_real_return)))
    ws = ws or s.withdrawal
    try:
        res = run_lifecycle(rets, plan, ws, income_yield, gold, record_buckets=True)
    except ValueError:
        ws = ws.model_copy(update={"strategy": WithdrawalStrategy.CONSTANT_REAL})
        res = run_lifecycle(rets, plan, ws, income_yield, gold, record_buckets=True)
    # FIRE age under "keep accumulating until max work age" – independent of retirement rule
    acc_plan = LifecyclePlan(**{**plan.__dict__, "retirement_rule": RetirementRule.FIXED_AGE, "fire_month": plan.max_work_month})
    acc = run_lifecycle(rets, acc_plan, ws.model_copy(update={"strategy": WithdrawalStrategy.CONSTANT_REAL}), None, gold)
    hit = int(acc.first_hit_month[0])
    fire_age = s.profile.current_age + hit / 12.0 if hit >= 0 else None
    ages = res.ages()
    infl = (1 + s.profile.inflation) ** (ages - s.profile.current_age)
    fm = int(res.fire_month[0])
    phase = np.where((ages - s.profile.current_age) * 12 < (fm if fm >= 0 else 10**9), "Akumulacja", "Dekumulacja")
    table = pd.DataFrame(
        {
            "age": ages,
            "wealth_real": res.wealth_start_year[0],
            "wealth_nominal": res.wealth_start_year[0] * infl,
            "contributions": np.append(res.contribution_year[0], np.nan),
            "withdrawals": np.append(res.withdrawal_year[0], np.nan),
            "target_real": plan.fire_target,
            "phase": phase,
        }
    )
    _, capped = contribution_schedule(s, plan.n_months)
    return DeterministicProjection(table, fire_age, required_capital(s, real_net), res, real, bool(capped[: plan.fire_month].any()))
