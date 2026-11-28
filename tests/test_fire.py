import numpy as np
import pytest

from core.fire import build_plan, contribution_schedule, deterministic_projection, events_schedule, pv_required_capital, required_capital
from core.lifecycle import run_lifecycle
from core.models import FinancialEvent, FireMode, FireSettings, InvestorProfile, PortfolioConfig, RetirementRule, Scenario, WithdrawalSettings


def scen(**fire_kw) -> Scenario:
    prof = InvestorProfile(current_age=30, fire_age=45, life_horizon_age=90, initial_capital=100_000,
                           monthly_net_income=12_000, monthly_expenses=7_000, monthly_contribution=3_000,
                           retirement_monthly_expenses=6_000, inflation=0.03, income_growth_real=0.0,
                           contribution_growth_real=0.0)
    return Scenario(profile=prof, fire=FireSettings(incomes=[], **fire_kw), portfolio=PortfolioConfig(management_fee=0.0))


def test_traditional_capital():
    t = required_capital(scen(swr=0.04))
    assert t.capital == pytest.approx(6_000 * 12 / 0.04)


def test_modes_have_distinct_logic():
    trad = required_capital(scen()).capital
    lean = required_capital(scen(mode=FireMode.LEAN, lean_essential_share=0.7))
    fat = required_capital(scen(mode=FireMode.FAT, fat_extra_share=0.5, fat_cash_buffer_years=2))
    barista = required_capital(scen(mode=FireMode.BARISTA, barista_monthly_income=2_000))
    coast = required_capital(scen(mode=FireMode.COAST), expected_real_net_return=0.05)
    assert lean.capital == pytest.approx(0.7 * trad)
    assert fat.cash_buffer == pytest.approx(2 * 9_000 * 12)
    assert fat.capital == pytest.approx(9_000 * 12 / 0.04 + 2 * 9_000 * 12)
    assert barista.capital == pytest.approx(4_000 * 12 / 0.04)
    assert coast.capital == pytest.approx(trad)
    assert coast.coast_number_today == pytest.approx(trad / 1.05**15)


def test_contribution_compounding_zero_return():
    s = scen()
    plan = build_plan(s, 0.0)
    plan.events[:] = 0
    res = run_lifecycle(np.zeros((1, plan.n_months)), plan, s.withdrawal)
    # after 1 year with zero return: initial + 12 contributions
    assert res.wealth_start_year[0, 1] == pytest.approx(100_000 + 12 * 3_000)


def test_contribution_compounding_positive_return_annuity_due():
    s = scen()
    plan = build_plan(s, 0.0)
    r = 0.005
    res = run_lifecycle(np.full((1, plan.n_months), r), plan, s.withdrawal)
    n = 12
    fv = 100_000 * (1 + r) ** n + 3_000 * ((1 + r) ** n - 1) / r * (1 + r)  # payments at month start
    assert res.wealth_start_year[0, 1] == pytest.approx(fv)


def test_negative_return_shrinks_capital():
    s = scen()
    plan = build_plan(s, 0.0)
    res = run_lifecycle(np.full((1, plan.n_months), -0.01), plan, s.withdrawal)
    assert res.wealth_start_year[0, 1] < 100_000 + 12 * 3_000


def test_contribution_capped_by_savings_capacity():
    s = scen()
    s.profile.monthly_contribution = 9_000  # capacity is 12k - 7k = 5k
    c, capped = contribution_schedule(s, 24)
    assert c.max() == pytest.approx(5_000) and capped.all()


def test_events_one_off_and_recurring():
    s = scen()
    s.profile.events = [
        FinancialEvent(name="Spadek", category="inflow", age=35, amount=50_000),
        FinancialEvent(name="Czesne", category="outflow", age=40, amount=10_000, recurring=True, end_age=42),
    ]
    ev = events_schedule(s, 60 * 12)
    assert ev[(35 - 30) * 12] == 50_000
    assert ev[(40 - 30) * 12] == -10_000 and ev[(42 - 30) * 12] == -10_000 and ev[(43 - 30) * 12] == 0


def test_deterministic_projection_and_fire_age():
    s = scen()
    proj = deterministic_projection(s, expected_nominal_geometric=0.07)
    assert proj.fire_age is not None and 30 < proj.fire_age < 67
    assert proj.real_return == pytest.approx(1.07 / 1.03 - 1)
    # nominal = real * inflation index
    row = proj.table.iloc[10]
    assert row.wealth_nominal == pytest.approx(row.wealth_real * 1.03**10)


def test_real_vs_nominal_no_double_inflation():
    """Zero real return → real wealth with no flows stays flat although nominal grows by inflation."""
    s = scen()
    s.profile.monthly_contribution = 0
    s.profile.initial_capital = 1_000_000
    proj = deterministic_projection(s, expected_nominal_geometric=0.03)  # = inflation
    assert proj.table.wealth_real.iloc[5] == pytest.approx(1_000_000, rel=1e-9)
    assert proj.table.wealth_nominal.iloc[5] == pytest.approx(1_000_000 * 1.03**5, rel=1e-9)


def test_coast_stops_contributions():
    s = scen(mode=FireMode.COAST)
    s.profile.initial_capital = 2_000_000  # above coast number
    plan = build_plan(s, 0.03)
    res = run_lifecycle(np.full((1, plan.n_months), 0.0025), plan, s.withdrawal)
    assert res.contribution_year[0].sum() == 0
    assert res.coast_month[0] == 0


def test_dynamic_retirement_when_target_reached():
    s = scen(retirement_rule=RetirementRule.WHEN_TARGET_REACHED, max_work_age=67)
    s.profile.initial_capital = 2_000_000  # already above 1.8m target
    plan = build_plan(s, 0.03)
    res = run_lifecycle(np.zeros((1, plan.n_months)), plan, s.withdrawal)
    assert res.fire_month[0] == 0


def test_pv_capital_positive_and_responds_to_return():
    s = scen()
    assert pv_required_capital(s, 0.04) < pv_required_capital(s, 0.0)
    assert pv_required_capital(s, 0.0) == pytest.approx(6_000 * 12 * 45)
