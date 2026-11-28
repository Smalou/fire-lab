import numpy as np
import pytest

from core.lifecycle import max_consumption_cut, run_lifecycle, summarize
from core.models import FireSettings, IncomeSource, InvestorProfile, PortfolioConfig, Scenario, WithdrawalSettings, WithdrawalStrategy
from core.withdrawal import decumulation_plan, run_decumulation, strategy_variants


def scen(incomes=None) -> Scenario:
    prof = InvestorProfile(current_age=40, fire_age=50, life_horizon_age=80, retirement_monthly_expenses=5_000,
                           inflation=0.0, monthly_contribution=0)
    return Scenario(profile=prof, fire=FireSettings(incomes=incomes or []), portfolio=PortfolioConfig(management_fee=0.0))


def test_constant_real_withdrawal_amount_zero_return():
    s = scen()
    ws = WithdrawalSettings(strategy=WithdrawalStrategy.CONSTANT_REAL, initial_rate=0.04)
    res = run_decumulation(np.zeros((1, 360)), s, 1_500_000, ws, 0.0)
    assert res.withdrawal_year[0, 0] == pytest.approx(0.04 * 1_500_000)
    assert res.withdrawal_year[0, 5] == pytest.approx(0.04 * 1_500_000)
    assert res.wealth_start_year[0, 1] == pytest.approx(1_500_000 - 60_000)


def test_depletion_detected_at_right_time():
    s = scen()
    ws = WithdrawalSettings(strategy=WithdrawalStrategy.CONSTANT_REAL, initial_rate=0.10)
    res = run_decumulation(np.zeros((1, 360)), s, 600_000, ws, 0.0)
    # 60k per year, 600k capital, zero return → empty after exactly 10 years (month 119)
    assert res.depletion_month[0] == 119
    out = summarize(res)
    assert out["p_not_depleted"] == 0.0


def test_constant_percent_never_depletes_but_may_not_cover():
    s = scen()
    ws = WithdrawalSettings(strategy=WithdrawalStrategy.CONSTANT_PERCENT, percent_rate=0.05, spending_floor_share=0.9)
    rets = np.full((1, 360), -0.01)
    res = run_decumulation(rets, s, 1_200_000, ws, 0.0)
    assert res.depletion_month[0] == -1
    assert res.shortfall[0]  # spending of 5k not covered once portfolio shrinks
    assert summarize(res)["p_success"] == 0.0


def test_pension_reduces_withdrawals():
    s = scen(incomes=[IncomeSource(name="Emerytura", monthly_amount=2_000, start_age=60)])
    ws = WithdrawalSettings(strategy=WithdrawalStrategy.CONSTANT_REAL, constant_basis="spending")
    res = run_decumulation(np.zeros((1, 360)), s, 1_500_000, ws, 0.0)
    assert res.withdrawal_year[0, 0] == pytest.approx(5_000 * 12)
    assert res.withdrawal_year[0, 11] == pytest.approx(3_000 * 12)


def test_tax_gross_up():
    s = scen()
    ws = WithdrawalSettings(strategy=WithdrawalStrategy.CONSTANT_REAL, constant_basis="spending", withdrawal_tax_rate=0.2)
    res = run_decumulation(np.zeros((1, 360)), s, 1_500_000, ws, 0.0)
    assert res.wealth_start_year[0, 1] == pytest.approx(1_500_000 - 60_000 / 0.8)


def test_guardrails_cut_after_crash():
    s = scen()
    ws = WithdrawalSettings(strategy=WithdrawalStrategy.GUARDRAILS, initial_rate=0.05)
    rets = np.zeros((1, 360))
    rets[0, :12] = (0.6) ** (1 / 12) - 1  # -40% in first year
    res = run_decumulation(rets, s, 1_000_000, ws, 0.0)
    assert res.withdrawal_year[0, 1] < res.withdrawal_year[0, 0]
    assert max_consumption_cut(res.consumption_year)[0] > 0


def test_income_only_requires_yield():
    s = scen()
    ws = WithdrawalSettings(strategy=WithdrawalStrategy.INCOME_ONLY)
    with pytest.raises(ValueError):
        run_decumulation(np.zeros((1, 360)), s, 1_000_000, ws, 0.0, income_yield=None)
    res = run_decumulation(np.zeros((1, 360)), s, 1_000_000, ws, 0.0, income_yield=0.03)
    assert res.withdrawal_year[0, 0] == pytest.approx(30_000)


def test_variants_present_4pct_as_constant_real():
    v = strategy_variants(WithdrawalSettings(initial_rate=0.04), income_available=False)
    rule = v["Reguła 4% (stała realna, 4%)"]
    assert rule.strategy == WithdrawalStrategy.CONSTANT_REAL and rule.initial_rate == 0.04
    assert not any("dochód" in k for k in v)


def test_plan_starts_at_fire_age():
    s = scen()
    plan = decumulation_plan(s, 1_000_000, 0.0)
    assert plan.current_age == 50 and plan.n_months == 360 and plan.contributions.sum() == 0
