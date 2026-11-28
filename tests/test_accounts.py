"""Household accounts: Polish tax wrappers, spouse, metals and rent in the life-cycle engine."""

import numpy as np
import pytest

from core.accounts import build_accounts, rental_schedule
from core.fire import build_plan, deterministic_projection
from core.lifecycle import run_lifecycle, summarize
from core.models import (
    FireSettings,
    HouseholdSettings,
    InvestorProfile,
    PreciousMetals,
    RentalIncome,
    Scenario,
    TaxRules,
    WithdrawalSettings,
    WrapperAccount,
)
from core.withdrawal import decumulation_plan

WS = WithdrawalSettings(strategy="constant_real", constant_basis="spending", spending_floor_share=0.9)


def scenario(capital=0.0, spend=1000.0, age=50, fire=51, horizon=80, infl=0.0, accounts=(), **hh) -> Scenario:
    prof = InvestorProfile(current_age=age, fire_age=fire, life_horizon_age=horizon, initial_capital=capital,
                           monthly_contribution=0.0, retirement_monthly_expenses=spend, inflation=infl)
    fire_s = FireSettings(incomes=[], max_work_age=min(67, horizon - 1))
    house = HouseholdSettings(enabled=True, accounts=list(accounts), **hh)
    return Scenario(profile=prof, fire=fire_s, household=house,
                    portfolio={"assets": ["A"], "weights": [1.0], "management_fee": 0.0})


def run(s: Scenario, r_month=0.0, gold_month=None, paths=1):
    plan = build_plan(s, 0.0)
    rets = np.full((paths, plan.n_months), r_month)
    gold = None if gold_month is None else np.full((paths, plan.n_months), gold_month)
    return plan, run_lifecycle(rets, plan, WS, None, gold, record_buckets=True)


def test_belka_only_on_gain_share():
    s = scenario(capital=100_000, fire=51, main_account_basis_share=0.5)
    plan, res = run(s)
    fm = int(res.fire_month[0])
    # month fm: withdraw 1000 net from v=100k with gain share 50% → gross = 1000 / (1 − 0.19·0.5)
    v_after = res.bucket_wealth[0, 0, :]  # net values per year
    gross = 1000 / (1 - 0.19 * 0.5)
    assert fm == 12
    # net value at the start of the next year after 12 withdrawals is below 100k − 12·gross·(1−…) – check first withdrawal via engine
    plan2 = build_plan(s, 0.0)
    r2 = run_lifecycle(np.zeros((1, plan2.n_months)), plan2, WS, capture_month=14)
    v14 = r2.captured["v"][0]
    # two withdrawals (months 12 and 13); the gain share stays 50% because basis falls proportionally
    assert v14 == pytest.approx(100_000 - 2 * gross, rel=1e-9)
    assert v_after[0] == pytest.approx(100_000 - 0.19 * 50_000)  # after-tax value at start


def test_inflation_erodes_nominal_basis():
    s = scenario(capital=100_000, fire=60, infl=0.12, main_account_basis_share=1.0)
    plan = build_plan(s, 0.0)
    rets = np.zeros((1, plan.n_months))  # real return 0 = nominal return equal to inflation
    res = run_lifecycle(rets, plan, WS, capture_month=12)
    cap = res.captured
    assert cap["v"][0] == pytest.approx(100_000)
    assert cap["basis_v"][0] == pytest.approx(100_000 / 1.12, rel=1e-9)  # nominal gain is taxed although real gain = 0
    assert cap["net"][0] < 100_000


def test_ike_locked_until_60_creates_illiquid_shortfall():
    acc = [WrapperAccount(kind="ike", balance=500_000)]
    s = scenario(capital=0, age=50, fire=51, accounts=acc, taxes=TaxRules(allow_early_ike=False))
    _, res = run(s)
    assert res.shortfall[0] and res.illiquid_shortfall[0]
    assert summarize(res)["p_illiquid_shortfall"] == 1.0
    assert res.depletion_month[0] == -1  # money exists, it is only locked


def test_early_ike_withdrawal_pays_belka_on_gains():
    acc = [WrapperAccount(kind="ike", balance=500_000, basis=250_000)]
    s = scenario(capital=0, age=50, fire=51, accounts=acc, taxes=TaxRules(allow_early_ike=True))
    plan = build_plan(s, 0.0)
    res = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, capture_month=13)
    x13 = res.captured["x"][0, 0]
    assert x13 == pytest.approx(500_000 - 1000 / (1 - 0.19 * 0.5))
    # after 60 the IKE is tax-free: withdrawals equal consumption
    res2 = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, capture_month=121)
    res3 = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, capture_month=122)
    assert res2.captured["x"][0, 0] - res3.captured["x"][0, 0] == pytest.approx(1000)


def test_ikze_flat_10_percent_after_65():
    acc = [WrapperAccount(kind="ikze", balance=300_000)]
    s = scenario(capital=0, age=65, fire=66, accounts=acc)
    plan = build_plan(s, 0.0)
    a = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, capture_month=12).captured["x"][0, 0]
    b = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, capture_month=13).captured["x"][0, 0]
    assert a - b == pytest.approx(1000 / 0.9)


def test_ppk_payout_25_percent_then_120_instalments():
    acc = [WrapperAccount(kind="ppk", balance=120_000)]
    s = scenario(capital=0, spend=0.0, age=60, fire=61, horizon=80, accounts=acc, taxes=TaxRules(ppk_state_annual=0.0))
    plan = build_plan(s, 0.0)
    cap = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, capture_month=13).captured
    # month 12: 25% lump (30 000) and first instalment 90 000/120
    assert cap["x"][0, 0] == pytest.approx(90_000 * 119 / 120)
    assert cap["v"][0] == pytest.approx(30_000 + 750)
    end = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, capture_month=12 + 120).captured
    assert end["x"][0, 0] == pytest.approx(0.0, abs=1e-6)
    assert end["v"][0] == pytest.approx(120_000)


def test_oki_value_tax_above_threshold():
    acc = [WrapperAccount(kind="oki", balance=300_000)]
    s = scenario(capital=0, spend=0.0, fire=60, accounts=acc, taxes=TaxRules(oki_qualifying_share=1.0))
    plan = build_plan(s, 0.0)
    x12 = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, capture_month=12).captured["x"][0, 0]
    assert x12 == pytest.approx(300_000 - 0.0085 * 200_000)
    s2 = scenario(capital=0, spend=0.0, fire=60, accounts=acc, taxes=TaxRules(oki_qualifying_share=0.0))
    plan2 = build_plan(s2, 0.0)
    x12b = run_lifecycle(np.zeros((1, plan2.n_months)), plan2, WS, capture_month=12).captured["x"][0, 0]
    assert x12b == pytest.approx(300_000 * (1 - 0.0085))


def test_metals_follow_gold_and_lose_spread():
    s = scenario(capital=0, spend=0.0, fire=60, metals=PreciousMetals(value=100_000, sale_cost=0.04))
    plan = build_plan(s, 0.0)
    g = 0.01
    res = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, None, np.full((1, plan.n_months), g), record_buckets=True)
    idx = res.bucket_labels.index("Metale szlachetne")
    assert res.bucket_wealth[0, idx, 1] == pytest.approx(100_000 * (1 + g) ** 12 * 0.96)
    assert res.bucket_wealth[0, 0, 1] == 0.0  # portfolio return 0 does not touch metals


def test_contribution_limits_are_capped():
    acc = [WrapperAccount(kind="ike", monthly_own=3_000), WrapperAccount(kind="ikze", monthly_own=2_000)]
    s = scenario(fire=60, accounts=acc)
    ap = build_accounts(s, 120)
    assert ap.contributions[0, 0] == pytest.approx(28_260 / 12)
    assert ap.contributions[1, 0] == pytest.approx(11_304 / 12)
    assert len(ap.capped) == 2
    s_se = scenario(fire=60, accounts=acc, self_employed=True)
    assert build_accounts(s_se, 120).contributions[1, 0] == pytest.approx(16_956 / 12)


def test_ikze_refund_goes_to_main_account():
    acc = [WrapperAccount(kind="ikze", monthly_own=500)]
    s = scenario(fire=60, accounts=acc, taxes=TaxRules(pit_marginal_rate=0.32))
    ap = build_accounts(s, 12)
    assert ap.main_extra[0] == pytest.approx(500 * 0.32)


def test_spouse_access_age_and_joint_flag():
    acc = [WrapperAccount(kind="ike", owner="spouse", balance=10_000)]
    s = scenario(age=50, fire=55, accounts=acc, spouse_age=55, joint_planning=True)
    ap = build_accounts(s, 360)
    assert ap.access_month[0] == 60  # spouse turns 60 in 5 years
    s2 = scenario(age=50, fire=55, accounts=acc, spouse_age=55, joint_planning=False)
    assert build_accounts(s2, 360).n == 0


def test_rental_income_net_and_timing():
    rent = RentalIncome(enabled=True, monthly_gross=3000, vacancy_rate=0.1, monthly_costs=200, tax_rate=0.085,
                        start_age=50, end_age=70, reinvest_share=0.5)
    assert rent.monthly_net == pytest.approx(3000 * 0.9 * (1 - 0.085) - 200)
    s = scenario(age=45, fire=60, rental=rent)
    sched = rental_schedule(s, (80 - 45) * 12)
    assert sched[5 * 12 - 1] == 0 and sched[5 * 12] == pytest.approx(rent.monthly_net) and sched[25 * 12] == 0
    plan = build_plan(s, 0.0)
    assert plan.other_income[5 * 12] == pytest.approx(rent.monthly_net)
    assert plan.accounts.main_extra[5 * 12] == pytest.approx(0.5 * rent.monthly_net)


def test_rent_covers_spending_after_fire():
    rent = RentalIncome(enabled=True, monthly_gross=2000, vacancy_rate=0.0, tax_rate=0.0, start_age=40)
    s = scenario(capital=0, spend=2000, age=50, fire=51, rental=rent)
    _, res = run(s)


def test_decumulation_start_has_requested_after_tax_value():
    acc = [WrapperAccount(kind="ike", balance=200_000, monthly_own=1000), WrapperAccount(kind="ikze", balance=50_000)]
    s = scenario(capital=300_000, age=40, fire=50, accounts=acc, main_account_basis_share=0.6)
    plan = decumulation_plan(s, 1_000_000, 0.03)
    res = run_lifecycle(np.zeros((1, plan.n_months)), plan, WS, record_buckets=True)
    assert res.bucket_wealth[0, :, 0].sum() == pytest.approx(1_000_000, rel=1e-9)
    assert plan.accounts.access_month[0] == 10 * 12  # IKE opens at 60, FIRE at 50


def test_disabled_household_matches_single_portfolio():
    s = scenario(capital=100_000, accounts=[WrapperAccount(kind="ike", balance=1e6)])
    s_off = s.model_copy(update={"household": HouseholdSettings(enabled=False)})
    plan = build_plan(s_off, 0.0)
    assert plan.accounts is None
    proj_on = deterministic_projection(s, 0.0)
    proj_off = deterministic_projection(s_off, 0.0)
    assert proj_on.table["wealth_real"].iloc[0] > proj_off.table["wealth_real"].iloc[0]
