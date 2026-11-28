"""Decumulation (post-FIRE) experiments built on the shared life-cycle engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.fire import build_plan
from core.lifecycle import LifecyclePlan, LifecycleResult, run_lifecycle
from core.models import RetirementRule, Scenario, WithdrawalSettings, WithdrawalStrategy

STRATEGY_LABELS = {
    WithdrawalStrategy.CONSTANT_REAL: "Stała realna wypłata",
    WithdrawalStrategy.CONSTANT_PERCENT: "Stały % bieżącej wartości",
    WithdrawalStrategy.GUARDRAILS: "Guardrails (Guyton–Klinger, uproszczone)",
    WithdrawalStrategy.INCOME_ONLY: "Tylko dochód portfela (dywidendy/kupony)",
}

STRATEGY_HELP = {
    WithdrawalStrategy.CONSTANT_REAL: "Kwota ustalana w chwili FIRE (np. 4% kapitału) i utrzymywana w stałej sile nabywczej. "
    "Reguła 4% to szczególny przypadek tej metody ze stopą początkową 4%.",
    WithdrawalStrategy.CONSTANT_PERCENT: "Co roku wypłacany jest stały procent bieżącej wartości portfela – portfel nie wyczerpie się, "
    "ale wydatki wahają się razem z rynkiem.",
    WithdrawalStrategy.GUARDRAILS: "Realna wypłata jest utrzymywana, dopóki bieżąca stopa wypłat mieści się w korytarzu "
    "wokół stopy początkowej; po przekroczeniu górnej bariery wypłata jest cięta, po spadku poniżej dolnej – podnoszona. "
    "Po roku ze stratą wypłata nie jest indeksowana inflacją.",
    WithdrawalStrategy.INCOME_ONLY: "Wypłacane są wyłącznie dywidendy i kupony (stopa dochodu z rzeczywistych danych o wypłatach). "
    "Kapitał nominalnie nie jest konsumowany.",
}


def decumulation_plan(s: Scenario, start_capital: float, expected_real_net_return: float,
                      gold_real_annual: float | None = None) -> LifecyclePlan:
    """Plan that starts at the FIRE age with a given capital and no contributions.

    With household accounts the composition at FIRE (main portfolio, IKE,
    IKZE, PPK, …, metals, and their cost bases) comes from the deterministic
    accumulation path and is rescaled so that its after-tax value equals
    ``start_capital``; access ages keep their position relative to FIRE.
    """
    full = build_plan(s, expected_real_net_return)
    offset = (s.profile.fire_age - s.profile.current_age) * 12
    n = full.n_months - offset
    initial, accounts = start_capital, None
    if full.accounts is not None:
        fee = s.portfolio.management_fee
        r_m = (1 + ((1 + expected_real_net_return) / (1 - fee) - 1)) ** (1 / 12) - 1
        g_ann = gold_real_annual if gold_real_annual is not None else s.household.metals.manual_real_return
        g_m = (1 + g_ann) ** (1 / 12) - 1
        cap = run_lifecycle(np.full((1, full.n_months), r_m), replace(full, retirement_rule=RetirementRule.FIXED_AGE), s.withdrawal, None,
                            np.full((1, full.n_months), g_m), capture_month=offset).captured
        ap = full.accounts
        scale = start_capital / cap["net"][0] if cap["net"][0] > 0 else 0.0
        initial = float(cap["v"][0] * scale)
        accounts = replace(
            ap,
            initial=cap["x"][0] * scale if ap.n else ap.initial,
            basis=cap["bx"][0] * scale if ap.n else ap.basis,
            contributions=np.zeros((ap.n, n)),
            access_month=np.maximum(ap.access_month - offset, 0),
            main_basis=float(cap["basis_v"][0] * scale),
            main_extra=np.zeros(n),
        )
    return replace(
        full,
        current_age=s.profile.fire_age,
        n_months=n,
        initial_capital=initial,
        contributions=np.zeros(n),
        events=full.events[offset:],
        other_income=full.other_income[offset:],
        barista_income=full.barista_income[offset:],
        retirement_rule=RetirementRule.FIXED_AGE,
        fire_month=0,
        max_work_month=0,
        coast_targets=None,
        accounts=accounts,
    )


def run_decumulation(
    real_returns: np.ndarray,
    s: Scenario,
    start_capital: float,
    ws: WithdrawalSettings,
    expected_real_net_return: float,
    income_yield: float | None = None,
    gold_real_returns: np.ndarray | None = None,
    gold_real_annual: float | None = None,
) -> LifecycleResult:
    plan = decumulation_plan(s, start_capital, expected_real_net_return, gold_real_annual)
    gold = gold_real_returns[:, : plan.n_months] if gold_real_returns is not None else None
    return run_lifecycle(real_returns[:, : plan.n_months], plan, ws, income_yield, gold)


def strategy_variants(base: WithdrawalSettings, income_available: bool) -> dict[str, WithdrawalSettings]:
    """Comparable set of strategies sharing the base parameters (same random paths = CRN)."""
    out = {
        "Reguła 4% (stała realna, 4%)": base.model_copy(update={"strategy": WithdrawalStrategy.CONSTANT_REAL, "constant_basis": "rate", "initial_rate": 0.04}),
        f"Stała realna ({base.initial_rate:.1%})".replace(".", ","): base.model_copy(update={"strategy": WithdrawalStrategy.CONSTANT_REAL, "constant_basis": "rate"}),
        "Stała realna = wymagane wydatki": base.model_copy(update={"strategy": WithdrawalStrategy.CONSTANT_REAL, "constant_basis": "spending"}),
        f"Stały % portfela ({base.percent_rate:.1%})".replace(".", ","): base.model_copy(update={"strategy": WithdrawalStrategy.CONSTANT_PERCENT}),
        "Guardrails": base.model_copy(update={"strategy": WithdrawalStrategy.GUARDRAILS}),
    }
    if abs(base.initial_rate - 0.04) < 1e-12:  # identical to the 4% preset – avoid a duplicate line
        out.pop(f"Stała realna ({base.initial_rate:.1%})".replace(".", ","))
    if income_available:
        out["Tylko dochód portfela"] = base.model_copy(update={"strategy": WithdrawalStrategy.INCOME_ONLY})
    return out
