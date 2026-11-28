"""Scenario repository helpers: defaults, cloning, JSON import/export (no Streamlit)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from pydantic import ValidationError

from core.models import (
    SCHEMA_VERSION,
    FireSettings,
    HouseholdSettings,
    IncomeSource,
    InvestorProfile,
    MarketSettings,
    PortfolioConfig,
    RetirementRule,
    Scenario,
    SimulationSettings,
    WithdrawalSettings,
)
from services import catalog

APP_TAG = "FIRE Lab"


def _portfolio(weights: dict[str, float], fee: float = 0.003) -> PortfolioConfig:
    return PortfolioConfig(assets=list(weights), weights=list(weights.values()), management_fee=fee)


def default_scenarios() -> list[Scenario]:
    pension = [IncomeSource(name="Emerytura publiczna", monthly_amount=2_500, start_age=65)]
    balanced = Scenario(
        name="Zrównoważony",
        description="Portfel globalny 70/30, przejście na FIRE w wieku 50 lat.",
        profile=InvestorProfile(current_age=32, fire_age=50, initial_capital=150_000, monthly_contribution=4_000),
        fire=FireSettings(swr=0.04, incomes=pension),
        portfolio=_portfolio(catalog.default_portfolio("demo")),
        market=MarketSettings(data_source="demo", benchmark="SYN_GLOBAL"),
    )
    aggressive = Scenario(
        name="Agresywny",
        description="Portfel akcyjny 85/15, wyższe wpłaty, FIRE możliwie wcześnie (po osiągnięciu celu).",
        profile=InvestorProfile(current_age=32, fire_age=45, initial_capital=150_000, monthly_contribution=5_000,
                                monthly_expenses=7_500, retirement_monthly_expenses=7_500),
        fire=FireSettings(swr=0.04, retirement_rule=RetirementRule.WHEN_TARGET_REACHED, max_work_age=60, incomes=pension),
        portfolio=_portfolio({"SYN_GLOBAL": 0.55, "SYN_US": 0.15, "SYN_EM": 0.15, "SYN_BONDS": 0.10, "SYN_GOLD": 0.05}),
        market=MarketSettings(data_source="demo", benchmark="SYN_GLOBAL"),
        simulation=SimulationSettings(fat_tails=True),
    )
    conservative = Scenario(
        name="Konserwatywny",
        description="Portfel 35/65 z przewagą obligacji, ostrożna stopa wypłat 3,5%, FIRE w wieku 55 lat.",
        profile=InvestorProfile(current_age=32, fire_age=55, initial_capital=150_000, monthly_contribution=3_500,
                                retirement_monthly_expenses=7_000),
        fire=FireSettings(swr=0.035, incomes=pension),
        portfolio=_portfolio({"SYN_GLOBAL": 0.30, "SYN_EUROPE": 0.05, "SYN_BONDS": 0.45, "SYN_GOLD": 0.10, "SYN_CASH": 0.10}),
        market=MarketSettings(data_source="demo", benchmark="SYN_GLOBAL"),
        withdrawal=WithdrawalSettings(initial_rate=0.035),
    )
    household = Scenario(
        name="Rodzina z kontami",
        description="Para 35/34 lata: IKE, IKZE, PPK, kruszce i mieszkanie na wynajem; wspólny FIRE w wieku 52 lat.",
        profile=InvestorProfile(current_age=35, fire_age=52, initial_capital=120_000, monthly_net_income=18_000,
                                monthly_expenses=11_000, monthly_contribution=3_000, retirement_monthly_expenses=9_500),
        fire=FireSettings(swr=0.04, incomes=[IncomeSource(name="Emerytura publiczna", monthly_amount=2_500, start_age=65),
                                             IncomeSource(name="Emerytura małżonka", monthly_amount=2_200, start_age=66)]),
        portfolio=_portfolio(catalog.default_portfolio("demo")),
        market=MarketSettings(data_source="demo", benchmark="SYN_GLOBAL"),
        household=HouseholdSettings.model_validate({
            "enabled": True, "joint_planning": True, "spouse_age": 34, "main_account_basis_share": 0.8,
            "accounts": [
                {"kind": "ike", "owner": "self", "balance": 60_000, "monthly_own": 1_500},
                {"kind": "ikze", "owner": "self", "balance": 20_000, "monthly_own": 942},
                {"kind": "ppk", "owner": "self", "balance": 25_000, "monthly_own": 160, "monthly_employer": 120},
                {"kind": "ike", "owner": "spouse", "balance": 30_000, "monthly_own": 800},
                {"kind": "ppk", "owner": "spouse", "balance": 15_000, "monthly_own": 120, "monthly_employer": 90},
            ],
            "metals": {"value": 40_000, "monthly_purchase": 200},
            "rental": {"enabled": True, "monthly_gross": 2_800, "monthly_costs": 300, "vacancy_rate": 0.05, "tax_rate": 0.085,
                       "start_age": 35, "reinvest_share": 1.0},
        }),
    )
    return [balanced, aggressive, conservative, household]


def clone(s: Scenario, name: str | None = None) -> Scenario:
    now = datetime.now().replace(microsecond=0)
    data = s.model_dump()
    data.update(id=uuid.uuid4().hex[:10], name=name or f"{s.name} (kopia)", created_at=now, updated_at=now)
    return Scenario.model_validate(data)


def to_json(scenarios: list[Scenario]) -> str:
    payload = {
        "app": APP_TAG,
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now().replace(microsecond=0).isoformat(),
        "scenarios": [s.model_dump(mode="json") for s in scenarios],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


class ImportErrorPL(ValueError):
    pass


def from_json(text: str | bytes, existing_ids: set[str] | None = None) -> list[Scenario]:
    """Parse exported JSON (single scenario or bundle). Raises ImportErrorPL with Polish message."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ImportErrorPL(f"Plik nie jest poprawnym JSON-em: {exc.msg} (wiersz {exc.lineno}).") from exc
    items = payload.get("scenarios") if isinstance(payload, dict) and "scenarios" in payload else [payload]
    if not isinstance(items, list) or not items:
        raise ImportErrorPL("Plik nie zawiera scenariuszy.")
    out: list[Scenario] = []
    existing_ids = set(existing_ids or ())
    for i, item in enumerate(items, start=1):
        if isinstance(item, dict) and item.get("schema_version", SCHEMA_VERSION) > SCHEMA_VERSION:
            raise ImportErrorPL(f"Scenariusz {i}: nowsza wersja schematu ({item['schema_version']}) niż obsługiwana.")
        try:
            s = Scenario.model_validate(item)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"])
            raise ImportErrorPL(f"Scenariusz {i}: niepoprawne pole „{loc}” – {first['msg']}.") from exc
        if s.id in existing_ids:
            s = s.model_copy(update={"id": uuid.uuid4().hex[:10]})
        existing_ids.add(s.id)
        out.append(s)
    return out


def remap_portfolio_to_source(s: Scenario, source: str) -> PortfolioConfig:
    """When switching data source, map assets by asset class to the defaults of the new source."""
    defaults = catalog.default_portfolio(source)
    if source == "csv":
        return s.portfolio
    current_classes = catalog.classes_for(s.portfolio.assets)
    if all((catalog.get(a) and catalog.get(a).source == ("demo" if source == "demo" else "yfinance")) for a in s.portfolio.assets):
        return s.portfolio
    by_class = {catalog.get(a).asset_class: a for a in defaults}
    for inst in catalog.instruments(source):
        by_class.setdefault(inst.asset_class, inst.id)
    weights: dict[str, float] = {}
    for a, w in zip(s.portfolio.assets, s.portfolio.weights):
        target = by_class.get(current_classes[a])
        if target:
            weights[target] = weights.get(target, 0.0) + w
    if not weights:
        weights = defaults
    total = sum(weights.values())
    return s.portfolio.model_copy(update={"assets": list(weights), "weights": [w / total for w in weights.values()]})
