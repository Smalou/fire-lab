"""Central scenario model.

One ``Scenario`` object parametrises every module of the application
(planner, portfolio builder, backtest, Monte Carlo, decumulation, comparison).

Conventions
-----------
* All money amounts entered by the user are **real** values expressed in
  today's money of the base currency (default PLN).
* Growth rates of income and contributions are **real** (above inflation).
* Inflation, the risk-free rate and market returns are **nominal**.
* Rates are decimals (0.04 == 4%).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1


class FireMode(str, Enum):
    TRADITIONAL = "traditional"
    LEAN = "lean"
    FAT = "fat"
    COAST = "coast"
    BARISTA = "barista"


class RetirementRule(str, Enum):
    FIXED_AGE = "fixed_age"
    WHEN_TARGET_REACHED = "when_target_reached"


class OptimizationMethod(str, Enum):
    MANUAL = "manual"
    EQUAL_WEIGHT = "equal_weight"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    TARGET_RETURN = "target_return"


class WithdrawalStrategy(str, Enum):
    CONSTANT_REAL = "constant_real"
    CONSTANT_PERCENT = "constant_percent"
    GUARDRAILS = "guardrails"
    INCOME_ONLY = "income_only"


class SimulationModel(str, Enum):
    PARAMETRIC = "parametric"
    BOOTSTRAP = "bootstrap"


class _Base(BaseModel):
    model_config = ConfigDict(validate_assignment=False, extra="forbid")


class FinancialEvent(_Base):
    """Additional cash flow in real money. Positive = inflow, negative = outflow."""

    enabled: bool = True
    name: str = "Zdarzenie"
    category: Literal["inflow", "outflow"] = "outflow"
    age: int = Field(40, ge=0, le=120)
    amount: float = Field(0.0, description="Absolute amount in real base currency")
    recurring: bool = False
    end_age: int | None = Field(None, ge=0, le=120)

    @model_validator(mode="after")
    def _check_end(self) -> "FinancialEvent":
        if self.recurring and self.end_age is not None and self.end_age < self.age:
            raise ValueError("Wiek zakończenia zdarzenia cyklicznego musi być ≥ wieku rozpoczęcia.")
        return self

    @property
    def signed_amount(self) -> float:
        return abs(self.amount) if self.category == "inflow" else -abs(self.amount)


class IncomeSource(_Base):
    """Post-FIRE income stream in real money (e.g. public pension, part-time work)."""

    enabled: bool = True
    name: str = "Emerytura publiczna"
    monthly_amount: float = Field(0.0, ge=0)
    start_age: int = Field(65, ge=0, le=120)
    end_age: int | None = Field(None, ge=0, le=120)


class InvestorProfile(_Base):
    current_age: int = Field(32, ge=16, le=90)
    fire_age: int = Field(50, ge=17, le=100)
    life_horizon_age: int = Field(95, ge=40, le=120)
    initial_capital: float = Field(150_000.0, ge=0)
    monthly_net_income: float = Field(14_000.0, ge=0)
    monthly_expenses: float = Field(8_000.0, ge=0)
    monthly_contribution: float = Field(4_000.0, ge=0)
    retirement_monthly_expenses: float = Field(7_500.0, ge=0)
    inflation: float = Field(0.03, ge=-0.05, le=0.30)
    income_growth_real: float = Field(0.01, ge=-0.10, le=0.20)
    contribution_growth_real: float = Field(0.01, ge=-0.10, le=0.20)
    events: list[FinancialEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_ages(self) -> "InvestorProfile":
        if not self.current_age < self.fire_age:
            raise ValueError("Docelowy wiek FIRE musi być większy niż obecny wiek.")
        if not self.fire_age < self.life_horizon_age:
            raise ValueError("Horyzont życia musi być dłuższy niż docelowy wiek FIRE.")
        return self


class FireSettings(_Base):
    mode: FireMode = FireMode.TRADITIONAL
    swr: float = Field(0.04, gt=0.0, le=0.15)
    retirement_rule: RetirementRule = RetirementRule.FIXED_AGE
    max_work_age: int = Field(67, ge=18, le=100)
    lean_essential_share: float = Field(0.75, gt=0.0, le=1.0)
    fat_extra_share: float = Field(0.30, ge=0.0, le=3.0)
    fat_cash_buffer_years: float = Field(2.0, ge=0.0, le=10.0)
    barista_monthly_income: float = Field(3_000.0, ge=0)
    barista_end_age: int = Field(60, ge=18, le=100)
    incomes: list[IncomeSource] = Field(
        default_factory=lambda: [IncomeSource(name="Emerytura publiczna", monthly_amount=2_500, start_age=65)]
    )


class PortfolioConfig(_Base):
    assets: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    method: OptimizationMethod = OptimizationMethod.MANUAL
    target_return: float | None = Field(None, description="Annual arithmetic nominal target return")
    min_weight: float = Field(0.0, ge=-1.0, le=1.0)
    max_weight: float = Field(1.0, ge=0.0, le=2.0)
    allow_short: bool = False
    management_fee: float = Field(0.003, ge=0.0, le=0.05, description="Annual cost (TER + platform)")

    @model_validator(mode="after")
    def _check_weights(self) -> "PortfolioConfig":
        if len(self.assets) != len(self.weights):
            raise ValueError("Liczba wag musi odpowiadać liczbie aktywów.")
        if len(set(self.assets)) != len(self.assets):
            raise ValueError("Aktywa w portfelu nie mogą się powtarzać.")
        if self.assets and abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError(f"Suma wag musi wynosić 100% (obecnie {sum(self.weights) * 100:.2f}%).".replace(".", ",", 1))
        if not self.allow_short and any(w < -1e-9 for w in self.weights):
            raise ValueError("Ujemne wagi są niedozwolone przy zakazie krótkiej sprzedaży.")
        if self.min_weight > self.max_weight:
            raise ValueError("Minimalna waga nie może przekraczać maksymalnej.")
        return self


class MarketSettings(_Base):
    data_source: Literal["demo", "yfinance", "snapshot", "csv"] = "demo"
    base_currency: Literal["PLN", "USD", "EUR"] = "PLN"
    start_date: date | None = None  # None == longest common history
    end_date: date | None = None
    estimation_years: int = Field(10, ge=3, le=40)
    risk_free_rate: float = Field(0.04, ge=-0.05, le=0.30)
    covariance_method: Literal["sample", "ledoit_wolf"] = "sample"
    benchmark: str | None = None

    @model_validator(mode="after")
    def _check_dates(self) -> "MarketSettings":
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("Data początkowa musi poprzedzać datę końcową.")
        return self


class BacktestSettings(_Base):
    rebalance: Literal["none", "monthly", "quarterly", "semiannual", "annual"] = "annual"
    initial_investment: float = Field(100_000.0, ge=0)
    monthly_contribution: float = Field(2_000.0, ge=0)
    commission_rate: float = Field(0.0019, ge=0.0, le=0.05, description="Commission as share of traded value")
    spread_bps: float = Field(5.0, ge=0.0, le=500.0, description="Bid-ask / market impact in basis points")
    estimation_months: int = Field(60, ge=12, le=240)
    glide_start_equity: float = Field(0.90, ge=0.0, le=1.0)
    glide_end_equity: float = Field(0.40, ge=0.0, le=1.0)


class SimulationSettings(_Base):
    model: SimulationModel = SimulationModel.PARAMETRIC
    n_paths: int = Field(2_000, ge=100, le=10_000)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    block_months: int = Field(12, ge=1, le=60)
    fat_tails: bool = False
    t_dof: float = Field(5.0, gt=2.0, le=100.0)
    shrinkage: bool = False
    return_haircut: float = Field(0.0, ge=-0.05, le=0.10, description="Annual reduction of expected log return")
    var_confidence: float = Field(0.95, ge=0.80, le=0.999)
    var_horizon_years: int = Field(1, ge=1, le=30)


class WithdrawalSettings(_Base):
    strategy: WithdrawalStrategy = WithdrawalStrategy.CONSTANT_REAL
    constant_basis: Literal["rate", "spending"] = "rate"
    initial_rate: float = Field(0.04, gt=0.0, le=0.20)
    percent_rate: float = Field(0.04, gt=0.0, le=0.30)
    guardrail_upper: float = Field(0.20, ge=0.0, le=1.0)
    guardrail_lower: float = Field(0.20, ge=0.0, le=1.0)
    guardrail_adjustment: float = Field(0.10, ge=0.0, le=0.5)
    spending_floor_share: float = Field(0.90, ge=0.0, le=1.0)
    withdrawal_tax_rate: float = Field(0.0, ge=0.0, le=0.6)


class Scenario(_Base):
    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str = Field("Nowy scenariusz", min_length=1, max_length=80)
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now().replace(microsecond=0))
    updated_at: datetime = Field(default_factory=lambda: datetime.now().replace(microsecond=0))
    profile: InvestorProfile = Field(default_factory=InvestorProfile)
    fire: FireSettings = Field(default_factory=FireSettings)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    market: MarketSettings = Field(default_factory=MarketSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    withdrawal: WithdrawalSettings = Field(default_factory=WithdrawalSettings)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Nazwa scenariusza nie może być pusta.")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> "Scenario":
        if self.fire.max_work_age < self.profile.fire_age and self.fire.retirement_rule == RetirementRule.WHEN_TARGET_REACHED:
            raise ValueError("Maksymalny wiek pracy nie może być niższy niż docelowy wiek FIRE.")
        if self.fire.max_work_age >= self.profile.life_horizon_age:
            raise ValueError("Maksymalny wiek pracy musi być niższy niż horyzont życia.")
        return self

    def fingerprint(self, *sections: str) -> str:
        """Stable hash of the selected sections; used to detect stale results."""
        payload = {s: getattr(self, s).model_dump(mode="json") for s in sorted(sections)}
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def touch(self) -> "Scenario":
        self.updated_at = datetime.now().replace(microsecond=0)
        return self


# Sections every result type depends on (stale-result detection).
RESULT_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "portfolio": ("market", "portfolio"),
    "backtest": ("market", "portfolio", "backtest"),
    "montecarlo": ("market", "portfolio", "profile", "fire", "simulation", "withdrawal"),
    "decumulation": ("market", "portfolio", "profile", "fire", "simulation", "withdrawal"),
}
