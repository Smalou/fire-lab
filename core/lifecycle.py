"""Vectorised life-cycle engine: accumulation → FIRE → decumulation.

The engine is shared by the deterministic projection (1 path, constant
return), the Monte Carlo risk lab and the post-FIRE (decumulation) lab, so all
modules use **one** set of conventions:

* Everything is computed in **real** money (today's base currency). Nominal
  market returns are deflated exactly once: r_real = (1+r)/(1+π_m) − 1.
* Monthly timing: at the start of month m the retirement decision is taken,
  then cash flows occur (contribution or withdrawal, events), then the
  month's market return is applied, and finally the management fee is charged.
* Withdrawal budgets are set once per retirement year (on each anniversary of
  the FIRE month) and paid in 12 equal monthly instalments.
* Post-FIRE income (pension, part-time work) is counted only while retired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.models import RetirementRule, WithdrawalSettings, WithdrawalStrategy

EPS = 1e-9


@dataclass
class LifecyclePlan:
    current_age: int
    n_months: int
    initial_capital: float
    contributions: np.ndarray  # (M,) real monthly contribution while accumulating
    events: np.ndarray  # (M,) real signed one-off/recurring flows
    spending: float  # required real monthly spending after FIRE
    other_income: np.ndarray  # (M,) real monthly income counted while retired (pension etc.)
    barista_income: np.ndarray  # (M,) real monthly part-time income while retired (Barista FIRE)
    fire_target: float  # real capital required for FIRE
    retirement_rule: RetirementRule
    fire_month: int
    max_work_month: int
    coast_targets: np.ndarray | None = None  # (M,) Coast FIRE number per month
    cash_buffer: float = 0.0  # Fat FIRE liquidity reserve moved to cash at FIRE
    cash_real_monthly: float = 0.0  # real monthly return of the cash buffer
    fee_monthly: float = 0.0
    inflation_annual: float = 0.0

    @property
    def n_years(self) -> int:
        return self.n_months // 12

    def age_at_month(self, m) -> np.ndarray:
        return self.current_age + np.asarray(m) / 12.0


@dataclass
class LifecycleResult:
    wealth_start_year: np.ndarray  # (P, Y+1) real wealth at the start of each year of age; last = horizon
    consumption_year: np.ndarray  # (P, Y) real consumption in fully retired years (NaN otherwise)
    withdrawal_year: np.ndarray  # (P, Y) real net withdrawals from the portfolio per year
    contribution_year: np.ndarray  # (P, Y) real contributions per year
    fire_month: np.ndarray  # (P,) month of FIRE (-1 never)
    first_hit_month: np.ndarray  # (P,) first month real wealth ≥ target while accumulating (-1 never)
    depletion_month: np.ndarray  # (P,) month the portfolio ran out (-1 never)
    shortfall: np.ndarray  # (P,) True if consumption fell below the floor in any retired month
    total_withdrawals: np.ndarray  # (P,) real net withdrawals over life
    terminal_wealth: np.ndarray  # (P,) real wealth at horizon
    coast_month: np.ndarray  # (P,) month Coast number reached (-1 never / n.a.)
    plan: LifecyclePlan

    @property
    def n_paths(self) -> int:
        return self.terminal_wealth.size

    def ages(self) -> np.ndarray:
        return self.plan.current_age + np.arange(self.wealth_start_year.shape[1])


def deflate_returns(nominal_monthly: np.ndarray, inflation_annual: float) -> np.ndarray:
    pi_m = (1.0 + inflation_annual) ** (1.0 / 12.0) - 1.0
    return (1.0 + nominal_monthly) / (1.0 + pi_m) - 1.0


def run_lifecycle(
    real_returns: np.ndarray,
    plan: LifecyclePlan,
    ws: WithdrawalSettings,
    income_yield: float | None = None,
) -> LifecycleResult:
    """Simulate all paths at once. ``real_returns`` has shape (P, M)."""
    r = np.atleast_2d(np.asarray(real_returns, dtype=float))
    p, m_total = r.shape
    if m_total < plan.n_months:
        raise ValueError("Za krótka macierz stóp zwrotu względem horyzontu planu.")
    if ws.strategy == WithdrawalStrategy.INCOME_ONLY and (income_yield is None or not np.isfinite(income_yield)):
        raise ValueError("Strategia „tylko dochód” wymaga danych o dywidendach / kuponach (stopa dochodu portfela).")

    n_years = plan.n_years
    tax_gross = 1.0 / (1.0 - ws.withdrawal_tax_rate)
    floor = ws.spending_floor_share * plan.spending
    consumption_targeting = ws.strategy in (WithdrawalStrategy.CONSTANT_REAL, WithdrawalStrategy.GUARDRAILS)

    v = np.full(p, float(plan.initial_capital))
    cash = np.zeros(p)
    retired = np.zeros(p, dtype=bool)
    fire_month = np.full(p, -1)
    first_hit = np.full(p, -1)
    coast_hit = np.zeros(p, dtype=bool)
    coast_month = np.full(p, -1)
    depleted = np.zeros(p, dtype=bool)
    depletion_month = np.full(p, -1)
    shortfall = np.zeros(p, dtype=bool)
    target_cons = np.zeros(p)  # monthly consumption target (constant real / guardrails)
    budget = np.zeros(p)  # monthly withdrawal budget (percent / income)
    init_rate = np.full(p, ws.initial_rate)
    trailing = np.zeros((p, 12))  # last 12 log real returns (ring buffer)
    total_w = np.zeros(p)

    wealth_sy = np.empty((p, n_years + 1))
    cons_y = np.zeros((p, n_years))
    wd_y = np.zeros((p, n_years))
    contrib_y = np.zeros((p, n_years))
    retired_months_y = np.zeros((p, n_years), dtype=np.int16)

    for m in range(plan.n_months):
        y = m // 12
        if m % 12 == 0:
            wealth_sy[:, y] = v + cash
        total = v + cash

        # 1) retirement decision ------------------------------------------------
        acc = ~retired
        hit = acc & (total >= plan.fire_target - EPS) & (first_hit < 0)
        first_hit[hit] = m
        if plan.retirement_rule == RetirementRule.FIXED_AGE:
            trigger = acc & (m == plan.fire_month)
        else:
            trigger = acc & ((total >= plan.fire_target - EPS) | (m >= plan.max_work_month))
        if trigger.any():
            retired |= trigger
            fire_month[trigger] = m
            if plan.cash_buffer > 0:
                moved = np.minimum(plan.cash_buffer, v[trigger])
                v[trigger] -= moved
                cash[trigger] += moved
            income_now = plan.other_income[m] + plan.barista_income[m]
            start_value = v[trigger] + cash[trigger]
            if ws.strategy == WithdrawalStrategy.CONSTANT_REAL and ws.constant_basis == "spending":
                target_cons[trigger] = plan.spending
            else:
                target_cons[trigger] = ws.initial_rate * start_value / 12.0 + income_now

        # 2) cash flows -----------------------------------------------------------
        acc = ~retired
        if plan.coast_targets is not None:
            new_coast = acc & ~coast_hit & (v >= plan.coast_targets[m] - EPS)
            coast_hit |= new_coast
            coast_month[new_coast] = m
            contrib = np.where(acc & ~coast_hit, plan.contributions[m], 0.0)
        else:
            contrib = np.where(acc, plan.contributions[m], 0.0)
        v += contrib
        contrib_y[:, y] += contrib
        if plan.events[m] != 0.0:
            v += plan.events[m]
            neg = v < 0
            if neg.any():  # outflow larger than the portfolio: debt is not modelled
                v[neg] = 0.0

        if retired.any():
            since = m - fire_month
            anniversary = retired & (since % 12 == 0)
            total = v + cash
            if anniversary.any():
                if ws.strategy == WithdrawalStrategy.CONSTANT_PERCENT:
                    budget[anniversary] = ws.percent_rate * total[anniversary] / 12.0
                elif ws.strategy == WithdrawalStrategy.INCOME_ONLY:
                    budget[anniversary] = income_yield * v[anniversary] / 12.0
                elif ws.strategy == WithdrawalStrategy.GUARDRAILS:
                    later = anniversary & (since > 0) & (total > EPS)
                    if later.any():
                        income_now = plan.other_income[m] + plan.barista_income[m]
                        need = np.maximum(target_cons - income_now, 0.0)
                        cur_rate = np.where(total > EPS, need * 12.0 / np.maximum(total, EPS), np.inf)
                        last_year_real = np.expm1(trailing.sum(axis=1))
                        # Guyton–Klinger simplified: no inflation adjustment after a losing year
                        freeze = later & (last_year_real < 0) & (cur_rate > init_rate)
                        target_cons[freeze] /= (1.0 + plan.inflation_annual)
                        cut = later & (cur_rate > init_rate * (1 + ws.guardrail_upper))
                        raise_ = later & (cur_rate < init_rate * (1 - ws.guardrail_lower))
                        target_cons[cut] *= 1.0 - ws.guardrail_adjustment
                        target_cons[raise_] *= 1.0 + ws.guardrail_adjustment

            income_now = plan.other_income[m] + plan.barista_income[m]
            if consumption_targeting:
                net = np.where(retired, np.maximum(target_cons - income_now, 0.0), 0.0)
            else:
                net = np.where(retired, budget, 0.0)
            gross = net * tax_gross
            # Fat FIRE: after a losing year draw from the cash buffer first.
            use_cash_first = (cash > 0) & (trailing.sum(axis=1) < 0)
            from_cash = np.where(use_cash_first, np.minimum(gross, cash), 0.0)
            rest = gross - from_cash
            from_port = np.minimum(rest, v)
            rest -= from_port
            extra_cash = np.minimum(rest, cash - from_cash)
            from_cash += extra_cash
            paid_gross = from_cash + from_port
            cash -= from_cash
            v -= from_port
            paid_net = paid_gross / tax_gross
            consumption = paid_net + income_now
            total_w += np.where(retired, paid_net, 0.0)
            wd_y[:, y] += np.where(retired, paid_net, 0.0)
            cons_y[:, y] += np.where(retired, consumption, 0.0)
            retired_months_y[:, y] += retired
            shortfall |= retired & (consumption < floor - 1e-6)
            newly_depleted = retired & ~depleted & (v + cash <= 1e-6) & (gross > 0)
            depleted |= newly_depleted
            depletion_month[newly_depleted] = m

        # 3) market return, fee, cash interest ---------------------------------
        rm = r[:, m]
        v *= (1.0 + rm) * (1.0 - plan.fee_monthly)
        np.maximum(v, 0.0, out=v)
        cash *= 1.0 + plan.cash_real_monthly
        trailing[:, m % 12] = np.log1p(np.maximum(rm, -0.999999))

    wealth_sy[:, n_years] = v + cash
    full = retired_months_y == 12
    cons_full = np.where(full, cons_y, np.nan)
    return LifecycleResult(
        wealth_start_year=wealth_sy,
        consumption_year=cons_full,
        withdrawal_year=wd_y,
        contribution_year=contrib_y,
        fire_month=fire_month,
        first_hit_month=first_hit,
        depletion_month=depletion_month,
        shortfall=shortfall,
        total_withdrawals=total_w,
        terminal_wealth=v + cash,
        coast_month=coast_month,
        plan=plan,
    )


# ------------------------------------------------------------------ summaries


def consumption_volatility(cons_year: np.ndarray) -> np.ndarray:
    """Per-path std of year-on-year % changes of real consumption (fully retired years)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ch = cons_year[:, 1:] / cons_year[:, :-1] - 1.0
    ch = np.where(np.isfinite(ch), ch, np.nan)
    counts = np.sum(np.isfinite(ch), axis=1)
    out = np.full(cons_year.shape[0], np.nan)
    ok = counts >= 2
    if ok.any():
        out[ok] = np.nanstd(ch[ok], axis=1, ddof=1)
    return out


def max_consumption_cut(cons_year: np.ndarray) -> np.ndarray:
    """Largest drop of real annual consumption from its running peak (0..1) per path."""
    c = np.where(np.isfinite(cons_year), cons_year, np.nan)
    peak = np.fmax.accumulate(np.nan_to_num(c, nan=-np.inf), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = 1.0 - c / peak
    dd = np.where(np.isfinite(dd), dd, np.nan)
    out = np.nanmax(np.where(np.isnan(dd), -np.inf, dd), axis=1)
    return np.where(np.isfinite(out), np.maximum(out, 0.0), np.nan)


def summarize(res: LifecycleResult) -> dict[str, float]:
    plan = res.plan
    retired = res.fire_month >= 0
    depleted = res.depletion_month >= 0
    fire_ok_by_target = (res.first_hit_month >= 0) & (res.first_hit_month <= plan.fire_month)
    out = {
        "n_paths": float(res.n_paths),
        "p_fire_by_target_age": float(np.mean(fire_ok_by_target)),
        "p_retired": float(np.mean(retired)),
        "p_not_depleted": float(np.mean(~depleted)),
        "p_spending_covered": float(np.mean(retired & ~res.shortfall)),
        "p_success": float(np.mean(retired & ~depleted & ~res.shortfall)),
        "median_terminal_wealth": float(np.median(res.terminal_wealth)),
        "median_total_withdrawals": float(np.median(res.total_withdrawals)),
        "median_consumption_vol": float(np.nanmedian(consumption_volatility(res.consumption_year)))
        if np.isfinite(res.consumption_year).any()
        else float("nan"),
        "median_max_cut": float(np.nanmedian(max_consumption_cut(res.consumption_year)))
        if np.isfinite(res.consumption_year).any()
        else float("nan"),
        "p95_max_cut": float(np.nanpercentile(max_consumption_cut(res.consumption_year), 95))
        if np.isfinite(res.consumption_year).any()
        else float("nan"),
    }
    hit = res.first_hit_month[res.first_hit_month >= 0]
    out["median_fire_age"] = float(plan.current_age + np.median(hit) / 12.0) if hit.size else float("nan")
    return out
