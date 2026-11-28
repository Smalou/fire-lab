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

from core.accounts import AccountsPlan, net_value
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
    accounts: AccountsPlan | None = None  # household wrappers / metals / rent (None = single portfolio)

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
    illiquid_shortfall: np.ndarray | None = None  # (P,) shortfall while money was locked in IKE/IKZE/PPK/PPE
    bucket_wealth: np.ndarray | None = None  # (P, 2+K, Y+1) net wealth per account at the start of each year
    bucket_labels: list[str] | None = None
    captured: dict | None = None  # gross state at ``capture_month`` (for starting a decumulation run)

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
    gold_real_returns: np.ndarray | None = None,
    record_buckets: bool = False,
    capture_month: int | None = None,
) -> LifecycleResult:
    """Simulate all paths at once. ``real_returns`` has shape (P, M).

    ``gold_real_returns`` (P, M) values the physical-metals account; if missing
    it follows the portfolio. With ``plan.accounts`` the household accounts are
    simulated alongside the main portfolio (see ``core.accounts``).
    """
    r = np.atleast_2d(np.asarray(real_returns, dtype=float))
    p, m_total = r.shape
    ap = plan.accounts
    k_acc = ap.n if ap is not None else 0
    g = None
    if ap is not None and ap.uses_gold.any():
        g = r if gold_real_returns is None else np.atleast_2d(np.asarray(gold_real_returns, dtype=float))
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
    pi_m = (1.0 + plan.inflation_annual) ** (1.0 / 12.0) - 1.0

    # household accounts ---------------------------------------------------------
    main_gt = ap.main_gains_tax if ap is not None else 0.0
    basis_v = np.full(p, float(ap.main_basis) if ap is not None else 0.0)
    if k_acc:
        x = np.tile(ap.initial, (p, 1))
        bx = np.tile(ap.basis, (p, 1))
        order = [k for k in ap.order() if not ap.is_ppk[k]]
        ppk_start = np.full((p, k_acc), -1)
        oki_sum = np.zeros((p, k_acc))
    illiquid = np.zeros(p, dtype=bool)
    captured = None
    n_b = 2 + k_acc
    buckets = np.empty((p, n_b, n_years + 1)) if record_buckets else None

    def net_total() -> np.ndarray:
        tot = v - main_gt * np.maximum(v - basis_v, 0.0) + cash
        if k_acc:
            tot = tot + net_value(x, bx, ap.gains_tax, ap.flat_tax, ap.sale_cost).sum(axis=1)
        return tot

    def snapshot(col: int) -> None:
        buckets[:, 0, col] = v - main_gt * np.maximum(v - basis_v, 0.0)
        buckets[:, 1, col] = cash
        if k_acc:
            buckets[:, 2:, col] = net_value(x, bx, ap.gains_tax, ap.flat_tax, ap.sale_cost)

    wealth_sy = np.empty((p, n_years + 1))
    cons_y = np.zeros((p, n_years))
    wd_y = np.zeros((p, n_years))
    contrib_y = np.zeros((p, n_years))
    retired_months_y = np.zeros((p, n_years), dtype=np.int16)

    for m in range(plan.n_months):
        y = m // 12
        if capture_month is not None and m == capture_month:
            captured = {"v": v.copy(), "basis_v": basis_v.copy(), "cash": cash.copy(),
                        "x": x.copy() if k_acc else None, "bx": bx.copy() if k_acc else None, "net": net_total()}
        if m % 12 == 0:
            wealth_sy[:, y] = net_total()
            if record_buckets:
                snapshot(y)
        total = net_total()

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
            start_value = net_total()[trigger]
            if ws.strategy == WithdrawalStrategy.CONSTANT_REAL and ws.constant_basis == "spending":
                target_cons[trigger] = plan.spending
            else:
                target_cons[trigger] = ws.initial_rate * start_value / 12.0 + income_now

        # 2) cash flows -----------------------------------------------------------
        acc = ~retired
        if plan.coast_targets is not None:
            new_coast = acc & ~coast_hit & (net_total() >= plan.coast_targets[m] - EPS)
            coast_hit |= new_coast
            coast_month[new_coast] = m
            contrib = np.where(acc & ~coast_hit, plan.contributions[m], 0.0)
        else:
            contrib = np.where(acc, plan.contributions[m], 0.0)
        v += contrib
        basis_v += contrib
        contrib_y[:, y] += contrib
        if ap is not None:
            extra = np.where(acc, ap.main_extra[m], 0.0)
            v += extra
            basis_v += extra
            contrib_y[:, y] += extra
            if k_acc:
                add = np.where(acc[:, None], ap.contributions[:, m][None, :], 0.0)
                x += add
                bx += add
                contrib_y[:, y] += add.sum(axis=1)
        if plan.events[m] != 0.0:
            v += plan.events[m]
            if plan.events[m] > 0:
                basis_v += plan.events[m]
            neg = v < 0
            if neg.any():  # outflow larger than the portfolio: debt is not modelled
                v[neg] = 0.0

        # PPK payout: from the access age, once retired – 25% at once, 75% in 120 instalments (tax-free)
        if k_acc and ap.is_ppk.any():
            for k in np.flatnonzero(ap.is_ppk):
                start_now = retired & (ppk_start[:, k] < 0) & (m >= ap.access_month[k])
                if start_now.any():
                    ppk_start[start_now, k] = m
                    lump = 0.25 * x[start_now, k]
                    x[start_now, k] -= lump
                    v[start_now] += lump
                    basis_v[start_now] += lump
                paying = ppk_start[:, k] >= 0
                if paying.any():
                    left = np.maximum(120 - (m - ppk_start[:, k]), 1)
                    inst = np.where(paying, x[:, k] / left, 0.0)
                    x[:, k] -= inst
                    v += inst
                    basis_v += inst

        if retired.any():
            since = m - fire_month
            anniversary = retired & (since % 12 == 0)
            total = v + cash
            if anniversary.any():
                total = net_total()
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
            # main portfolio; Belka tax on the gain share of the sale (household mode)
            if main_gt > 0:
                gain_frac = np.where(v > EPS, np.clip(1.0 - basis_v / np.maximum(v, EPS), 0.0, 1.0), 0.0)
                keep = 1.0 - main_gt * gain_frac
                sold = np.minimum(rest / np.maximum(keep, EPS), v)
                basis_v -= basis_v * np.where(v > EPS, sold / np.maximum(v, EPS), 0.0)
                from_port = sold
                rest -= sold * keep
            else:
                from_port = np.minimum(rest, v)
                rest -= from_port
            extra_cash = np.minimum(rest, cash - from_cash)
            from_cash += extra_cash
            rest -= extra_cash
            cash -= from_cash
            v -= from_port
            # household accounts in order of tax efficiency (only once accessible)
            if k_acc:
                for k in order:
                    need = retired & (rest > EPS)
                    if not need.any():
                        break
                    accessible = m >= ap.access_month[k]
                    if accessible:
                        t_g = ap.gains_tax[k]
                    elif np.isfinite(ap.early_gains_tax[k]):
                        t_g = ap.early_gains_tax[k]
                    else:
                        continue
                    xk, bk = x[:, k], bx[:, k]
                    gain_frac = np.where(xk > EPS, np.clip(1.0 - bk / np.maximum(xk, EPS), 0.0, 1.0), 0.0)
                    keep = np.maximum(1.0 - ap.flat_tax[k] - ap.sale_cost[k] - t_g * gain_frac, EPS)
                    sold = np.where(need, np.minimum(rest / keep, xk), 0.0)
                    bx[:, k] -= bk * np.where(xk > EPS, sold / np.maximum(xk, EPS), 0.0)
                    x[:, k] -= sold
                    rest -= sold * keep
            paid_net_total = gross - np.maximum(rest, 0.0)  # in "gross of the generic withdrawal tax" units
            paid_net = paid_net_total / tax_gross
            consumption = paid_net + income_now
            total_w += np.where(retired, paid_net, 0.0)
            wd_y[:, y] += np.where(retired, paid_net, 0.0)
            cons_y[:, y] += np.where(retired, consumption, 0.0)
            retired_months_y[:, y] += retired
            short_now = retired & (consumption < floor - 1e-6)
            shortfall |= short_now
            if k_acc:
                locked = np.zeros(p)
                for k in range(k_acc):
                    if m < ap.access_month[k] or ap.is_ppk[k]:
                        locked += x[:, k]
                illiquid |= short_now & (locked > 1.0)
                remaining = v + cash + x.sum(axis=1)
            else:
                remaining = v + cash
            newly_depleted = retired & ~depleted & (remaining <= 1e-6) & (gross > 0)
            depleted |= newly_depleted
            depletion_month[newly_depleted] = m

        # 3) market return, fee, cash interest ---------------------------------
        rm = r[:, m]
        v *= (1.0 + rm) * (1.0 - plan.fee_monthly)
        np.maximum(v, 0.0, out=v)
        cash *= 1.0 + plan.cash_real_monthly
        if main_gt > 0:
            basis_v /= 1.0 + pi_m  # nominal cost basis loses real value
        if k_acc:
            port_growth = (1.0 + rm) * (1.0 - plan.fee_monthly)
            for k in range(k_acc):
                if ap.uses_gold[k]:
                    x[:, k] *= 1.0 + g[:, m]
                else:
                    x[:, k] *= port_growth if ap.pays_fee[k] else (1.0 + rm)
            np.maximum(x, 0.0, out=x)
            bx /= 1.0 + pi_m
            if ap.is_oki.any():
                oki_sum += np.where(ap.is_oki[None, :], x, 0.0)
                if (m + 1) % 12 == 0:
                    avg = oki_sum / 12.0
                    q = ap.oki_qualifying
                    base = np.maximum(q * avg - ap.oki_threshold, 0.0) + (1.0 - q) * avg
                    tax = np.minimum(ap.oki_value_tax * base, x) * ap.is_oki[None, :]
                    x -= tax
                    bx = np.minimum(bx, x)
                    oki_sum[:] = 0.0
        trailing[:, m % 12] = np.log1p(np.maximum(rm, -0.999999))

    wealth_sy[:, n_years] = net_total()
    if record_buckets:
        snapshot(n_years)
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
        terminal_wealth=wealth_sy[:, n_years].copy(),
        coast_month=coast_month,
        plan=plan,
        illiquid_shortfall=illiquid,
        bucket_wealth=buckets,
        captured=captured,
        bucket_labels=(["Portfel główny", "Bufor gotówkowy"] + (ap.labels if ap is not None else [])) if record_buckets else None,
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
        "p_illiquid_shortfall": float(np.mean(res.illiquid_shortfall)) if res.illiquid_shortfall is not None else 0.0,
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
