"""Household balance sheet: Polish tax wrappers, spouse accounts, metals, rent.

Turns ``Scenario.household`` into arrays used by the life-cycle engine.
All amounts are real (today's PLN); tax rules follow the 2026 state of law
(see ``TaxRules`` and the methodology page) with documented simplifications:

* **Rachunek maklerski** – Belka tax (19%) on *nominal* gains at sale; the
  cost basis is kept in nominal terms, i.e. it shrinks in real terms with
  inflation (inflation is taxed, as in reality).
* **IKE** – no tax when withdrawn from the access age (60); earlier
  withdrawal ("zwrot") is allowed with Belka tax on gains (optional).
* **IKZE** – withdrawal from 65 taxed with a flat 10% of the amount; the
  contribution reduces PIT (refund at the marginal rate, optionally reinvested
  in the main account). Early withdrawal is not modelled (not a source).
* **PPK** – own + employer contributions + annual state subsidy; from 60
  (and only once retired) 25% is paid out at once and 75% in 120 monthly
  instalments – both tax-free – into the main account.
* **PPE** – employer (+ own) contributions; tax-free withdrawal from 60.
* **OKI** – (from 1.01.2027) no Belka tax; annual tax of 0.85% on the average
  value above the 100 000 PLN exemption for qualifying assets; non-qualifying
  assets are taxed in full; free withdrawals.
* **Metale** – physical gold valued with the gold price series; no PIT after
  6 months of holding (art. 10 ust. 1 pkt 8 PIT); selling loses the spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from core.models import Scenario

KIND_LABELS = {
    "taxable": "Rachunek maklerski",
    "ike": "IKE",
    "ikze": "IKZE",
    "ppk": "PPK",
    "ppe": "PPE",
    "oki": "OKI",
    "metals": "Metale szlachetne",
}
OWNER_LABELS = {"self": "Ja", "spouse": "Małżonek"}

# Withdrawal priority after the main account and the cash buffer (lower = earlier).
DRAW_PRIORITY = {"taxable": 0, "oki": 1, "metals": 2, "ike": 3, "ppe": 4, "ikze": 5, "ppk": 99}
OKI_START = date(2027, 1, 1)


@dataclass
class AccountsPlan:
    labels: list[str]
    kinds: list[str]
    initial: np.ndarray  # (K,)
    basis: np.ndarray  # (K,) real paid-in capital
    contributions: np.ndarray  # (K, M) real monthly, paid only while working
    access_month: np.ndarray  # (K,) first month a withdrawal is allowed
    gains_tax: np.ndarray  # (K,) tax rate on gains at withdrawal
    flat_tax: np.ndarray  # (K,) tax rate on the withdrawn amount
    sale_cost: np.ndarray  # (K,)
    uses_gold: np.ndarray  # (K,) bool – valued with the gold series
    pays_fee: np.ndarray  # (K,) bool – portfolio management fee applies
    is_ppk: np.ndarray  # (K,) bool – forced 25% + 120 instalments payout
    early_gains_tax: np.ndarray  # (K,) NaN = no early withdrawal; else tax on gains before access
    is_oki: np.ndarray  # (K,) bool
    oki_value_tax: float
    oki_threshold: float
    oki_qualifying: float
    main_gains_tax: float  # Belka on the main portfolio
    main_basis: float  # real paid-in part of the main capital
    main_extra: np.ndarray  # (M,) extra main-account inflow while working (IKZE refund, reinvested rent)
    capped: list[str]  # messages about contribution limits

    @property
    def n(self) -> int:
        return len(self.kinds)

    def order(self) -> list[int]:
        return sorted(range(self.n), key=lambda k: (DRAW_PRIORITY[self.kinds[k]], k))


def months_until(d: date, today: date | None = None) -> int:
    today = today or date.today()
    return max(0, (d.year - today.year) * 12 + (d.month - today.month))


def rental_schedule(s: Scenario, n_months: int) -> np.ndarray:
    r = s.household.rental
    if not (s.household.enabled and r.enabled) or r.monthly_net <= 0:
        return np.zeros(n_months)
    ages = s.profile.current_age + np.arange(n_months) / 12.0
    end = r.end_age if r.end_age is not None else 1e9
    return np.where((ages >= r.start_age) & (ages < end), r.monthly_net, 0.0)


def build_accounts(s: Scenario, n_months: int, today: date | None = None) -> AccountsPlan | None:
    hh = s.household
    if not hh.enabled:
        return None
    tx = hh.taxes
    prof = s.profile
    labels, kinds = [], []
    cols: dict[str, list] = {k: [] for k in ("initial", "basis", "contrib", "access", "gains", "flat", "sale", "gold", "fee", "ppk",
                                              "early", "oki")}
    ikze_refund = np.zeros(n_months)
    capped: list[str] = []
    oki_start = months_until(OKI_START, today)
    years = np.arange(n_months)

    # contribution limits per owner and wrapper kind (annual, in real terms)
    own_sum: dict[tuple[str, str], float] = {}
    for a in hh.accounts:
        if a.enabled and (a.owner == "self" or hh.joint_planning):
            own_sum[(a.owner, a.kind)] = own_sum.get((a.owner, a.kind), 0.0) + 12 * a.monthly_own

    def limit_scale(owner: str, kind: str) -> float:
        if kind == "ike":
            lim = tx.ike_limit
        elif kind == "ikze":
            se = hh.self_employed if owner == "self" else hh.spouse_self_employed
            lim = tx.ikze_limit_self_employed if se else tx.ikze_limit
        else:
            return 1.0
        total = own_sum.get((owner, kind), 0.0)
        if total > lim + 1e-6:
            capped.append(f"{KIND_LABELS[kind]} ({OWNER_LABELS[owner].lower()}): wpłaty {total:,.0f} zł/rok przekraczają limit "
                          f"{lim:,.0f} zł – przyjęto limit.".replace(",", " "))
            return lim / total
        return 1.0

    for a in hh.accounts:
        if not a.enabled or (a.owner == "spouse" and not hh.joint_planning):
            continue
        owner_age = prof.current_age if a.owner == "self" else hh.spouse_age
        own = a.monthly_own * limit_scale(a.owner, a.kind)
        contrib = np.full(n_months, own + a.monthly_employer)
        access_age = {"ike": tx.ike_access_age, "ikze": tx.ikze_access_age, "ppk": tx.ppk_access_age,
                      "ppe": tx.ppe_access_age}.get(a.kind)
        access = 0 if access_age is None else max(0, (access_age - owner_age) * 12)
        if a.kind == "ppk":
            contrib = contrib + tx.ppk_state_annual / 12.0
        if a.kind == "oki":
            contrib[:oki_start] = 0.0
        if a.kind == "ikze":
            ikze_refund += own * tx.pit_marginal_rate if tx.reinvest_ikze_refund else 0.0
        labels.append(f"{KIND_LABELS[a.kind]} · {OWNER_LABELS[a.owner]}")
        kinds.append(a.kind)
        cols["initial"].append(a.balance)
        cols["basis"].append(a.balance if a.basis is None else min(a.basis, a.balance) if a.balance > 0 else 0.0)
        cols["contrib"].append(contrib)
        cols["access"].append(access)
        cols["gains"].append(tx.belka_rate if a.kind == "taxable" else 0.0)
        cols["flat"].append(tx.ikze_withdrawal_tax if a.kind == "ikze" else 0.0)
        cols["sale"].append(0.0)
        cols["gold"].append(False)
        cols["fee"].append(True)
        cols["ppk"].append(a.kind == "ppk")
        cols["early"].append(tx.belka_rate if (a.kind == "ike" and tx.allow_early_ike) else np.nan)
        cols["oki"].append(a.kind == "oki")

    m = hh.metals
    if m.value > 0 or m.monthly_purchase > 0:
        labels.append(KIND_LABELS["metals"])
        kinds.append("metals")
        cols["initial"].append(m.value)
        cols["basis"].append(m.value)
        cols["contrib"].append(np.full(n_months, m.monthly_purchase))
        cols["access"].append(0)
        cols["gains"].append(0.0)
        cols["flat"].append(0.0)
        cols["sale"].append(m.sale_cost)
        cols["gold"].append(True)
        cols["fee"].append(False)
        cols["ppk"].append(False)
        cols["early"].append(np.nan)
        cols["oki"].append(False)

    rent = rental_schedule(s, n_months)
    main_extra = ikze_refund + rent * hh.rental.reinvest_share
    k = len(kinds)
    del years
    return AccountsPlan(
        labels=labels,
        kinds=kinds,
        initial=np.array(cols["initial"], dtype=float),
        basis=np.array(cols["basis"], dtype=float),
        contributions=np.array(cols["contrib"], dtype=float).reshape(k, n_months),
        access_month=np.array(cols["access"], dtype=int),
        gains_tax=np.array(cols["gains"], dtype=float),
        flat_tax=np.array(cols["flat"], dtype=float),
        sale_cost=np.array(cols["sale"], dtype=float),
        uses_gold=np.array(cols["gold"], dtype=bool),
        pays_fee=np.array(cols["fee"], dtype=bool),
        is_ppk=np.array(cols["ppk"], dtype=bool),
        early_gains_tax=np.array(cols["early"], dtype=float),
        is_oki=np.array(cols["oki"], dtype=bool),
        oki_value_tax=tx.oki_value_tax,
        oki_threshold=tx.oki_exempt_threshold,
        oki_qualifying=tx.oki_qualifying_share,
        main_gains_tax=tx.belka_rate if hh.main_account_taxable else 0.0,
        main_basis=prof.initial_capital * hh.main_account_basis_share,
        main_extra=main_extra,
        capped=capped,
    )


def net_value(x: np.ndarray, basis: np.ndarray, gains_tax: np.ndarray, flat_tax: np.ndarray, sale_cost: np.ndarray) -> np.ndarray:
    """After-tax liquidation value of balances ``x`` (broadcast over the last axis = accounts)."""
    gains = np.maximum(x - basis, 0.0)
    return x * (1.0 - flat_tax - sale_cost) - gains_tax * gains


def total_balance(s: Scenario) -> dict[str, float]:
    """Current gross balances by category (for the balance-sheet overview)."""
    hh = s.household
    out = {"Portfel główny": s.profile.initial_capital}
    if not hh.enabled:
        return out
    for a in hh.accounts:
        if a.enabled and (a.owner == "self" or hh.joint_planning) and a.balance > 0:
            key = f"{KIND_LABELS[a.kind]} · {OWNER_LABELS[a.owner]}"
            out[key] = out.get(key, 0.0) + a.balance
    if hh.metals.value > 0:
        out[KIND_LABELS["metals"]] = hh.metals.value
    return out
