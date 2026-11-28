"""Strona – Majątek i konta: gospodarstwo domowe, IKE/IKZE/PPK/PPE/OKI, kruszce, wynajem."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from components.charts import wealth_by_account_chart
from components.formatting import fmt_age, fmt_money, fmt_pct, polish_frame
from components.kpi import kpi
from components.layout import context_guard, data_badge, page_header
from components.theme import ICONS
from components.validation import polish_errors
from core.accounts import KIND_LABELS, OWNER_LABELS, build_accounts
from core.fire import build_plan
from core.models import HouseholdSettings
from services import analytics, state

state.init()
s = state.active()
page_header("Majątek i konta",
            "Majątek gospodarstwa domowego poza portfelem głównym: konta IKE, IKZE, PPK, PPE, OKI, kruszce i wynajem – "
            "z wiekiem dostępu i podatkami.")
ctx = analytics.get_context(s)
data_badge(ctx)

hh = s.household
tx = hh.taxes
KINDS = ["taxable", "ike", "ikze", "ppk", "ppe", "oki"]
KIND_HELP = {
    "taxable": "Podatek Belki (19%) od nominalnego zysku przy sprzedaży.",
    "ike": f"Bez podatku od {tx.ike_access_age}. roku życia; wcześniejszy zwrot z podatkiem Belki od zysku. "
           f"Limit 2026: {fmt_money(tx.ike_limit)}.",
    "ikze": f"Wpłata obniża PIT; wypłata od {tx.ikze_access_age}. roku życia z podatkiem {fmt_pct(tx.ikze_withdrawal_tax, 0)}. "
            f"Limit 2026: {fmt_money(tx.ikze_limit)} ({fmt_money(tx.ikze_limit_self_employed)} dla JDG).",
    "ppk": f"Od {tx.ppk_access_age}. roku życia: 25% jednorazowo i 75% w 120 ratach – bez podatku. "
           "Wpłata podstawowa: 2% pracownik + 1,5% pracodawca; dopłata roczna państwa.",
    "ppe": f"Składka pracodawcy (do 7% pensji) i dobrowolna własna; wypłata bez podatku od {tx.ppe_access_age}. roku życia.",
    "oki": f"Od 1.01.2027: bez podatku Belki; podatek od wartości {fmt_pct(tx.oki_value_tax, 2)} rocznie od średniej "
           f"wartości ponad {fmt_money(tx.oki_exempt_threshold)} (aktywa kwalifikowane). Wypłaty w dowolnym momencie.",
}

with st.form("household_form"):
    enabled = st.toggle("Uwzględniaj majątek poza portfelem głównym w obliczeniach", value=hh.enabled,
                        help="Po włączeniu wszystkie strony (planer, Monte Carlo, życie po FIRE) liczą majątek po podatkach "
                             "i respektują wiek dostępu do kont emerytalnych.")
    t_house, t_acc, t_metal, t_rent, t_tax = st.tabs(
        [":material/family_restroom: Gospodarstwo", ":material/account_balance: Konta", ":material/diamond: Kruszce",
         ":material/home_work: Wynajem", ":material/gavel: Podatki i limity"])
    with t_house:
        c1, c2, c3 = st.columns(3)
        joint = c1.toggle("Planujemy wspólnie z małżonkiem", value=hh.joint_planning,
                          help="Konta małżonka wchodzą do majątku gospodarstwa; wydatki po FIRE dotyczą całego gospodarstwa. "
                               "Oboje przechodzą na FIRE w tym samym momencie.")
        spouse_age = c2.number_input("Wiek małżonka", 16, 100, hh.spouse_age, 1, disabled=not joint,
                                     help="Decyduje o wieku dostępu do IKE/IKZE/PPK/PPE małżonka.")
        c3.caption("Emeryturę małżonka dodaj jako dochód w Planerze FIRE (wiek liczony według Twojego wieku).")
        c1, c2 = st.columns(2)
        self_emp = c1.toggle("Prowadzę działalność gospodarczą (wyższy limit IKZE)", value=hh.self_employed)
        spouse_emp = c2.toggle("Małżonek prowadzi działalność gospodarczą", value=hh.spouse_self_employed, disabled=not joint)
        st.markdown("**Portfel główny** (kapitał i wpłaty z Planera FIRE)")
        c1, c2 = st.columns(2)
        main_tax = c1.toggle("Zwykły rachunek – podatek Belki od zysku przy wypłacie", value=hh.main_account_taxable)
        basis_opts = [x / 20 for x in range(0, 21)]
        basis_share = c2.select_slider("Udział wpłat w obecnym kapitale (reszta to zysk)", basis_opts,
                                       value=min(basis_opts, key=lambda x: abs(x - hh.main_account_basis_share)),
                                       format_func=lambda x: fmt_pct(x, 0),
                                       help="Podstawa opodatkowania: 100% = kapitał to same wpłaty, bez niezrealizowanego zysku.")
    with t_acc:
        st.caption("Kwoty realne, w dzisiejszych złotych. Wpłaty trwają do przejścia na FIRE. "
                   "„W tym wpłaty” ma znaczenie tylko dla rachunku opodatkowanego (podstawa podatku Belki); puste = całe saldo.")
        acc_df = pd.DataFrame(
            [{"Aktywne": a.enabled, "Rodzaj": a.kind, "Właściciel": a.owner, "Saldo (PLN)": a.balance, "W tym wpłaty (PLN)": a.basis,
              "Wpłata własna (PLN/mies.)": a.monthly_own, "Wpłata pracodawcy (PLN/mies.)": a.monthly_employer} for a in hh.accounts],
            columns=["Aktywne", "Rodzaj", "Właściciel", "Saldo (PLN)", "W tym wpłaty (PLN)", "Wpłata własna (PLN/mies.)",
                     "Wpłata pracodawcy (PLN/mies.)"],
        ).astype({"W tym wpłaty (PLN)": "float64"})
        acc_edit = st.data_editor(
            acc_df, num_rows="dynamic", hide_index=True, width="stretch", key="ui_accounts_editor",
            column_config={
                "Aktywne": st.column_config.CheckboxColumn(default=True),
                "Rodzaj": st.column_config.SelectboxColumn(options=KINDS, required=True, default="ike", format_func=KIND_LABELS.get),
                "Właściciel": st.column_config.SelectboxColumn(options=["self", "spouse"], required=True, default="self",
                                                               format_func=OWNER_LABELS.get),
                "Saldo (PLN)": st.column_config.NumberColumn(min_value=0, step=1000, default=0, format="localized"),
                "W tym wpłaty (PLN)": st.column_config.NumberColumn(min_value=0, step=1000, format="localized"),
                "Wpłata własna (PLN/mies.)": st.column_config.NumberColumn(min_value=0, step=100, default=0, format="localized"),
                "Wpłata pracodawcy (PLN/mies.)": st.column_config.NumberColumn(
                    min_value=0, step=50, default=0, format="localized", help="PPK: 1,5% pensji brutto (+ dodatkowa); PPE: do 7%."),
            },
        )
        with st.expander("Zasady poszczególnych kont", icon=":material/help:"):
            for k in KINDS:
                st.markdown(f"- **{KIND_LABELS[k]}** – {KIND_HELP[k]}")
    with t_metal:
        m = hh.metals
        c1, c2 = st.columns(2)
        metal_val = c1.number_input("Wartość posiadanych kruszców (PLN)", 0.0, 1e9, m.value, 5_000.0, format="%.0f",
                                    help="Złoto i srebro inwestycyjne (sztabki, monety) wycenione po bieżącej cenie skupu.")
        metal_buy = c2.number_input("Zakupy kruszców (PLN/mies., do FIRE)", 0.0, 1e7, m.monthly_purchase, 100.0, format="%.0f")
        spread_opts = [x / 100 for x in range(0, 16)]
        c1, c2, c3 = st.columns(3)
        metal_spread = c1.select_slider("Koszt sprzedaży (spread)", spread_opts, value=min(spread_opts, key=lambda x: abs(x - m.sale_cost)),
                                        format_func=lambda x: fmt_pct(x, 0),
                                        help="Różnica między ceną sprzedaży a skupu – tracona przy upłynnianiu.")
        metal_ret = c2.number_input("Zapasowo: realna stopa zwrotu (%)", -10.0, 15.0, m.manual_real_return * 100, 0.25,
                                    help="Używana tylko, gdy w danych nie ma notowań złota dla okresu estymacji.")
        metal_vol = c3.number_input("Zapasowo: zmienność (%)", 0.0, 60.0, m.manual_volatility * 100, 1.0)
        st.caption("Sprzedaż ruchomości po upływie pół roku od końca miesiąca nabycia nie podlega PIT (art. 10 ust. 1 pkt 8 ustawy o PIT). "
                   "Wartość kruszców zmienia się zgodnie z ceną złota z wybranego źródła danych.")
    with t_rent:
        r = hh.rental
        rent_on = st.toggle("Mam stały dochód z wynajmu", value=r.enabled)
        c1, c2, c3, c4 = st.columns(4)
        rent_gross = c1.number_input("Czynsz najmu (PLN/mies.)", 0.0, 1e7, r.monthly_gross, 100.0, format="%.0f", disabled=not rent_on)
        rent_costs = c2.number_input("Koszty utrzymania (PLN/mies.)", 0.0, 1e7, r.monthly_costs, 50.0, format="%.0f", disabled=not rent_on,
                                     help="Czynsz administracyjny, ubezpieczenie, remonty (średnio).")
        vac_opts = [x / 100 for x in range(0, 51)]
        rent_vac = c3.select_slider("Pustostany", vac_opts, value=min(vac_opts, key=lambda x: abs(x - r.vacancy_rate)),
                                    format_func=lambda x: fmt_pct(x, 0), disabled=not rent_on)
        rent_tax = c4.number_input("Ryczałt od najmu (%)", 0.0, 50.0, r.tax_rate * 100, 0.5, disabled=not rent_on,
                                   help="8,5% do 100 tys. zł przychodu rocznie, 12,5% od nadwyżki.")
        c1, c2, c3 = st.columns(3)
        rent_start = c1.number_input("Od wieku", 0, 120, r.start_age, 1, disabled=not rent_on)
        rent_end = c2.number_input("Do wieku (0 = do końca horyzontu)", 0, 120, r.end_age or 0, 1, disabled=not rent_on)
        reinv_opts = [x / 10 for x in range(0, 11)]
        rent_reinv = c3.select_slider("Reinwestowana część w akumulacji", reinv_opts,
                                      value=min(reinv_opts, key=lambda x: abs(x - r.reinvest_share)),
                                      format_func=lambda x: fmt_pct(x, 0), disabled=not rent_on,
                                      help="Przed FIRE ta część dochodu z najmu trafia do portfela głównego; po FIRE cały dochód "
                                           "netto pokrywa wydatki.")
        st.caption("Dochód z najmu jest traktowany jako realny (czynsz indeksowany inflacją). Wartość nieruchomości nie jest "
                   "częścią kapitału FIRE.")
    with t_tax:
        st.caption("Wartości domyślne obowiązują w 2026 r. Limity IKE/IKZE są traktowane jako stałe realnie.")
        c1, c2, c3 = st.columns(3)
        belka = c1.number_input("Podatek Belki (%)", 0.0, 50.0, tx.belka_rate * 100, 1.0)
        ikze_tax = c2.number_input("Podatek od wypłaty z IKZE (%)", 0.0, 50.0, tx.ikze_withdrawal_tax * 100, 1.0)
        pit = c3.selectbox("Krańcowa stawka PIT (zwrot za IKZE)", [0.12, 0.32, 0.19, 0.0],
                           index=[0.12, 0.32, 0.19, 0.0].index(tx.pit_marginal_rate) if tx.pit_marginal_rate in (0.12, 0.32, 0.19, 0.0) else 0,
                           format_func=lambda x: {0.12: "12% (skala)", 0.32: "32% (skala)", 0.19: "19% (liniowy)", 0.0: "brak ulgi"}[x])
        c1, c2, c3 = st.columns(3)
        refund = c1.toggle("Reinwestuj zwrot podatku za IKZE", value=tx.reinvest_ikze_refund)
        early_ike = c2.toggle("Dopuszczaj wcześniejszy zwrot z IKE (z Belką)", value=tx.allow_early_ike,
                              help="Ostatnia deska ratunku przed 60. rokiem życia: wypłata z IKE z 19% podatkiem od zysku.")
        oki_q = c3.select_slider("OKI: udział aktywów kwalifikowanych", [x / 10 for x in range(0, 11)],
                                 value=min([x / 10 for x in range(0, 11)], key=lambda x: abs(x - tx.oki_qualifying_share)),
                                 format_func=lambda x: fmt_pct(x, 0),
                                 help="Polskie akcje, obligacje i fundusze z ≥70% takich aktywów korzystają z kwoty wolnej 100 tys. zł. "
                                      "Zagraniczne ETF-y – nie.")
        c1, c2, c3, c4 = st.columns(4)
        ike_lim = c1.number_input("Limit IKE (PLN/rok)", 0.0, 1e6, tx.ike_limit, 100.0, format="%.0f")
        ikze_lim = c2.number_input("Limit IKZE (PLN/rok)", 0.0, 1e6, tx.ikze_limit, 100.0, format="%.0f")
        ikze_lim_se = c3.number_input("Limit IKZE – JDG (PLN/rok)", 0.0, 1e6, tx.ikze_limit_self_employed, 100.0, format="%.0f")
        ppk_state = c4.number_input("Dopłata roczna PPK (PLN)", 0.0, 10_000.0, tx.ppk_state_annual, 10.0, format="%.0f")
        c1, c2, c3, c4 = st.columns(4)
        ike_age = c1.number_input("Wiek dostępu IKE", 40, 80, tx.ike_access_age, 1)
        ikze_age = c2.number_input("Wiek dostępu IKZE", 40, 80, tx.ikze_access_age, 1)
        ppk_age = c3.number_input("Wiek wypłaty PPK", 40, 80, tx.ppk_access_age, 1)
        ppe_age = c4.number_input("Wiek wypłaty PPE", 40, 80, tx.ppe_access_age, 1)
        c1, c2 = st.columns(2)
        oki_rate = c1.number_input("OKI: podatek od wartości (%)", 0.0, 5.0, tx.oki_value_tax * 100, 0.05, format="%.2f")
        oki_thr = c2.number_input("OKI: kwota wolna (PLN)", 0.0, 1e7, tx.oki_exempt_threshold, 5_000.0, format="%.0f")

    submitted = st.form_submit_button("Zapisz majątek i konta", type="primary", icon=ICONS["save"])

if submitted:
    try:
        accounts = []
        for _, row in acc_edit.iterrows():
            if pd.isna(row["Rodzaj"]):
                continue
            basis = row["W tym wpłaty (PLN)"]
            accounts.append({
                "enabled": bool(row["Aktywne"]) if not pd.isna(row["Aktywne"]) else True,
                "kind": row["Rodzaj"], "owner": row["Właściciel"] if not pd.isna(row["Właściciel"]) else "self",
                "balance": float(row["Saldo (PLN)"] or 0), "basis": None if pd.isna(basis) else float(basis),
                "monthly_own": float(row["Wpłata własna (PLN/mies.)"] or 0),
                "monthly_employer": float(row["Wpłata pracodawcy (PLN/mies.)"] or 0)})
        new_hh = HouseholdSettings.model_validate({
            "enabled": enabled, "joint_planning": joint, "spouse_age": int(spouse_age), "self_employed": self_emp,
            "spouse_self_employed": spouse_emp, "main_account_taxable": main_tax, "main_account_basis_share": basis_share,
            "accounts": accounts,
            "metals": {"value": metal_val, "monthly_purchase": metal_buy, "sale_cost": metal_spread,
                       "manual_real_return": metal_ret / 100, "manual_volatility": metal_vol / 100},
            "rental": {"enabled": rent_on, "monthly_gross": rent_gross, "monthly_costs": rent_costs, "vacancy_rate": rent_vac,
                       "tax_rate": rent_tax / 100, "start_age": int(rent_start), "end_age": int(rent_end) or None,
                       "reinvest_share": rent_reinv},
            "taxes": {**tx.model_dump(), "belka_rate": belka / 100, "ikze_withdrawal_tax": ikze_tax / 100, "pit_marginal_rate": pit,
                      "reinvest_ikze_refund": refund, "allow_early_ike": early_ike, "oki_qualifying_share": oki_q,
                      "ike_limit": ike_lim, "ikze_limit": ikze_lim, "ikze_limit_self_employed": ikze_lim_se,
                      "ppk_state_annual": ppk_state, "ike_access_age": int(ike_age), "ikze_access_age": int(ikze_age),
                      "ppk_access_age": int(ppk_age), "ppe_access_age": int(ppe_age), "oki_value_tax": oki_rate / 100,
                      "oki_exempt_threshold": oki_thr},
        })
        new = s.model_copy(deep=True)
        new.household = new_hh
        state.save(new, "Zapisano majątek i konta.")
        st.rerun()
    except (ValidationError, ValueError, KeyError) as exc:
        for msg in (polish_errors(exc) if isinstance(exc, ValidationError) else [str(exc)]):
            st.error(msg, icon=":material/error:")

if not hh.enabled:
    st.info("Majątek poza portfelem głównym nie jest uwzględniany. Włącz przełącznik powyżej i zapisz, aby planer, Monte Carlo "
            "i analiza wypłat uwzględniały konta, kruszce i wynajem.", icon=ICONS["info"])
    st.stop()
if not context_guard(ctx):
    st.stop()

# ----------------------------------------------------------------------------- results (deterministic)
proj = analytics.deterministic(ctx)
res = proj.result
plan = build_plan(s, ctx.stats["geometric_real_net"])
ap = plan.accounts
prof = s.profile
ages = res.ages()
labels = res.bucket_labels
bw = pd.DataFrame(res.bucket_wealth[0].T, columns=labels)
bw = bw.loc[:, (bw.abs() > 1).any(axis=0)]
now = res.bucket_wealth[0, :, 0]
locked_now = sum(now[2 + k] for k in range(ap.n) if ap.access_month[k] > 0 or ap.is_ppk[k])
fire_m = (prof.fire_age - prof.current_age) * 12
access = {}
for k in range(ap.n):
    if ap.access_month[k] > 0 and ap.kinds[k] not in ("taxable", "oki", "metals"):
        a_age = prof.current_age + ap.access_month[k] / 12
        access.setdefault(f"{ap.labels[k]} – {fmt_age(a_age)}", a_age)
locked_at_fire = [k for k in range(ap.n) if ap.access_month[k] > fire_m and ap.kinds[k] in ("ike", "ikze", "ppk", "ppe")]
first_unlock = min((ap.access_month[k] for k in locked_at_fire), default=None)

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Majątek netto dziś", fmt_money(now.sum(), compact=True), icon=":material/account_balance:",
        help="Po odliczeniu podatków, które trzeba by zapłacić przy wypłacie (Belka od zysku, 10% IKZE, spread kruszców).")
with c2:
    kpi("W tym zablokowane", fmt_money(locked_now, compact=True), delta=fmt_pct(locked_now / now.sum() if now.sum() else 0, 0) + " majątku",
        delta_color="off", help="IKE, IKZE, PPK i PPE przed wiekiem dostępu.")
with c3:
    bridge = None
    if first_unlock is not None:
        gap = np.maximum(plan.spending - plan.other_income - plan.barista_income, 0.0)[fire_m:first_unlock]
        bridge = float(gap.sum())
    kpi("Pomost do odblokowania kont", fmt_money(bridge, compact=True) if bridge is not None else "—",
        delta=(f"FIRE {prof.fire_age} → {fmt_age(prof.current_age + first_unlock / 12)}" if first_unlock is not None else "brak blokad po FIRE"),
        delta_color="off", help="Suma realnych wydatków (po odjęciu dochodów) między FIRE a pierwszym odblokowaniem konta emerytalnego, "
                                "bez stóp zwrotu – tyle trzeba mieć w płynnych aktywach.")
with c4:
    fire_y = min(prof.fire_age - prof.current_age, len(ages) - 1)
    liquid_fire = float(res.bucket_wealth[0, 0, fire_y] + res.bucket_wealth[0, 1, fire_y]
                        + sum(res.bucket_wealth[0, 2 + k, fire_y] for k in range(ap.n) if k not in locked_at_fire and not ap.is_ppk[k]))
    ok = bridge is None or liquid_fire >= bridge
    kpi("Płynne środki w chwili FIRE", fmt_money(liquid_fire, compact=True),
        delta=("wystarczą na pomost" if ok else f"brakuje {fmt_money(bridge - liquid_fire, compact=True)}"),
        delta_color="off" if ok else "inverse",
        help="Portfel główny, gotówka, OKI, rachunki opodatkowane i kruszce na ścieżce deterministycznej.")

for msg in ap.capped:
    st.warning(msg, icon=ICONS["warning"])
short_years = [int(a) for a, flag in zip(ages[:-1], res.consumption_year[0] < 12 * s.withdrawal.spending_floor_share * plan.spending - 1)
               if flag]
if res.illiquid_shortfall is not None and res.illiquid_shortfall[0] and short_years:
    st.error(f"Luka płynnościowa: na ścieżce deterministycznej wydatki nie są pokryte w wieku {short_years[0]}–{short_years[-1]} lat, "
             "choć część majątku jest zablokowana na kontach emerytalnych. Rozważ większe wpłaty na rachunek opodatkowany lub OKI, "
             "późniejszy FIRE albo dopuszczenie wcześniejszego zwrotu z IKE.", icon=":material/lock_clock:")
elif short_years:
    st.warning(f"Wydatki nie są w pełni pokryte w wieku {short_years[0]}–{short_years[-1]} lat (ścieżka deterministyczna).",
               icon=ICONS["warning"])

with st.container(border=True):
    st.markdown("**Majątek po podatkach według kont** (ścieżka deterministyczna, ceny stałe)")
    st.plotly_chart(wealth_by_account_chart(ages, bw, prof.fire_age, access), key="hh_stack", config={"displaylogo": False})
    gold_src = (f"złoto: {ctx.gold_source}, stopa geometryczna {fmt_pct(ctx.stats['gold_geometric_nominal'], 2)} nominalnie"
                if ctx.gold is not None else
                f"złoto: brak notowań – założenie ręczne {fmt_pct(hh.metals.manual_real_return, 2)} realnie")
    acc_txt = "; ".join(sorted(access, key=access.get))
    st.caption("Każde konto pokazane po odliczeniu podatku należnego przy wypłacie. Linie pionowe: przejście na FIRE (pomarańczowa) "
               "oraz odblokowanie kont emerytalnych (szare" + (f": {acc_txt}" if acc_txt else "") + ")."
               + (f" Wycena kruszców – {gold_src}." if analytics.needs_gold(s) else ""))

rows = []
for k in range(ap.n):
    rows.append({"Konto": ap.labels[k], "Saldo dziś": ap.initial[k],
                 "Wpłaty (mies.)": ap.contributions[k, 0] if ap.contributions.shape[1] else 0.0,
                 "Dostęp od (Twój wiek)": fmt_age(prof.current_age + ap.access_month[k] / 12) if ap.access_month[k] > 0 else "od razu",
                 "Podatek przy wypłacie": {"taxable": f"Belka {fmt_pct(ap.gains_tax[k], 0)} od zysku",
                                           "ike": "0% (wcześniej Belka od zysku)" if np.isfinite(ap.early_gains_tax[k]) else "0%",
                                           "ikze": fmt_pct(ap.flat_tax[k], 0) + " od kwoty",
                                           "ppk": "0% (25% + 120 rat)", "ppe": "0%",
                                           "oki": f"{fmt_pct(tx.oki_value_tax, 2)} rocznie od wartości",
                                           "metals": f"0%, spread {fmt_pct(ap.sale_cost[k], 0)}"}[ap.kinds[k]],
                 "Wartość w chwili FIRE": res.bucket_wealth[0, 2 + k, fire_y],
                 "Wartość na końcu": res.bucket_wealth[0, 2 + k, -1]})
if rows:
    tbl = pd.DataFrame(rows)
    st.dataframe(polish_frame(tbl, money=["Saldo dziś", "Wpłaty (mies.)", "Wartość w chwili FIRE", "Wartość na końcu"],
                              ),
                 hide_index=True, width="stretch")
    st.caption("Wpłaty miesięczne zawierają składki pracodawcy i dopłatę państwa (PPK) po zastosowaniu limitów. "
               "Wartości po podatkach, realnie.")
if hh.rental.enabled:
    st.caption(f"Dochód z wynajmu netto: {fmt_money(hh.rental.monthly_net)}/mies. "
               f"(czynsz {fmt_money(hh.rental.monthly_gross)} × (1 − pustostany {fmt_pct(hh.rental.vacancy_rate, 0)}) × "
               f"(1 − ryczałt {fmt_pct(hh.rental.tax_rate, 1)}) − koszty {fmt_money(hh.rental.monthly_costs)}).")

with st.expander("Kolejność wypłat i uproszczenia modelu", icon=":material/menu_book:"):
    st.markdown(
        "**Kolejność wypłat po FIRE:** bufor gotówkowy (po roku ze stratą) → portfel główny → gotówka → rachunki opodatkowane → "
        "OKI → kruszce → IKE → PPE → IKZE; PPK wypłaca się automatycznie (25% + 120 rat) do portfela głównego od 60. roku życia. "
        "Przed wiekiem dostępu IKE może być użyte jako ostatnie (z podatkiem Belki od zysku), jeśli to dopuszczono.\n\n"
        "**Uproszczenia:** wszystkie konta inwestują w ten sam portfel (poza kruszcami); podstawa podatku Belki jest nominalna "
        "(inflacja zmniejsza ją realnie); składka pracodawcy do PPK/PPE nie zwiększa PIT pracownika; limity IKE/IKZE stałe realnie; "
        "OKI – podatek od średniej z wartości na koniec miesięcy, kwota wolna stała realnie (ustawa: stała do 2029 r., potem indeksowana); "
        "oboje małżonkowie przechodzą na FIRE jednocześnie; nie modelujemy wcześniejszej wypłaty z IKZE ani PPK."
    )
