"""Strona 2 – Planer FIRE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from components.charts import accumulation_chart
from components.formatting import nearest, fmt_age, fmt_money, fmt_pct, polish_frame, years_pl
from components.kpi import kpi
from components.layout import context_guard, data_badge, page_header
from components.theme import ICONS
from components.validation import polish_errors
from core.fire import MODE_LABELS, pv_required_capital
from core.models import FinancialEvent, FireMode, IncomeSource, RetirementRule
from services import analytics, state

state.init()
s = state.active()
page_header("Planer FIRE", "Zdefiniuj cel niezależności finansowej i sprawdź deterministyczną ścieżkę dojścia do niego.")
ctx = analytics.get_context(s)
data_badge(ctx)

SWR_OPTIONS = [round(x, 4) for x in np.arange(0.025, 0.0601, 0.0025)]
MODE_HELP = {
    FireMode.TRADITIONAL: "Kapitał pokrywa pełne wydatki: C = E / SWR.",
    FireMode.LEAN: "Minimalistyczny wariant: kapitał finansuje tylko wydatki podstawowe.",
    FireMode.FAT: "Wyższy standard życia oraz bufor gotówkowy na złe lata rynkowe.",
    FireMode.COAST: "Po zgromadzeniu kwoty Coast wpłaty ustają – portfel sam rośnie do celu.",
    FireMode.BARISTA: "Po FIRE praca w niepełnym wymiarze pokrywa część wydatków do wskazanego wieku.",
}

prof, fire = s.profile, s.fire

with st.form("planner_form"):
    t_profile, t_macro, t_mode, t_events = st.tabs(
        [":material/person: Profil i budżet", ":material/percent: Założenia", ":material/flag: Tryb FIRE",
         ":material/event: Zdarzenia i dochody"]
    )
    with t_profile:
        c1, c2, c3 = st.columns(3)
        current_age = c1.number_input("Obecny wiek", 16, 90, prof.current_age, 1)
        fire_age = c2.number_input("Docelowy wiek FIRE", 17, 100, prof.fire_age, 1)
        horizon = c3.number_input("Horyzont życia (wiek)", 40, 120, prof.life_horizon_age, 1,
                                  help="Wiek, do którego kapitał musi wystarczyć. Konserwatywnie: 95–100 lat.")
        c1, c2, c3 = st.columns(3)
        capital = c1.number_input("Kapitał początkowy (PLN)", 0.0, 1e9, prof.initial_capital, 10_000.0, format="%.0f")
        income = c2.number_input("Dochód netto miesięcznie (PLN)", 0.0, 1e7, prof.monthly_net_income, 500.0, format="%.0f")
        expenses = c3.number_input("Wydatki miesięczne przed FIRE (PLN)", 0.0, 1e7, prof.monthly_expenses, 500.0, format="%.0f")
        c1, c2 = st.columns(2)
        contrib = c1.number_input("Miesięczna wpłata inwestycyjna (PLN)", 0.0, 1e7, prof.monthly_contribution, 250.0, format="%.0f",
                                  help="Wpłata nie może trwale przekraczać nadwyżki dochodu nad wydatkami – silnik ją do niej ogranicza.")
        ret_exp = c2.number_input("Wydatki miesięczne po FIRE (PLN, dzisiejsze ceny)", 0.0, 1e7,
                                  prof.retirement_monthly_expenses, 250.0, format="%.0f")
    with t_macro:
        c1, c2, c3 = st.columns(3)
        inflation = c1.number_input("Inflacja (rocznie, %)", -5.0, 30.0, prof.inflation * 100, 0.1,
                                    help="Wszystkie kwoty podajesz w dzisiejszych złotych; inflacja służy do przeliczeń realne ↔ nominalne.")
        inc_g = c2.number_input("Realny wzrost dochodów (rocznie, %)", -10.0, 20.0, prof.income_growth_real * 100, 0.25,
                                help="Wzrost ponad inflację. Zwiększa możliwą nadwyżkę na inwestycje.")
        con_g = c3.number_input("Realna waloryzacja wpłat (rocznie, %)", -10.0, 20.0, prof.contribution_growth_real * 100, 0.25,
                                help="O ile co roku rosną wpłaty ponad inflację (0% = stała siła nabywcza wpłaty).")
        c1, c2 = st.columns(2)
        swr = c1.select_slider("Początkowa stopa wypłat (SWR)", SWR_OPTIONS, value=min(SWR_OPTIONS, key=lambda x: abs(x - fire.swr)),
                               format_func=lambda x: fmt_pct(x, 2),
                               help="Safe Withdrawal Rate – bezpieczna stopa wypłat. Określa wymagany kapitał: C = roczne wydatki / SWR.")
        fee = c2.number_input("Koszty zarządzania portfelem (rocznie, %)", 0.0, 5.0, s.portfolio.management_fee * 100, 0.05,
                              help="TER funduszy + opłaty platformy, potrącane co miesiąc od wartości portfela.")
    with t_mode:
        mode = st.segmented_control("Wariant FIRE", list(FireMode), default=fire.mode, format_func=lambda m: MODE_LABELS[m],
                                    required=True)
        st.caption(MODE_HELP[mode] if mode else "")
        c1, c2 = st.columns(2)
        rule = c1.radio("Moment przejścia na FIRE", list(RetirementRule), index=list(RetirementRule).index(fire.retirement_rule),
                        format_func={RetirementRule.FIXED_AGE: "W docelowym wieku (niezależnie od kapitału)",
                                     RetirementRule.WHEN_TARGET_REACHED: "Gdy kapitał osiągnie cel (najpóźniej w wieku maks.)"}.get,
                        help="Wpływa na symulacje Monte Carlo: w drugim wariancie każda ścieżka ma własny wiek FIRE.")
        max_work = c2.number_input("Maksymalny wiek pracy", 18, 100, fire.max_work_age, 1)
        c1, c2, c3 = st.columns(3)
        lean_share = c1.select_slider("Lean: udział wydatków podstawowych", [x / 20 for x in range(6, 21)],
                                      value=nearest([x / 20 for x in range(6, 21)], fire.lean_essential_share), format_func=lambda x: fmt_pct(x, 0),
                                      disabled=mode != FireMode.LEAN)
        fat_extra = c2.select_slider("Fat: dodatkowe wydatki (ponad bazę)", [x / 20 for x in range(0, 31)],
                                     value=nearest([x / 20 for x in range(0, 31)], fire.fat_extra_share), format_func=lambda x: fmt_pct(x, 0),
                                     disabled=mode != FireMode.FAT)
        fat_buf = c3.select_slider("Fat: bufor gotówkowy (lata wydatków)", [x / 2 for x in range(0, 11)],
                                   value=nearest([x / 2 for x in range(0, 11)], fire.fat_cash_buffer_years), format_func=lambda x: years_pl(x),
                            disabled=mode != FireMode.FAT,
                            help="Rezerwa przenoszona do gotówki w chwili FIRE; po roku ze stratą wypłaty pochodzą najpierw z bufora.")
        c1, c2 = st.columns(2)
        bar_inc = c1.number_input("Barista: dochód z pracy częściowej (PLN/mies.)", 0.0, 1e6, fire.barista_monthly_income, 250.0,
                                  format="%.0f", disabled=mode != FireMode.BARISTA)
        bar_end = c2.number_input("Barista: praca do wieku", 18, 100, fire.barista_end_age, 1, disabled=mode != FireMode.BARISTA)
    with t_events:
        st.markdown("**Dodatkowe zdarzenia finansowe** (kwoty realne, w dzisiejszych złotych)")
        ev_df = pd.DataFrame(
            [{"Aktywne": e.enabled, "Nazwa": e.name, "Rodzaj": e.category, "Wiek": e.age, "Kwota (PLN)": abs(e.amount),
              "Cykliczne": e.recurring, "Do wieku": e.end_age} for e in prof.events],
            columns=["Aktywne", "Nazwa", "Rodzaj", "Wiek", "Kwota (PLN)", "Cykliczne", "Do wieku"],
        ).astype({"Do wieku": "float64"})
        ev_edit = st.data_editor(
            ev_df, num_rows="dynamic", hide_index=True, key="ui_events_editor", width="stretch",
            column_config={
                "Aktywne": st.column_config.CheckboxColumn(default=True),
                "Nazwa": st.column_config.TextColumn(default="Zdarzenie", required=True),
                "Rodzaj": st.column_config.SelectboxColumn(options=["inflow", "outflow"], default="outflow", required=True,
                                                           format_func={"inflow": "Wpływ", "outflow": "Wydatek"}.get),
                "Wiek": st.column_config.NumberColumn(min_value=0, max_value=120, step=1, default=40, required=True),
                "Kwota (PLN)": st.column_config.NumberColumn(min_value=0, step=1000, format="localized", default=0, required=True),
                "Cykliczne": st.column_config.CheckboxColumn(help="Zdarzenie powtarzane co roku do wskazanego wieku", default=False),
                "Do wieku": st.column_config.NumberColumn(min_value=0, max_value=120, step=1,
                                                          help="Dla zdarzeń cyklicznych; puste = do końca horyzontu"),
            },
        )
        st.markdown("**Dodatkowe źródła dochodu po FIRE** (np. emerytura publiczna, wynajem)")
        inc_df = pd.DataFrame(
            [{"Aktywne": i.enabled, "Nazwa": i.name, "Kwota miesięczna (PLN)": i.monthly_amount, "Od wieku": i.start_age,
              "Do wieku": i.end_age} for i in fire.incomes],
            columns=["Aktywne", "Nazwa", "Kwota miesięczna (PLN)", "Od wieku", "Do wieku"],
        ).astype({"Do wieku": "float64"})
        inc_edit = st.data_editor(
            inc_df, num_rows="dynamic", hide_index=True, key="ui_incomes_editor", width="stretch",
            column_config={
                "Aktywne": st.column_config.CheckboxColumn(default=True),
                "Nazwa": st.column_config.TextColumn(default="Dochód", required=True),
                "Kwota miesięczna (PLN)": st.column_config.NumberColumn(min_value=0, step=100, format="localized", default=0, required=True),
                "Od wieku": st.column_config.NumberColumn(min_value=0, max_value=120, step=1, default=65, required=True),
                "Do wieku": st.column_config.NumberColumn(min_value=0, max_value=120, step=1, help="Puste = dożywotnio"),
            },
        )
    submitted = st.form_submit_button("Zatwierdź parametry planu", type="primary", icon=ICONS["save"])

if submitted:
    def _int_or_none(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else int(v)

    try:
        events = [
            FinancialEvent(enabled=bool(r["Aktywne"]), name=str(r["Nazwa"] or "Zdarzenie"), category=r["Rodzaj"] or "outflow",
                           age=int(r["Wiek"]), amount=float(r["Kwota (PLN)"] or 0), recurring=bool(r["Cykliczne"]),
                           end_age=_int_or_none(r["Do wieku"]))
            for _, r in ev_edit.dropna(subset=["Wiek"]).iterrows()
        ]
        incomes = [
            IncomeSource(enabled=bool(r["Aktywne"]), name=str(r["Nazwa"] or "Dochód"), monthly_amount=float(r["Kwota miesięczna (PLN)"] or 0),
                         start_age=int(r["Od wieku"]), end_age=_int_or_none(r["Do wieku"]))
            for _, r in inc_edit.dropna(subset=["Od wieku"]).iterrows()
        ]
        new = s.model_copy(deep=True)
        new.profile = new.profile.model_validate({
            **new.profile.model_dump(), "current_age": int(current_age), "fire_age": int(fire_age), "life_horizon_age": int(horizon),
            "initial_capital": capital, "monthly_net_income": income, "monthly_expenses": expenses, "monthly_contribution": contrib,
            "retirement_monthly_expenses": ret_exp, "inflation": inflation / 100, "income_growth_real": inc_g / 100,
            "contribution_growth_real": con_g / 100, "events": [e.model_dump() for e in events]})
        new.fire = new.fire.model_validate({
            **new.fire.model_dump(), "mode": mode, "swr": swr, "retirement_rule": rule, "max_work_age": int(max_work),
            "lean_essential_share": lean_share, "fat_extra_share": fat_extra, "fat_cash_buffer_years": fat_buf,
            "barista_monthly_income": bar_inc, "barista_end_age": int(bar_end), "incomes": [i.model_dump() for i in incomes]})
        new.portfolio = new.portfolio.model_copy(update={"management_fee": fee / 100})
        state.save(new, "Zapisano parametry planu FIRE.")
        st.rerun()
    except ValidationError as exc:
        for msg in polish_errors(exc):
            st.error(msg, icon=":material/error:")

if not context_guard(ctx):
    st.stop()

# ----------------------------------------------------------------------------- results (committed scenario)
proj = analytics.deterministic(ctx)
target = proj.target
current = float(proj.table["wealth_real"].iloc[0])  # after-tax, incl. household accounts when enabled
progress = current / target.capital if target.capital > 0 else 1.0
years_to = (proj.fire_age - s.profile.current_age) if proj.fire_age is not None else None
contrib_share = s.profile.monthly_contribution / s.profile.monthly_net_income if s.profile.monthly_net_income else float("nan")

st.subheader("Cel i ścieżka deterministyczna", anchor=False)
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Wymagany kapitał FIRE", fmt_money(target.capital, compact=True), help=target.explanation, icon=":material/flag:")
with c2:
    kpi("Stopień realizacji celu", fmt_pct(min(progress, 9.99), 1),
        delta=f"brakuje {fmt_money(max(target.capital - current, 0), compact=True)}" if progress < 1 else "cel osiągnięty",
        delta_color="off", chart=proj.table["wealth_real"].iloc[: s.profile.fire_age - s.profile.current_age + 1].tolist())
with c3:
    kpi("Prognozowany wiek FIRE", fmt_age(proj.fire_age) if proj.fire_age else "poza horyzontem pracy",
        delta=(f"{years_pl(round(years_to, 1))} do celu" if years_to is not None else None), delta_color="off",
        help="Pierwszy moment, w którym realna wartość portfela na ścieżce deterministycznej (stopa geometryczna) osiąga cel.")
with c4:
    kpi("Stopa oszczędności", fmt_pct(contrib_share, 1), help="Miesięczna wpłata / dochód netto.",
        delta=f"realna stopa netto {fmt_pct(ctx.stats['geometric_real_net'], 2)}", delta_color="off")

if proj.capped_contributions:
    st.warning("Planowana wpłata przekracza w niektórych latach nadwyżkę dochodu nad wydatkami – silnik ograniczył ją do tej nadwyżki.",
               icon=ICONS["warning"])

with st.container(border=True):
    head, toggle_col = st.columns([3, 1], vertical_alignment="center")
    head.markdown("**Ścieżka akumulacji i dekumulacji kapitału** (stała stopa geometryczna, bez losowości)")
    nominal = toggle_col.toggle("Wartości nominalne", key="ui_planner_nominal",
                                help="Wartości realne są w dzisiejszych złotych. Nominalne = realne × (1 + inflacja)^t.")
    st.plotly_chart(accumulation_chart(proj.table, target.capital, proj.fire_age, nominal, s.profile.fire_age),
                    key="planner_path", config={"displaylogo": False})
    st.caption(
        f"Założenia: nominalna oczekiwana stopa geometryczna portfela {fmt_pct(ctx.stats['geometric_nominal'], 2)}, "
        f"inflacja {fmt_pct(s.profile.inflation, 1)}, koszty {fmt_pct(s.portfolio.management_fee, 2)} rocznie, "
        f"strategia wypłat zgodna z ustawieniami strony „Życie po FIRE”."
    )

left, right = st.columns([1, 1])
with left:
    with st.container(border=True):
        st.markdown(f"**Struktura celu – {MODE_LABELS[s.fire.mode]}**")
        rows = [("Wydatki po FIRE (mies.)", fmt_money(target.monthly_spending)),
                ("Kwota pokrywana z portfela (rocznie)", fmt_money(target.annual_need_from_portfolio)),
                ("Stopa wypłat (SWR)", fmt_pct(s.fire.swr, 2))]
        rows += [(k, fmt_money(v)) for k, v in target.components.items()]
        pv = pv_required_capital(s, ctx.stats["geometric_real_net"])
        rows.append(("Kapitał wg wartości bieżącej wypłat do horyzontu", fmt_money(pv)))
        st.dataframe(pd.DataFrame(rows, columns=["Pozycja", "Wartość"]), hide_index=True, width="stretch")  # already formatted
        st.caption("Wartość bieżąca uwzględnia emeryturę i inne dochody oraz skończony horyzont – porównaj z regułą SWR.")
        if s.fire.mode == FireMode.COAST and target.coast_number_today is not None:
            reached = current >= target.coast_number_today
            (st.success if reached else st.info)(
                f"Kwota Coast FIRE dziś: **{fmt_money(target.coast_number_today)}** – "
                + ("osiągnięta, wpłaty mogą ustać." if reached else "jeszcze nieosiągnięta."))
with right:
    with st.container(border=True):
        st.markdown("**Metodologia wyznaczenia celu**")
        st.latex(r"C_{FIRE} = \frac{12 \cdot E}{SWR}")
        if s.fire.mode == FireMode.FAT:
            st.latex(r"C_{Fat} = \frac{12E(1+m)}{SWR} + k \cdot 12E(1+m)")
        if s.fire.mode == FireMode.COAST:
            st.latex(r"C_{Coast}(t) = \frac{C_{FIRE}}{(1+g_{real,netto})^{T-t}}")
        if s.fire.mode == FireMode.BARISTA:
            st.latex(r"C_{Barista} = \frac{12\,(E - I_{part})}{SWR}")
        if s.fire.mode == FireMode.LEAN:
            st.latex(r"C_{Lean} = \frac{12\, s \cdot E}{SWR}")
        st.latex(r"W_{nom,t} = W_{real,t}\,(1+\pi)^t \qquad r_{real} = \frac{1+r_{nom}}{1+\pi}-1")
        with st.expander("Jak czytać wyniki", icon=":material/help:"):
            st.markdown(
                "- Wszystkie kwoty wprowadzane są **realnie** (w dzisiejszych złotych); inflacja jest naliczana raz.\n"
                "- Ścieżka deterministyczna używa stopy **geometrycznej** (mediany), a nie średniej arytmetycznej – "
                "średnia arytmetyczna zawyżałaby długoterminowy wynik.\n"
                "- Wpłaty, wypłaty i rebalancing następują na początku miesiąca, potem nalicza się stopę zwrotu i koszty.\n"
                "- Prawdopodobieństwa sukcesu znajdziesz w Laboratorium ryzyka – ta strona nie zawiera losowości."
            )

with st.expander("Tabela roczna ścieżki deterministycznej", icon=":material/table_rows:"):
    show = proj.table.rename(columns={"age": "Wiek", "wealth_real": "Majątek realny", "wealth_nominal": "Majątek nominalny",
                                      "contributions": "Wpłaty (realnie)", "withdrawals": "Wypłaty (realnie)",
                                      "target_real": "Cel FIRE", "phase": "Faza"})
    money_cols = ["Majątek realny", "Majątek nominalny", "Wpłaty (realnie)", "Wypłaty (realnie)", "Cel FIRE"]
    st.dataframe(polish_frame(show, money=money_cols), hide_index=True, width="stretch")
    st.download_button("Pobierz ścieżkę (CSV)", show.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       "fire_lab_sciezka_deterministyczna.csv", "text/csv", icon=ICONS["download"])

# A deliberate user action related to reaching FIRE → balloons exactly once per scenario.
if progress >= 1.0:
    st.success("Twój obecny kapitał pokrywa wymagany kapitał FIRE.", icon=ICONS["celebrate"])
    if s.id not in st.session_state["celebrated"]:
        if st.button("Ogłoś niezależność finansową", type="primary", icon=ICONS["celebrate"]):
            st.session_state["celebrated"].add(s.id)
            st.balloons()

c1, c2, c3 = st.columns(3)
c1.page_link("views/portfolio_builder.py", label="Dobierz portfel", icon=ICONS["portfolio"])
c2.page_link("views/risk_lab.py", label="Sprawdź prawdopodobieństwo sukcesu", icon=ICONS["risk"])
c3.page_link("views/post_fire.py", label="Zaplanuj wypłaty po FIRE", icon=ICONS["post_fire"])
