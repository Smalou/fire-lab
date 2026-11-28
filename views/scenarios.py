"""Strona 7 – Porównywarka scenariuszy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from components.formatting import fmt_age, fmt_money, fmt_num, fmt_pct, polish_frame
from components.layout import page_header
from components.theme import NAVY, ORANGE, SERIES, TEAL, style
from components.validation import polish_errors
from core.fire import MODE_LABELS
from core.models import FireMode, Scenario
from services import analytics, state
from services.scenarios import ImportErrorPL, clone, default_scenarios, from_json, to_json

state.init()
page_header("Porównywarka scenariuszy", "Twórz, zapisuj i porównuj warianty planu: szybciej do FIRE czy bezpieczniej po FIRE?")
if state.user().persisted:
    st.caption(f":material/cloud_done: Scenariusze profilu **{state.user().display_name}** zapisują się automatycznie w bazie danych.")
else:
    st.warning("Pracujesz jako gość: scenariusze są przechowywane wyłącznie w pamięci bieżącej sesji i mogą zostać utracone "
               "po zamknięciu karty lub restarcie aplikacji. Zaloguj się albo eksportuj je do pliku JSON.",
               icon=":material/save:")

sc = state.scenarios()
active = state.active()


@st.dialog("Nowy scenariusz", icon=":material/add:")
def new_dialog() -> None:
    with st.form("new_scenario"):
        name = st.text_input("Nazwa", "Nowy scenariusz", max_chars=80)
        desc = st.text_area("Opis", "")
        base = st.radio("Punkt wyjścia", ["active", "balanced"],
                        format_func={"active": f"Kopia aktywnego („{active.name}”)", "balanced": "Domyślny zrównoważony"}.get)
        if st.form_submit_button("Utwórz", type="primary", icon=":material/add:"):
            src = active if base == "active" else default_scenarios()[0]
            try:
                s = clone(src, name)
                s.description = desc
                state.add(s)
                st.toast(f"Utworzono scenariusz „{name}”.", icon=":material/check_circle:")
                st.rerun()
            except ValidationError as exc:
                st.error("; ".join(polish_errors(exc)))


@st.dialog("Zmień nazwę scenariusza", icon=":material/edit:")
def rename_dialog(sid: str) -> None:
    s = sc[sid]
    with st.form("rename_form"):
        name = st.text_input("Nowa nazwa", s.name, max_chars=80)
        desc = st.text_area("Opis", s.description)
        if st.form_submit_button("Zapisz", type="primary"):
            if any(o.name == name.strip() and o.id != sid for o in sc.values()):
                st.error("Scenariusz o tej nazwie już istnieje.")
                return
            new = s.model_copy(deep=True)
            new.name, new.description = name, desc
            try:
                state.save(new, "Zmieniono nazwę scenariusza.")
                st.rerun()
            except ValidationError as exc:
                st.error("; ".join(polish_errors(exc)))


@st.dialog("Edycja parametrów scenariusza", width="large", icon=":material/tune:")
def edit_dialog(sid: str) -> None:
    s = sc[sid]
    p = s.profile
    with st.form("edit_form"):
        c1, c2, c3 = st.columns(3)
        cur = c1.number_input("Obecny wiek", 16, 90, p.current_age)
        fa = c2.number_input("Wiek FIRE", 17, 100, p.fire_age)
        hz = c3.number_input("Horyzont", 40, 120, p.life_horizon_age)
        c1, c2, c3 = st.columns(3)
        cap = c1.number_input("Kapitał (PLN)", 0.0, 1e10, p.initial_capital, 10_000.0, format="%.0f")
        con = c2.number_input("Wpłata mies. (PLN)", 0.0, 1e7, p.monthly_contribution, 250.0, format="%.0f")
        exp = c3.number_input("Wydatki po FIRE (PLN/mies.)", 0.0, 1e7, p.retirement_monthly_expenses, 250.0, format="%.0f")
        c1, c2, c3 = st.columns(3)
        swr = c1.number_input("SWR (%)", 1.0, 10.0, s.fire.swr * 100, 0.25)
        infl = c2.number_input("Inflacja (%)", -5.0, 30.0, p.inflation * 100, 0.1)
        mode = c3.selectbox("Wariant FIRE", list(FireMode), index=list(FireMode).index(s.fire.mode), format_func=MODE_LABELS.get)
        st.caption("Pełny zestaw parametrów edytujesz na stronach Planera, Konstruktora portfela i Życia po FIRE.")
        if st.form_submit_button("Zapisz zmiany", type="primary", icon=":material/save:"):
            try:
                new = s.model_copy(deep=True)
                new.profile = new.profile.model_validate({**p.model_dump(), "current_age": int(cur), "fire_age": int(fa),
                                                          "life_horizon_age": int(hz), "initial_capital": cap,
                                                          "monthly_contribution": con, "retirement_monthly_expenses": exp,
                                                          "inflation": infl / 100})
                new.fire = new.fire.model_validate({**s.fire.model_dump(), "swr": swr / 100, "mode": mode})
                state.save(new, f"Zapisano zmiany w „{s.name}”.")
                st.rerun()
            except ValidationError as exc:
                for m in polish_errors(exc):
                    st.error(m)


@st.dialog("Usuń scenariusz", icon=":material/delete:")
def delete_dialog(sid: str) -> None:
    st.markdown(f"Czy na pewno usunąć scenariusz **{sc[sid].name}**? Scenariusz zostanie usunięty także z bazy danych – tej operacji nie można cofnąć (chyba że masz eksport JSON).")
    c1, c2 = st.columns(2)
    if c1.button("Usuń", type="primary", icon=":material/delete:"):
        if state.delete(sid):
            st.toast("Usunięto scenariusz.", icon=":material/delete:")
        else:
            st.toast("Nie można usunąć ostatniego scenariusza.", icon=":material/block:")
        st.rerun()
    if c2.button("Anuluj"):
        st.rerun()


# ------------------------------------------------------------------ management
with st.container(border=True):
    st.markdown("**Zarządzanie scenariuszami**")
    ids = list(sc)
    sid = st.selectbox("Scenariusz", ids, index=ids.index(active.id), format_func=lambda i: sc[i].name, key="ui_sc_pick")
    st.caption(sc[sid].description or "Brak opisu.")
    with st.container(horizontal=True, gap="small"):
        if st.button("Nowy", icon=":material/add:"):
            new_dialog()
        if st.button("Ustaw jako aktywny", icon=":material/radio_button_checked:", disabled=sid == active.id):
            state.set_active(sid)
            st.query_params["scenariusz"] = sc[sid].name
            st.toast(f"Aktywny scenariusz: {sc[sid].name}")
            st.rerun()
        if st.button("Klonuj", icon=":material/content_copy:"):
            c = clone(sc[sid])
            state.add(c, make_active=False)
            st.toast(f"Utworzono kopię „{c.name}”.", icon=":material/content_copy:")
            st.rerun()
        if st.button("Zmień nazwę", icon=":material/edit:"):
            rename_dialog(sid)
        if st.button("Edytuj parametry", icon=":material/tune:"):
            edit_dialog(sid)
        if st.button("Usuń", icon=":material/delete:", disabled=len(sc) <= 1):
            delete_dialog(sid)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Eksportuj wszystkie scenariusze (JSON)", to_json(list(sc.values())).encode("utf-8"),
                           "fire_lab_scenariusze.json", "application/json", icon=":material/download:")
        st.download_button("Eksportuj wybrany scenariusz (JSON)", to_json([sc[sid]]).encode("utf-8"),
                           f"fire_lab_{sc[sid].name}.json", "application/json", icon=":material/download:")
    with c2:
        up = st.file_uploader("Importuj scenariusze z JSON", type=["json"], key="ui_sc_import")
        if up is not None and st.button("Wczytaj scenariusze z pliku", icon=":material/upload_file:"):
            try:
                items = from_json(up.getvalue(), set(sc))
                for it in items:
                    state.add(it, make_active=False)
                st.toast(f"Zaimportowano {len(items)} scenariusz(y).", icon=":material/check_circle:")
                st.rerun()
            except ImportErrorPL as exc:
                st.error(str(exc), icon=":material/error:")

# ------------------------------------------------------------------ comparison
st.subheader("Porównanie", anchor=False)
chosen = st.multiselect("Scenariusze do porównania", list(sc), default=list(sc), format_func=lambda i: sc[i].name,
                        key="ui_sc_compare")
if not chosen:
    st.info("Wybierz co najmniej jeden scenariusz.")
    st.stop()

if st.button("Przelicz Monte Carlo dla wybranych scenariuszy", icon=":material/play_arrow:",
             help="Każdy scenariusz liczony z własnymi ustawieniami symulacji (liczba ścieżek, ziarno, model)."):
    bar = st.progress(0.0, text="Start")
    for i, k in enumerate(chosen):
        ctx_k = analytics.get_context(sc[k])
        if not ctx_k.ok:
            st.warning(f"{sc[k].name}: {ctx_k.error}")
            continue
        res = analytics.run_monte_carlo(ctx_k, lambda f, t, i=i: bar.progress(min((i + f) / len(chosen), 1.0),
                                                                              text=f"{sc[k].name}: {t}"))
        state.store_result(sc[k], "montecarlo", res)
    bar.progress(1.0, text="Gotowe")
    st.toast("Przeliczono scenariusze.", icon=":material/check_circle:")

rows, paths = [], {}
for k in chosen:
    s = sc[k]
    ctx = analytics.get_context(s)
    row = {"id": k, "Scenariusz": s.name, "Wariant": MODE_LABELS[s.fire.mode], "Wiek FIRE (cel)": s.profile.fire_age}
    if ctx.ok:
        proj = analytics.deterministic(ctx)
        paths[k] = proj.table
        row.update({"Kapitał FIRE": proj.target.capital, "Wiek FIRE (deterministycznie)": proj.fire_age,
                    "E[R] portfela": ctx.stats["expected_arithmetic"], "Zmienność": ctx.stats["volatility"],
                    "Wpłata mies.": s.profile.monthly_contribution, "SWR": s.fire.swr})
    mc, fresh, _ = state.get_result(s, "montecarlo")
    if mc is not None and fresh:
        sm = mc["summary"]
        row.update({"P(FIRE w wieku docelowym)": sm["p_fire_by_target_age"], "P(utrzymania kapitału)": sm["p_not_depleted"],
                    "P(sukcesu)": sm["p_success"], "Mediana wieku FIRE (MC)": sm["median_fire_age"],
                    "Mediana majątku końcowego": sm["median_terminal_wealth"]})
    rows.append(row)
comp = pd.DataFrame(rows)
for col in ["P(FIRE w wieku docelowym)", "P(utrzymania kapitału)", "P(sukcesu)", "Mediana wieku FIRE (MC)", "Mediana majątku końcowego"]:
    if col not in comp:
        comp[col] = np.nan
if comp["P(sukcesu)"].isna().any():
    st.caption(":material/info: Puste prawdopodobieństwa = Monte Carlo nie zostało policzone lub wyniki są nieaktualne.")
pct_col = lambda: st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")  # noqa: E731
shown = polish_frame(comp.drop(columns=["id"]), money=["Kapitał FIRE", "Wpłata mies.", "Mediana majątku końcowego"],
                     pct=["E[R] portfela", "Zmienność", "SWR"], age=["Wiek FIRE (deterministycznie)", "Mediana wieku FIRE (MC)"])
st.dataframe(shown, hide_index=True, width="stretch", column_config={
    "P(FIRE w wieku docelowym)": pct_col(), "P(utrzymania kapitału)": pct_col(), "P(sukcesu)": pct_col()})
st.download_button("Eksportuj porównanie (CSV)", comp.drop(columns=["id"]).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                   "fire_lab_porownanie_scenariuszy.csv", "text/csv", icon=":material/download:")

t1, t2, t3 = st.tabs([":material/show_chart: Ścieżki kapitału", ":material/shield: Ryzyko", ":material/balance: Czas vs bezpieczeństwo"])
with t1:
    fig = go.Figure()
    for i, (k, tab) in enumerate(paths.items()):
        fig.add_trace(go.Scatter(x=tab["age"], y=tab["wealth_real"], name=sc[k].name, line=dict(color=SERIES[i % len(SERIES)], width=2.5)))
        fig.add_hline(y=tab["target_real"].iloc[0], line=dict(color=SERIES[i % len(SERIES)], dash="dot", width=1))
    fig.update_xaxes(title="Wiek")
    fig.update_yaxes(title="Majątek realny (ścieżka deterministyczna)", tickformat=",.0f", ticksuffix=" zł")
    st.plotly_chart(style(fig, 420), key="sc_paths", config={"displaylogo": False})
    st.caption("Linie przerywane – cel FIRE danego scenariusza.")
with t2:
    have = comp.dropna(subset=["P(sukcesu)"])
    if have.empty:
        st.info("Przelicz Monte Carlo, aby porównać ryzyko scenariuszy.")
    else:
        fig = go.Figure()
        for j, col in enumerate(["P(FIRE w wieku docelowym)", "P(utrzymania kapitału)", "P(sukcesu)"]):
            fig.add_trace(go.Bar(x=have["Scenariusz"], y=have[col], name=col, marker_color=[ORANGE, TEAL, NAVY][j]))
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(style(fig, 380, barmode="group", hovermode="x"), key="sc_risk", config={"displaylogo": False})
with t3:
    have = comp.dropna(subset=["P(sukcesu)"])
    if have.empty:
        st.info("Przelicz Monte Carlo, aby zobaczyć kompromis między czasem dojścia do FIRE a bezpieczeństwem.")
    else:
        x = have["Mediana wieku FIRE (MC)"].fillna(have["Wiek FIRE (cel)"])
        fig = go.Figure(go.Scatter(x=x, y=have["P(sukcesu)"], mode="markers+text", text=have["Scenariusz"],
                                   textposition="top center", marker=dict(size=18, color=TEAL, line=dict(width=2, color=NAVY)),
                                   hovertemplate="%{text}<br>Wiek FIRE: %{x:.1f}<br>P(sukcesu): %{y:.1%}<extra></extra>"))
        fig.update_xaxes(title="Mediana wieku osiągnięcia FIRE (Monte Carlo) – wcześniej = lepiej")
        fig.update_yaxes(title="P(sukcesu planu) – wyżej = bezpieczniej", tickformat=".0%")
        st.plotly_chart(style(fig, 420, hovermode="closest"), key="sc_tradeoff", config={"displaylogo": False})
        st.caption("Idealny scenariusz leży w lewym górnym rogu: wcześnie osiągnięta niezależność przy wysokim prawdopodobieństwie sukcesu.")
