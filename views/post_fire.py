"""Strona 6 – Życie po FIRE (dekumulacja, strategie wypłat, Sequence of Returns Risk)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from components.charts import fan_chart, histogram
from components.formatting import nearest, fmt_age, fmt_money, fmt_num, fmt_pct, polish_frame
from components.kpi import kpi
from components.layout import context_guard, data_badge, page_header, stale_notice
from components.theme import GREY, NAVY, ORANGE, RED, SERIES, TEAL, style
from components.validation import polish_errors
from core.fire import required_capital
from core.models import IncomeSource, WithdrawalStrategy
from core.sequence import crash_sequence, geometric_mean, orderings, simulate_sequence
from core.withdrawal import STRATEGY_HELP, STRATEGY_LABELS
from services import analytics, state

state.init()
s = state.active()
page_header("Życie po FIRE", "Porównaj strategie wypłat: czy kapitał wystarczy i czy pokryje wymagane wydatki?")
ctx = analytics.get_context(s)
data_badge(ctx)
if not context_guard(ctx):
    st.stop()

ws = s.withdrawal
target = required_capital(s, ctx.stats["geometric_real_net"])
mc, mc_fresh, _ = state.get_result(s, "montecarlo")
RATES = [round(x, 4) for x in np.arange(0.025, 0.0801, 0.0025)]
income_ok = ctx.stats["income_available"]

with st.form("withdrawal_form"):
    opts = [x for x in WithdrawalStrategy if income_ok or x != WithdrawalStrategy.INCOME_ONLY]
    strat = st.segmented_control("Strategia wypłat w scenariuszu", opts, default=ws.strategy if ws.strategy in opts else opts[0],
                                 format_func=STRATEGY_LABELS.get, required=True)
    if not income_ok:
        st.caption(":material/block: Strategia „tylko dochód” jest niedostępna – brak danych o wypłatach dywidend/kuponów w bieżącym źródle.")
    c1, c2, c3 = st.columns(3)
    basis = c1.radio("Stała realna wypłata – podstawa", ["rate", "spending"], index=["rate", "spending"].index(ws.constant_basis),
                     format_func={"rate": "% kapitału w chwili FIRE", "spending": "Wymagane wydatki"}.get,
                     help="Reguła 4% = stała realna wypłata ustalona jako 4% kapitału w chwili FIRE.")
    init_rate = c2.select_slider("Początkowa stopa wypłat", RATES, value=min(RATES, key=lambda x: abs(x - ws.initial_rate)),
                                 format_func=lambda x: fmt_pct(x, 2), help="Dla stałej realnej wypłaty i Guardrails.")
    pct_rate = c3.select_slider("Stały % portfela (rocznie)", RATES, value=min(RATES, key=lambda x: abs(x - ws.percent_rate)),
                                format_func=lambda x: fmt_pct(x, 2))
    c1, c2, c3 = st.columns(3)
    floor = c1.select_slider("Minimalny akceptowalny poziom wydatków", [round(x / 20, 2) for x in range(10, 21)],
                             value=nearest([round(x / 20, 2) for x in range(10, 21)], ws.spending_floor_share), format_func=lambda x: fmt_pct(x, 0),
                      help="Wydatki pokryte = konsumpcja ≥ próg × wymagane wydatki w każdym miesiącu po FIRE.")
    tax = c2.number_input("Efektywny podatek od wypłat (%)", 0.0, 60.0, ws.withdrawal_tax_rate * 100, 1.0,
                          help="Uproszczenie: wypłata brutto = netto / (1 − t). Nie jest to model polskiego systemu podatkowego.")
    cap_opts = ["target", "custom"] + (["mc"] if mc is not None and mc_fresh else [])
    cap_mode = c3.radio("Kapitał na start FIRE", cap_opts, key="ui_cap_mode",
                        format_func={"target": "Wymagany kapitał FIRE", "custom": "Własna kwota",
                                     "mc": "Mediana z Monte Carlo"}.get)
    custom_cap = st.number_input("Własna kwota kapitału (PLN, realnie)", 0.0, 1e10, float(round(target.capital, -3)), 50_000.0,
                                 format="%.0f", disabled=cap_mode != "custom")
    with st.popover("Parametry Guardrails", icon=":material/tune:"):
        steps = [round(x / 20, 2) for x in range(1, 13)]
        g_up = st.select_slider("Górna bariera (+% stopy początkowej)", steps, value=nearest(steps, ws.guardrail_upper),
                                format_func=lambda x: fmt_pct(x, 0),
                         help="Gdy bieżąca stopa wypłat > stopa początkowa × (1 + bariera) → cięcie wypłaty.")
        g_lo = st.select_slider("Dolna bariera (−% stopy początkowej)", steps, value=nearest(steps, ws.guardrail_lower),
                                format_func=lambda x: fmt_pct(x, 0))
        g_adj = st.select_slider("Wielkość korekty wypłaty", steps[:6], value=min(steps[:6], key=lambda x: abs(x - ws.guardrail_adjustment)),
                                 format_func=lambda x: fmt_pct(x, 0))
    st.markdown("**Dodatkowe źródła dochodu po FIRE** (wspólne z Planerem FIRE)")
    inc_df = pd.DataFrame([{"Aktywne": i.enabled, "Nazwa": i.name, "Kwota miesięczna (PLN)": i.monthly_amount,
                            "Od wieku": i.start_age, "Do wieku": i.end_age} for i in s.fire.incomes],
                          columns=["Aktywne", "Nazwa", "Kwota miesięczna (PLN)", "Od wieku", "Do wieku"]).astype({"Do wieku": "float64"})
    inc_edit = st.data_editor(inc_df, num_rows="dynamic", hide_index=True, width="stretch", key="ui_pf_incomes",
                              column_config={"Aktywne": st.column_config.CheckboxColumn(default=True),
                                             "Nazwa": st.column_config.TextColumn(default="Praca w niepełnym wymiarze"),
                                             "Kwota miesięczna (PLN)": st.column_config.NumberColumn(min_value=0, step=100, default=0),
                                             "Od wieku": st.column_config.NumberColumn(min_value=0, max_value=120, default=s.profile.fire_age),
                                             "Do wieku": st.column_config.NumberColumn(min_value=0, max_value=120)})
    run = st.form_submit_button("Zapisz i porównaj strategie", type="primary", icon=":material/play_arrow:")

if run:
    try:
        incomes = [IncomeSource(enabled=bool(r["Aktywne"]), name=str(r["Nazwa"] or "Dochód"),
                                monthly_amount=float(r["Kwota miesięczna (PLN)"] or 0), start_age=int(r["Od wieku"]),
                                end_age=None if pd.isna(r["Do wieku"]) else int(r["Do wieku"]))
                   for _, r in inc_edit.dropna(subset=["Od wieku"]).iterrows()]
        new = s.model_copy(deep=True)
        new.withdrawal = new.withdrawal.model_validate({**ws.model_dump(), "strategy": strat, "constant_basis": basis,
                                                        "initial_rate": init_rate, "percent_rate": pct_rate,
                                                        "spending_floor_share": floor, "withdrawal_tax_rate": tax / 100,
                                                        "guardrail_upper": g_up, "guardrail_lower": g_lo, "guardrail_adjustment": g_adj})
        new.fire = new.fire.model_copy(update={"incomes": incomes})
        state.save(new, None)
    except ValidationError as exc:
        for m in polish_errors(exc):
            st.error(m)
        st.stop()
    s = state.active()
    ctx = analytics.get_context(s)
    start_cap = {"target": target.capital, "custom": custom_cap,
                 "mc": float(np.median(mc["wealth_at_fire"])) if mc is not None else target.capital}[cap_mode]
    bar = st.progress(0.0, text="Start")
    with st.status("Symulacja dekumulacji", expanded=False) as status:
        out = analytics.run_decumulation_lab(ctx, start_cap, lambda f, t: bar.progress(min(f, 1.0), text=t))
        status.update(label="Porównanie strategii gotowe", state="complete")
    state.store_result(s, "decumulation", out)
    st.toast("Zapisano ustawienia i wyniki dekumulacji.", icon=":material/check_circle:")

res, fresh, at = state.get_result(s, "decumulation")
if res is None:
    st.info("Ustaw parametry i uruchom porównanie strategii wypłat. Wszystkie strategie są testowane na tych samych ścieżkach "
            "losowych (common random numbers), więc różnice wynikają wyłącznie z reguły wypłat.", icon=":material/info:")
else:
    stale_notice(at, fresh)
    variants = res["variants"]
    st.caption(f"Kapitał na start: **{fmt_money(res['start_capital'])}** · wymagane wydatki: {fmt_money(res['required_spending'])}/mies. · "
               f"{fmt_num(res['n_paths'])} ścieżek · wiek {s.profile.fire_age}–{s.profile.life_horizon_age}")
    rows = []
    for label, v in variants.items():
        sm = v["summary"]
        d = v["depletion_ages"]
        rows.append({"Strategia": label, "P(niewyczerpania)": sm["p_not_depleted"],
                     "P(pokrycia wydatków)": sm["p_spending_covered"],
                     "Mediana wieku wyczerpania": float(np.nanmedian(d)) if np.isfinite(d).any() else None,
                     "Mediana majątku końcowego": sm["median_terminal_wealth"],
                     "Łączne wypłaty (mediana)": sm["median_total_withdrawals"],
                     "Zmienność konsumpcji": sm["median_consumption_vol"],
                     "Największy spadek wydatków (mediana)": sm["median_max_cut"],
                     "Największy spadek wydatków (P95)": sm["p95_max_cut"]})
    tab = pd.DataFrame(rows)
    shown = polish_frame(tab, money=["Mediana majątku końcowego", "Łączne wypłaty (mediana)"],
                         pct=["Zmienność konsumpcji", "Największy spadek wydatków (mediana)", "Największy spadek wydatków (P95)"],
                         age=["Mediana wieku wyczerpania"])
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "P(niewyczerpania)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent",
                                                            help="Portfel nie spada do zera przed końcem horyzontu."),
        "P(pokrycia wydatków)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent",
                                                               help="Konsumpcja (wypłata + inne dochody) ≥ progu w każdym miesiącu."),
        "Mediana wieku wyczerpania": st.column_config.TextColumn(help="Wśród ścieżek, na których kapitał się wyczerpał."),
        "Zmienność konsumpcji": st.column_config.TextColumn(help="Odchylenie std. rocznych zmian realnej konsumpcji (mediana ścieżek)."),
    })
    st.info("Sukces ≠ dodatni kapitał. Stały % portfela praktycznie nigdy nie wyczerpuje kapitału, ale może nie pokryć wymaganych "
            "wydatków. Oceniaj łącznie: P(niewyczerpania), P(pokrycia wydatków) i stabilność konsumpcji.", icon=":material/lightbulb:")
    st.download_button("Porównanie strategii (CSV)", tab.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       "fire_lab_strategie_wyplat.csv", "text/csv", icon=":material/download:")

    pick = st.selectbox("Szczegóły strategii", list(variants), key="ui_pf_pick")
    v = variants[pick]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("P(niewyczerpania)", fmt_pct(v["summary"]["p_not_depleted"], 1))
    with c2:
        kpi("P(pokrycia wydatków)", fmt_pct(v["summary"]["p_spending_covered"], 1))
    with c3:
        kpi("Mediana majątku końcowego", fmt_money(v["summary"]["median_terminal_wealth"], compact=True))
    with c4:
        kpi("Największy spadek wydatków", fmt_pct(v["summary"]["median_max_cut"], 1), delta=f"P95: {fmt_pct(v['summary']['p95_max_cut'], 0)}",
            delta_color="off", help="Największy spadek realnych rocznych wydatków od wcześniejszego maksimum (mediana ścieżek).")
    t1, t2, t3 = st.tabs([":material/ssid_chart: Majątek", ":material/shopping_cart: Konsumpcja", ":material/hourglass_bottom: Wyczerpanie"])
    with t1:
        st.plotly_chart(fan_chart(v["ages"], v["wealth_percentiles"], None, None), key="pf_fan", config={"displaylogo": False})
    with t2:
        if v["consumption_median"] is not None:
            ages = v["ages"][:-1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ages, y=v["consumption_median"], name="Mediana konsumpcji", line=dict(color=TEAL, width=3)))
            fig.add_trace(go.Scatter(x=ages, y=v["consumption_p10"], name="P10 konsumpcji", line=dict(color=RED, dash="dot")))
            fig.add_hline(y=res["required_spending"] * 12, line=dict(color=ORANGE, dash="dash"), annotation_text="Wymagane wydatki")
            fig.add_hline(y=res["required_spending"] * 12 * s.withdrawal.spending_floor_share, line=dict(color=GREY, dash="dot"),
                          annotation_text="Próg minimalny")
            fig.update_yaxes(title="Roczna konsumpcja realna", tickformat=",.0f", ticksuffix=" zł")
            fig.update_xaxes(title="Wiek")
            st.plotly_chart(style(fig, 380), key="pf_cons", config={"displaylogo": False})
            st.caption("Konsumpcja = wypłata netto z portfela + dodatkowe dochody (emerytura, praca).")
    with t3:
        d = v["depletion_ages"]
        if np.isfinite(d).any():
            st.plotly_chart(histogram(d, "Wiek wyczerpania kapitału", RED, nbins=30, money=False), key="pf_depl",
                            config={"displaylogo": False})
        else:
            st.success("Na żadnej ścieżce kapitał nie wyczerpał się przed końcem horyzontu.")
    with st.expander("Opis strategii", icon=":material/menu_book:"):
        for k, txt in STRATEGY_HELP.items():
            st.markdown(f"**{STRATEGY_LABELS[k]}** – {txt}")
        st.latex(r"W_t^{stała} = r_0\,V_0 \;\;(\text{realnie}), \qquad W_t^{\%} = p\,V_t, \qquad W^{GR}_{t} = W_{t-1}\cdot\begin{cases}1-a & \text{gdy } W/V > r_0(1+u)\\ 1+a & \text{gdy } W/V < r_0(1-l)\end{cases}")

# ------------------------------------------------------------------ Sequence of Returns Risk
st.divider()
st.subheader("Eksperyment: Sequence of Returns Risk", anchor=False)
st.caption("Te same stopy zwrotu w innej kolejności dają ten sam wynik bez wypłat – ale zupełnie inny, gdy regularnie wypłacasz kapitał.")


@st.fragment
def sequence_experiment() -> None:
    mode = st.segmented_control("Tryb", ["crash", "hist"], default="crash", required=True, key="ui_sorr_mode",
                                format_func={"crash": "Krach w wybranym roku", "hist": "Historyczne stopy portfela w innej kolejności"}.get)
    c1, c2, c3 = st.columns(3)
    capital = c1.number_input("Kapitał początkowy (PLN)", 100_000.0, 1e9, float(round(target.capital, -4)), 100_000.0, format="%.0f",
                              key="ui_sorr_cap")
    rate = c2.select_slider("Roczna wypłata (% kapitału początkowego)", [x / 200 for x in range(0, 21)], value=0.04,
                            format_func=lambda x: fmt_pct(x, 1), key="ui_sorr_rate")
    years = c3.slider("Liczba lat", 10, 50, 30, 1, key="ui_sorr_years")
    withdrawal = capital * rate
    if mode == "crash":
        c1, c2, c3 = st.columns(3)
        crash = c1.select_slider("Wielkość krachu", [-x / 20 for x in range(12, 0, -1)], value=-0.35,
                                 format_func=lambda x: fmt_pct(x, 0), key="ui_sorr_crash")
        cagr = c2.select_slider("Średnia geometryczna (realnie)", [x / 200 for x in range(-4, 21)], value=0.045,
                                format_func=lambda x: fmt_pct(x, 1), key="ui_sorr_cagr")
        year = c3.slider("Rok wystąpienia krachu", 1, years, 1, key="ui_sorr_year")
        seqs = {f"Krach w roku {year}": crash_sequence(years, year, crash, cagr),
                "Krach w ostatnim roku": crash_sequence(years, years, crash, cagr)}
    else:
        hist = pd.Series(ctx.stats["hist_monthly"], index=ctx.est_returns.index)
        annual = (1 + hist).groupby(hist.index.year).prod() - 1
        annual = annual[hist.groupby(hist.index.year).size() == 12]
        real = (1 + annual) / (1 + s.profile.inflation) - 1
        if len(real) < 5:
            st.warning("Za mało pełnych lat w oknie estymacji (min. 5).")
            return
        base = np.resize(real.to_numpy(), years)
        st.caption(f"Roczne realne stopy portfela z lat {real.index.min()}–{real.index.max()} (powtórzone cyklicznie do {years} lat).")
        seqs = orderings(base)
    fig = go.Figure()
    rows = []
    for i, (label, r) in enumerate(seqs.items()):
        path = simulate_sequence(r, capital, withdrawal, label)
        no_w = simulate_sequence(r, capital, 0.0, label)
        fig.add_trace(go.Scatter(x=np.arange(years + 1), y=path.wealth, name=label, line=dict(color=SERIES[i], width=2.5)))
        rows.append({"Kolejność": label, "Średnia geometryczna": fmt_pct(geometric_mean(r), 2),
                     "Średnia arytmetyczna": fmt_pct(float(np.mean(r)), 2),
                     "Majątek końcowy bez wypłat": fmt_money(no_w.wealth[-1], compact=True),
                     "Majątek końcowy z wypłatami": fmt_money(path.wealth[-1], compact=True),
                     "Rok wyczerpania": str(path.depletion_year) if path.depletion_year else "—",
                     "Łączne wypłaty": fmt_money(path.total_withdrawn, compact=True)})
    fig.update_xaxes(title="Rok od przejścia na FIRE")
    fig.update_yaxes(title="Majątek realny", tickformat=",.0f", ticksuffix=" zł")
    st.plotly_chart(style(fig, 400), key="sorr_chart", config={"displaylogo": False})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Wypłata na początku roku, potem stopa zwrotu roku. Bez wypłat majątek końcowy jest identyczny dla każdej kolejności.")


sequence_experiment()
