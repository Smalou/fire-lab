"""Strona 1 – Centrum dowodzenia."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import accumulation_chart, allocation_donut, class_allocation
from components.formatting import fmt_age, fmt_money, fmt_num, fmt_pct, years_pl
from components.kpi import kpi
from components.layout import context_guard, data_badge, page_header
from components.theme import ICONS, ORANGE, TEAL, hex_to_rgba
from core import portfolio as pf
from core.fire import MODE_LABELS
from core.metrics import max_drawdown
from services import analytics, catalog, state
from services.reports import build_markdown, markdown_to_html

state.init()
s = state.active()
page_header(f"Centrum dowodzenia · {s.name}", s.description or "Podsumowanie aktualnego scenariusza inwestora.")
ctx = analytics.get_context(s)
data_badge(ctx)
if not context_guard(ctx):
    st.stop()

proj = analytics.deterministic(ctx)
target = proj.target
mc, mc_fresh, mc_at = state.get_result(s, "montecarlo")
progress = s.profile.initial_capital / target.capital if target.capital else 1.0
years_to = proj.fire_age - s.profile.current_age if proj.fire_age is not None else None
acc_part = proj.table[proj.table["age"] <= s.profile.fire_age]

# ------------------------------------------------------------------ KPIs
r1 = st.columns(4)
with r1[0]:
    kpi("Obecny kapitał", fmt_money(s.profile.initial_capital, compact=True), icon=":material/account_balance_wallet:",
        delta=f"+{fmt_money(s.profile.monthly_contribution)}/mies.", delta_color="off")
with r1[1]:
    kpi("Wymagany kapitał FIRE", fmt_money(target.capital, compact=True), icon=":material/flag:",
        help=f"{MODE_LABELS[s.fire.mode]} · {target.explanation}", delta=f"SWR {fmt_pct(s.fire.swr, 2)}", delta_color="off")
with r1[2]:
    kpi("Stopień realizacji celu", fmt_pct(min(progress, 9.99), 1), chart=acc_part["wealth_real"].tolist(),
        help="Obecny kapitał / wymagany kapitał. Miniwykres: deterministyczna ścieżka do wieku FIRE.")
with r1[3]:
    kpi("Prognozowany wiek FIRE", fmt_age(proj.fire_age) if proj.fire_age else "poza horyzontem",
        delta=f"cel: {s.profile.fire_age} lat", delta_color="off",
        help="Ścieżka deterministyczna ze stopą geometryczną (mediana), po inflacji i kosztach.")
r2 = st.columns(3)
with r2[0]:
    kpi("Lata do FIRE", years_pl(round(years_to, 1)) if years_to is not None else "—",
        delta=(f"{fmt_num(proj.fire_age - s.profile.fire_age, 1)} r. względem celu" if proj.fire_age else None),
        delta_color="inverse")

prob_help = "Wynik symulacji Monte Carlo z Laboratorium ryzyka."
if mc is not None and mc_fresh:
    sm = mc["summary"]
    with r2[1]:
        kpi("P(osiągnięcia celu)", fmt_pct(sm["p_fire_by_target_age"], 1), help=prob_help + f" Do wieku {s.profile.fire_age} lat.",
            icon=":material/casino:")
    with r2[2]:
        kpi("P(utrzymania kapitału)", fmt_pct(sm["p_not_depleted"], 1), help=prob_help + f" Do wieku {s.profile.life_horizon_age} lat.",
            delta=f"sukces planu: {fmt_pct(sm['p_success'], 0)}", delta_color="off")
else:
    label = "nieaktualne" if mc is not None else "nie obliczono"
    with r2[1]:
        kpi("P(osiągnięcia celu)", "—", delta=label, delta_color="off", help="Prawdopodobieństwa są pokazywane dopiero po obliczeniach.")
    with r2[2]:
        kpi("P(utrzymania kapitału)", "—", delta=label, delta_color="off", help="Prawdopodobieństwa są pokazywane dopiero po obliczeniach.")
r3 = st.columns(3)
with r3[0]:
    kpi("Oczekiwana stopa zwrotu", fmt_pct(ctx.stats["expected_arithmetic"], 2),
        delta=f"geometryczna {fmt_pct(ctx.stats['geometric_nominal'], 2)}", delta_color="off",
        help="Nominalnie, w walucie bazowej; średnia arytmetyczna z okresu estymacji.")
with r3[1]:
    kpi("Zmienność portfela", fmt_pct(ctx.stats["volatility"], 2), chart_type="bar",
        chart=list(np.round(ctx.stats["hist_monthly"][-24:] * 100, 2)), help="Roczne odchylenie standardowe. Miniwykres: ostatnie 24 stopy miesięczne (%).")
with r3[2]:
    kpi("Realna stopa netto", fmt_pct(ctx.stats["geometric_real_net"], 2), help="Geometryczna, po inflacji i kosztach portfela.")

if mc is None or not mc_fresh:
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        st.markdown(":material/casino: **Prawdopodobieństwa nie zostały jeszcze obliczone dla bieżących parametrów.**")
        if st.button("Oblicz teraz", type="primary", icon=ICONS["run"],
                     help=f"Monte Carlo wg ustawień scenariusza: {fmt_num(s.simulation.n_paths)} ścieżek, ziarno {s.simulation.seed}."):
            bar = st.progress(0.0, text="Start")
            res = analytics.run_monte_carlo(ctx, lambda f, t: bar.progress(min(f, 1.0), text=t))
            state.store_result(s, "montecarlo", res)
            st.toast("Obliczono prawdopodobieństwa.", icon=":material/check_circle:")
            st.rerun()
        st.page_link("views/risk_lab.py", label="Ustawienia symulacji", icon=ICONS["risk"])

# ------------------------------------------------------------------ charts
left, right = st.columns([3, 2])
with left:
    with st.container(border=True):
        st.markdown("**Ścieżka akumulacji kapitału**")
        fig = accumulation_chart(proj.table, target.capital, proj.fire_age, False, s.profile.fire_age, height=380)
        if mc is not None and mc_fresh and st.toggle("Przedział Monte Carlo (P25–P75)", key="ui_cc_band", value=True):
            pct = mc["wealth_percentiles"]
            fig.add_trace(go.Scatter(x=mc["ages"], y=pct[3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=mc["ages"], y=pct[1], fill="tonexty", fillcolor=hex_to_rgba(ORANGE, 0.12), line=dict(width=0),
                                     name="MC: P25–P75", hoverinfo="skip"))
        st.plotly_chart(fig, key="cc_path", config={"displaylogo": False})
with right:
    with st.container(border=True):
        st.markdown("**Struktura aktywów**")
        st.plotly_chart(allocation_donut(ctx.assets, ctx.weights, ctx.classes, height=260), key="cc_donut",
                        config={"displaylogo": False})
        ca = class_allocation(ctx.assets, ctx.weights, ctx.classes)
        st.caption(" · ".join(f"{k}: {fmt_pct(v, 0)}" for k, v in ca.items()))

# ------------------------------------------------------------------ risk factors (all computed)
mom = ctx.moments
w = ctx.weights
rc = pf.risk_contributions(w, mom.cov)
vol = pf.portfolio_volatility(w, mom.cov)
top = int(np.argmax(rc))
foreign = sum(wi for a, wi in zip(ctx.assets, w) if ctx.currencies.get(a, s.market.base_currency) != s.market.base_currency)
equity = sum(wi for a, wi in zip(ctx.assets, w) if ctx.classes[a].endswith("equity"))
mdd = max_drawdown(ctx.stats["hist_monthly"])
hhi = float(np.sum(w**2))
factors = [
    ("Największe źródło ryzyka", f"{catalog.short_name(ctx.assets[top])} – {fmt_pct(rc[top] / vol, 0)} zmienności portfela",
     rc[top] / vol > 0.5),
    ("Udział akcji", fmt_pct(equity, 0), equity > 0.8),
    ("Ekspozycja walutowa", fmt_pct(foreign, 0) + f" aktywów poza {s.market.base_currency}", foreign > 0.7),
    ("Historyczne maks. obsunięcie portfela", fmt_pct(mdd, 1) + " w okresie estymacji", mdd < -0.3),
    ("Stopa wypłat (SWR)", fmt_pct(s.fire.swr, 2) + f" przy horyzoncie {s.profile.life_horizon_age - s.profile.fire_age} lat",
     s.fire.swr > 0.04 and s.profile.life_horizon_age - s.profile.fire_age > 35),
    ("Koncentracja portfela (HHI)", fmt_num(hhi, 2) + f" · efektywna liczba aktywów {fmt_num(1 / hhi, 1)}", hhi > 0.4),
]
with st.container(border=True):
    st.markdown("**Najważniejsze czynniki ryzyka**", help="Wszystkie wartości wyliczone dla bieżącego portfela i scenariusza.")
    cols = st.columns(3)
    for i, (name, value, alert) in enumerate(factors):
        with cols[i % 3]:
            st.markdown(f"{':material/warning:' if alert else ':material/check_circle:'} **{name}**")
            st.caption(value)

# ------------------------------------------------------------------ navigation
st.markdown("**Szczegółowe analizy**")
nav = [
    ("views/planner.py", "Planer FIRE", ICONS["planner"], "Cel, tryb FIRE, zdarzenia finansowe"),
    ("views/portfolio_builder.py", "Konstruktor portfela", ICONS["portfolio"], "Granica efektywna, Risk Parity, CAPM"),
    ("views/strategy_lab.py", "Laboratorium strategii", ICONS["strategy"], "Backtest walk-forward"),
    ("views/risk_lab.py", "Laboratorium ryzyka", ICONS["risk"], "Monte Carlo, VaR, stres testy"),
    ("views/post_fire.py", "Życie po FIRE", ICONS["post_fire"], "Strategie wypłat, ryzyko sekwencji"),
    ("views/scenarios.py", "Porównywarka scenariuszy", ICONS["scenarios"], "Warianty planu i kompromisy"),
]
cols = st.columns(3)
for i, (page, label, icon, desc) in enumerate(nav):
    with cols[i % 3].container(border=True):
        st.page_link(page, label=label, icon=icon)
        st.caption(desc)

# ------------------------------------------------------------------ report export
bt, bt_fresh, _ = state.get_result(s, "backtest")
dec, dec_fresh, _ = state.get_result(s, "decumulation")
md = build_markdown(s, ctx, proj, mc, mc_fresh, bt, bt_fresh, dec, dec_fresh)
with st.container(horizontal=True):
    st.download_button("Raport (Markdown)", md.encode("utf-8"), f"fire_lab_raport_{s.name}.md", "text/markdown", icon=ICONS["download"])
    st.download_button("Raport (HTML)", markdown_to_html(md, f"FIRE Lab – {s.name}").encode("utf-8"), f"fire_lab_raport_{s.name}.html",
                       "text/html", icon=ICONS["download"])
