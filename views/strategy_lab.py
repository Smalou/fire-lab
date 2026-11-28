"""Strona 4 – Laboratorium strategii (backtest walk-forward)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import heatmap, lines
from components.formatting import nearest, fmt_money, fmt_num, fmt_pct, polish_frame
from components.layout import context_guard, data_badge, page_header, stale_notice
from components.theme import NAVY, RED, SERIES, hex_to_rgba, style
from core.backtest import ESTIMATED, STRATEGY_LABELS
from core.metrics import drawdown_series, monthly_return_table, rolling_sharpe, rolling_volatility
from services import analytics, catalog, state

state.init()
s = state.active()
page_header("Laboratorium strategii", "Historyczny backtest strategii z wpłatami, kosztami i estymacją wyłącznie na danych z przeszłości.")
ctx = analytics.get_context(s)
data_badge(ctx)
if not context_guard(ctx):
    st.stop()

bt = s.backtest
assets = ctx.assets
REBAL = {"none": "Brak", "monthly": "Miesięcznie", "quarterly": "Kwartalnie", "semiannual": "Półrocznie", "annual": "Rocznie"}
default_strats = st.session_state.get("ui_bt_strategies_saved", list(STRATEGY_LABELS))

with st.form("backtest_form"):
    strategies = st.multiselect("Strategie do porównania", list(STRATEGY_LABELS), default=default_strats,
                                format_func=STRATEGY_LABELS.get)
    c1, c2, c3 = st.columns([2, 1, 1])
    rebalance = c1.segmented_control("Rebalancing", list(REBAL), default=bt.rebalance, format_func=REBAL.get, required=True,
                                     help="Dotyczy strategii z rebalancingiem. Buy & Hold nigdy nie rebalansuje.")
    initial = c2.number_input("Inwestycja początkowa (PLN)", 0.0, 1e9, bt.initial_investment, 10_000.0, format="%.0f")
    monthly = c3.number_input("Wpłata miesięczna (PLN)", 0.0, 1e7, bt.monthly_contribution, 500.0, format="%.0f")
    c1, c2, c3, c4 = st.columns(4)
    commission = c1.number_input("Prowizja (% obrotu)", 0.0, 5.0, bt.commission_rate * 100, 0.01, format="%.2f")
    spread = c2.number_input("Koszt obrotu / spread (pb)", 0.0, 500.0, bt.spread_bps, 1.0,
                             help="Punkty bazowe (1 pb = 0,01%) od wartości transakcji.")
    est_m = c3.slider("Okno estymacji (mies.)", 12, 120, bt.estimation_months, 6,
                      help="Strategie optymalizowane estymują parametry na tym kroczącym oknie. Okres rozgrzewki jest wspólny dla wszystkich strategii.")
    bench = c4.selectbox("Benchmark", assets, index=assets.index(s.market.benchmark) if s.market.benchmark in assets else 0,
                         format_func=catalog.short_name)
    c1, c2 = st.columns(2)
    shares = [round(x / 20, 2) for x in range(21)]
    g0 = c1.select_slider("Glide Path: udział akcji na starcie", shares, value=nearest(shares, bt.glide_start_equity),
                          format_func=lambda x: fmt_pct(x, 0))
    g1 = c2.select_slider("Glide Path: udział akcji na końcu", shares, value=nearest(shares, bt.glide_end_equity),
                          format_func=lambda x: fmt_pct(x, 0))
    run = st.form_submit_button("Uruchom backtest", type="primary", icon=":material/play_arrow:")

if run:
    if not strategies:
        st.error("Wybierz co najmniej jedną strategię.")
    else:
        new = s.model_copy(deep=True)
        new.backtest = new.backtest.model_validate({**bt.model_dump(), "rebalance": rebalance, "initial_investment": initial,
                                                    "monthly_contribution": monthly, "commission_rate": commission / 100,
                                                    "spread_bps": spread, "estimation_months": est_m,
                                                    "glide_start_equity": g0, "glide_end_equity": g1})
        new.market.benchmark = bench
        state.save(new, None)
        st.session_state["ui_bt_strategies_saved"] = strategies
        s = state.active()
        ret = ctx.returns
        results, errors = {}, {}
        with st.status("Backtest walk-forward…", expanded=True) as status:
            st.write(f"Dane: {len(ret)} mies. wspólnej historii, okres rozgrzewki {est_m} mies.")
            for strat in strategies:
                st.write(f"• {STRATEGY_LABELS[strat]}" + (" – reestymacja parametrów na każdej dacie rebalancingu" if strat in ESTIMATED else ""))
                out = analytics.backtests(ret, (strat,), tuple(ctx.classes.items()), tuple(s.portfolio.weights),
                                          s.backtest.model_dump_json(), s.market.risk_free_rate, s.portfolio.management_fee,
                                          s.portfolio.min_weight, s.portfolio.max_weight, s.portfolio.allow_short,
                                          s.market.covariance_method, bench)
                results.update(out["results"])
                errors.update(out["errors"])
            status.update(label=f"Backtest zakończony – {len(results)} strategii", state="complete" if results else "error",
                          expanded=False)
        state.store_result(s, "backtest", {"results": results, "errors": errors, "benchmark": bench})
        st.toast("Zapisano wyniki backtestu.", icon=":material/check_circle:")

payload, fresh, at = state.get_result(s, "backtest")
if payload is None:
    st.info("Ustaw parametry i uruchom backtest. Strategie Minimum Variance, Maximum Sharpe i Risk Parity są estymowane "
            "krocząco (walk-forward) – każda decyzja wykorzystuje wyłącznie dane sprzed jej podjęcia.", icon=":material/info:")
    st.stop()
stale_notice(at, fresh)
results = payload["results"]
for k, msg in payload["errors"].items():
    st.error(f"{STRATEGY_LABELS[k]}: {msg}", icon=":material/error:")
if not results:
    st.stop()
for r in results.values():
    for n in r.notes:
        st.warning(f"{STRATEGY_LABELS[r.strategy]}: {n}")

first = next(iter(results.values()))
st.caption(f"Okres testu: {first.values.index[0]:%m.%Y} – {first.values.index[-1]:%m.%Y} · benchmark: "
           f"{catalog.short_name(payload['benchmark'])} · koszty i opłata za zarządzanie ({fmt_pct(s.portfolio.management_fee, 2)}) uwzględnione.")

rows = []
for k, r in results.items():
    m = r.metrics
    rows.append({
        "Strategia": STRATEGY_LABELS[k], "key": k,
        "Wartość końcowa": m["final_value"], "Wpłaty": m["total_invested"], "Zysk inwestycyjny": m["investment_gain"],
        "Skumulowana (TWR)": m["twr"], "CAGR": m["cagr"], "XIRR": m["xirr"],
        "Zmienność": m["volatility"], "Sharpe": m["sharpe"], "Sortino": m["sortino"], "Max DD": m["max_drawdown"],
        "Beta": m.get("beta"), "Alfa": m.get("alpha"), "Tracking Error": m.get("tracking_error"),
        "Information Ratio": m.get("information_ratio"), "Koszty": m["total_costs"], "Obrót roczny": m["avg_turnover_annual"],
        "Krzywa": (r.twr_returns + 1).cumprod().iloc[::3].round(4).tolist(),
    })
table = pd.DataFrame(rows)
MONEY = ["Wartość końcowa", "Wpłaty", "Zysk inwestycyjny", "Koszty"]
PCTS = ["Skumulowana (TWR)", "CAGR", "XIRR", "Zmienność", "Max DD", "Alfa", "Tracking Error", "Obrót roczny"]
RATIOS = {"Sharpe": 2, "Sortino": 2, "Beta": 2, "Information Ratio": 2}
HELP = {
    "Wpłaty": "Suma wpłat – nie jest wynikiem inwestycyjnym.",
    "Zysk inwestycyjny": "Wartość końcowa − suma wpłat.",
    "Skumulowana (TWR)": "Stopa ważona czasem – niezależna od wpłat.",
    "CAGR": "Średnioroczna geometryczna stopa zwrotu (z TWR).",
    "XIRR": "Stopa ważona kapitałem (uwzględnia moment wpłat).",
    "Zmienność": "Odchylenie standardowe miesięcznych stóp × √12.",
    "Sharpe": "Nadwyżkowa stopa / zmienność (annualizowane).",
    "Sortino": "Nadwyżkowa stopa / odchylenie w dół.",
    "Max DD": "Maximum Drawdown – największy spadek od szczytu indeksu TWR.",
    "Alfa": "Alfa Jensena (rocznie).",
    "Tracking Error": "Zmienność różnicy stóp względem benchmarku.",
    "Koszty": "Suma prowizji i kosztów obrotu.",
}

st.subheader("Porównanie strategii", anchor=False)
display = polish_frame(table.drop(columns=["key"]), money=MONEY, pct=PCTS, num=RATIOS)
cfg = {c: st.column_config.TextColumn(c, help=h) for c, h in HELP.items()}
cfg["Krzywa"] = st.column_config.LineChartColumn("Indeks TWR")
sel = st.dataframe(display, hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row",
                   key="ui_bt_table", column_config=cfg)
picked = [table.iloc[i]["key"] for i in (sel.selection.rows if sel and sel.selection else [])] or list(results)
st.caption("Zaznacz wiersze, aby zawęzić wykresy do wybranych strategii.")

colors = {k: SERIES[i % len(SERIES)] for i, k in enumerate(results)}
t_eq, t_dd, t_roll, t_heat, t_w = st.tabs([":material/show_chart: Krzywa kapitału", ":material/water: Obsunięcia",
                                          ":material/timeline: Wskaźniki kroczące", ":material/calendar_month: Stopy miesięczne",
                                          ":material/stacked_line_chart: Wagi w czasie"])
with t_eq:
    view = st.segmented_control("Widok", ["twr", "value"], default="twr", required=True, key="ui_bt_view",
                                format_func={"twr": "Wynik inwestycyjny (indeks TWR = 100)", "value": "Wartość portfela z wpłatami"}.get)
    fig = go.Figure()
    for k in picked:
        r = results[k]
        y = 100 * (1 + r.twr_returns).cumprod() if view == "twr" else r.values
        fig.add_trace(go.Scatter(x=y.index, y=y, name=STRATEGY_LABELS[k], line=dict(color=colors[k], width=2)))
    if view == "value":
        inv = results[picked[0]].invested
        fig.add_trace(go.Scatter(x=inv.index, y=inv, name="Suma wpłat", line=dict(color=NAVY, dash="dot", width=2)))
    else:
        b = ctx.md.returns[payload["benchmark"]].reindex(first.twr_returns.index)
        fig.add_trace(go.Scatter(x=b.index, y=100 * (1 + b).cumprod(), name=f"Benchmark: {catalog.short_name(payload['benchmark'])}",
                                 line=dict(color="#8A94A6", dash="dash")))
    fig.update_yaxes(title="Indeks (start = 100)" if view == "twr" else "PLN", tickformat=",.0f")
    fig.update_xaxes(tickformat="%m.%Y")
    st.plotly_chart(style(fig, 440), key="bt_equity", config={"displaylogo": False})
    st.caption("Indeks TWR pokazuje wyłącznie wynik inwestycyjny; wartość portfela zawiera również efekt wpłat.")
with t_dd:
    fig = go.Figure()
    for k in picked:
        r = results[k]
        dd = pd.Series(drawdown_series(r.twr_returns.to_numpy())[1:], index=r.twr_returns.index)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=STRATEGY_LABELS[k], line=dict(color=colors[k], width=1.5),
                                 fill="tozeroy" if len(picked) == 1 else None, fillcolor=hex_to_rgba(RED, 0.15)))
    fig.update_yaxes(title="Obsunięcie od szczytu", tickformat=".0%")
    st.plotly_chart(style(fig, 380), key="bt_dd", config={"displaylogo": False})
with t_roll:
    c1, c2 = st.columns(2)
    vol_df = pd.DataFrame({STRATEGY_LABELS[k]: rolling_volatility(results[k].twr_returns, 12) for k in picked}).dropna(how="all")
    sh_df = pd.DataFrame({STRATEGY_LABELS[k]: rolling_sharpe(results[k].twr_returns, s.market.risk_free_rate, 36) for k in picked}).dropna(how="all")
    with c1:
        st.markdown("**Zmienność krocząca (12 mies.)**")
        st.plotly_chart(lines(vol_df, "Zmienność roczna", pct=True, colors=[colors[k] for k in picked]), key="bt_rvol",
                        config={"displaylogo": False})
    with c2:
        st.markdown("**Sharpe kroczący (36 mies.)**")
        if sh_df.empty:
            st.info("Za krótki okres testu dla okna 36 miesięcy.")
        else:
            st.plotly_chart(lines(sh_df, "Sharpe Ratio", colors=[colors[k] for k in picked]), key="bt_rsh",
                            config={"displaylogo": False})
with t_heat:
    k = st.selectbox("Strategia", picked, format_func=STRATEGY_LABELS.get, key="ui_bt_heat")
    mt = monthly_return_table(results[k].twr_returns)
    months = ["Sty", "Lut", "Mar", "Kwi", "Maj", "Cze", "Lip", "Sie", "Wrz", "Paź", "Lis", "Gru"]
    st.plotly_chart(heatmap(mt.to_numpy(), months, [str(y) for y in mt.index], "Miesiąc", "Rok", fmt=".1%",
                            height=max(300, 22 * len(mt)), colorscale=[[0, RED], [0.5, "#FFFFFF"], [1, "#0E9F9A"]], zmid=0)
                    .update_traces(zmin=None, zmax=None), key="bt_heat", config={"displaylogo": False})
with t_w:
    k = st.selectbox("Strategia", picked, format_func=STRATEGY_LABELS.get, key="ui_bt_weights")
    wh = results[k].weights.rename(columns=catalog.short_name)
    fig = go.Figure()
    for i, c in enumerate(wh.columns):
        fig.add_trace(go.Scatter(x=wh.index, y=wh[c], name=c, stackgroup="w", line=dict(width=0.5, color=SERIES[i % len(SERIES)])))
    fig.update_yaxes(tickformat=".0%", title="Waga docelowa")
    st.plotly_chart(style(fig, 380), key="bt_w", config={"displaylogo": False})
    st.caption("Wagi docelowe ustalane na początku miesiąca wyłącznie na podstawie danych z poprzednich miesięcy.")

export = pd.concat({STRATEGY_LABELS[k]: pd.DataFrame({"wartosc": r.values, "wplaty": r.invested, "stopa_twr": r.twr_returns,
                                                       "koszty": r.costs}) for k, r in results.items()}, axis=1)
c1, c2 = st.columns(2)
c1.download_button("Wyniki miesięczne (CSV)", export.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                   "fire_lab_backtest_miesieczny.csv", "text/csv", icon=":material/download:")
c2.download_button("Tabela wskaźników (CSV)", table.drop(columns=["key", "Krzywa"]).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                   "fire_lab_backtest_wskazniki.csv", "text/csv", icon=":material/download:")

with st.expander("Metodologia i ograniczenia backtestu", icon=":material/menu_book:"):
    st.markdown(
        "- **Walk-forward:** na każdej dacie rebalancingu parametry (μ, Σ) są estymowane na oknie "
        f"{s.backtest.estimation_months} mies. kończącym się miesiąc przed decyzją. Brak look-ahead bias.\n"
        "- **Konwencja:** wpłata i rebalancing na początku miesiąca, następnie stopa zwrotu miesiąca i opłata za zarządzanie.\n"
        "- **TWR vs XIRR:** TWR mierzy jakość strategii niezależnie od wpłat; XIRR – stopę ważoną kapitałem inwestora.\n"
        "- **Survivorship bias:** katalog obejmuje wyłącznie istniejące dziś ETF-y – fundusze zlikwidowane nie są uwzględnione.\n"
        "- Stopa wolna od ryzyka jest stała w całym okresie (brak historycznej krzywej stóp).\n"
        "- Wyniki historyczne nie gwarantują przyszłych wyników."
    )
    st.latex(r"r^{TWR}_t = \frac{V_t}{V_{t-1} + CF_t} - 1, \qquad \sum_t \frac{CF_t}{(1+XIRR)^{\tau_t}} = 0")
