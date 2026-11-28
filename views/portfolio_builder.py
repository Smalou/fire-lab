"""Strona 3 – Konstruktor portfela (Markowitz, Risk Parity, CAPM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from components.charts import allocation_donut, class_allocation, matrix_heatmap
from components.formatting import fmt_num, fmt_pct, polish_frame
from components.kpi import kpi
from components.layout import context_guard, data_badge, page_header
from components.validation import polish_errors
from components.theme import ASSET_CLASS_COLORS, GREY, NAVY, ORANGE, RED, TEAL, hex_to_rgba, style
from core import portfolio as pf
from core.models import OptimizationMethod
from services import analytics, catalog, state

state.init()
s = state.active()
page_header("Konstruktor portfela", "Zbuduj portfel metodami teorii Markowitza i sprawdź, skąd pochodzi jego ryzyko.")

METHOD_LABELS = {
    OptimizationMethod.MANUAL: "Ręczna alokacja",
    OptimizationMethod.EQUAL_WEIGHT: "Equal Weight",
    OptimizationMethod.MIN_VARIANCE: "Minimum Variance",
    OptimizationMethod.MAX_SHARPE: "Maximum Sharpe",
    OptimizationMethod.RISK_PARITY: "Risk Parity",
    OptimizationMethod.TARGET_RETURN: "Zadana stopa zwrotu",
}
METHOD_HELP = {
    OptimizationMethod.EQUAL_WEIGHT: "Każde aktywo ma wagę 1/N – brak estymacji, odporność na błędy prognoz.",
    OptimizationMethod.MIN_VARIANCE: "Portfel o najmniejszej wariancji: min wᵀΣw przy Σw = 1.",
    OptimizationMethod.MAX_SHARPE: "Portfel styczny: max (wᵀμ − r_f)/√(wᵀΣw).",
    OptimizationMethod.RISK_PARITY: "Równy udział każdego aktywa w ryzyku portfela (tylko pozycje długie).",
    OptimizationMethod.TARGET_RETURN: "Minimalna wariancja przy zadanej oczekiwanej (arytmetycznej) stopie zwrotu.",
    OptimizationMethod.MANUAL: "Wagi wpisujesz samodzielnie w tabeli.",
}

source = s.market.data_source
labels = catalog.asset_class_labels()
ds = st.session_state.get("csv_dataset")
if source == "csv" and ds:
    options = list(ds["prices"].columns)
    option_class = ds["classes"]
else:
    options = [i.id for i in catalog.instruments(source)]
    option_class = {i.id: i.asset_class for i in catalog.instruments(source)}

# ------------------------------------------------------------------ 1. composition
with st.container(border=True):
    head, cat_col = st.columns([4, 1], vertical_alignment="center")
    head.markdown("**1. Skład portfela**")
    with cat_col.popover("Katalog instrumentów", icon=":material/list_alt:", width="stretch"):
        st.caption("Konfigurowalny katalog (data/instruments.json). Daty notowań orientacyjne – faktyczną historię aplikacja "
                   "sprawdza po pobraniu danych. Katalog obejmuje wyłącznie instrumenty istniejące dziś (survivorship bias).")
        st.dataframe(catalog.catalog_frame(None if source == "csv" else source), hide_index=True, width="stretch")
    present_classes = [c for c in labels if c in set(option_class.values())]
    chosen_classes = st.pills("Klasy aktywów", present_classes, selection_mode="multi", default=present_classes,
                              format_func=labels.get, key="ui_pf_classes")
    filtered = [o for o in options if option_class.get(o) in (chosen_classes or present_classes)] or options
    with st.form("composition_form", border=False):
        default_sel = [a for a in s.portfolio.assets if a in options]
        sel = st.multiselect("Instrumenty w portfelu", sorted(set(filtered) | set(default_sel), key=options.index),
                             default=default_sel, format_func=lambda a: catalog.display_name(a),
                             help="Lista filtrowana klasami aktywów powyżej. Minimum 2 instrumenty.")
        if st.form_submit_button("Zatwierdź skład", icon=":material/check:"):
            if len(sel) < 2:
                st.error("Wybierz co najmniej dwa instrumenty.")
            else:
                w = np.full(len(sel), 1 / len(sel))
                new = s.model_copy(deep=True)
                new.portfolio = new.portfolio.model_copy(update={"assets": sel, "weights": list(w),
                                                                 "method": OptimizationMethod.MANUAL})
                if new.market.benchmark not in sel:
                    eq = [a for a in sel if option_class.get(a, "").endswith("equity")]
                    new.market.benchmark = eq[0] if eq else sel[0]
                state.save(new, "Zapisano skład portfela – ustawiono wagi równe; wybierz metodę alokacji poniżej.")
                st.rerun()

ctx = analytics.get_context(s)
data_badge(ctx)
if not context_guard(ctx):
    st.stop()

mom = ctx.moments
assets = ctx.assets
names = [catalog.short_name(a) for a in assets]
rf = s.market.risk_free_rate
p = s.portfolio

# ------------------------------------------------------------------ 2. method & weights
with st.container(border=True):
    st.markdown("**2. Metoda alokacji i ograniczenia**")
    with st.form("method_form", border=False):
        method = st.segmented_control("Metoda", list(OptimizationMethod), default=p.method, required=True,
                                      format_func=lambda m: METHOD_LABELS[m], key="ui_pf_method")
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        with c2.popover("Ograniczenia wag", icon=":material/tune:", width="stretch"):
            allow_short = st.toggle("Dopuść krótką sprzedaż", value=p.allow_short,
                                    help="Ujemne wagi. Risk Parity zawsze działa na pozycjach długich.")
            min_w = st.number_input("Minimalna waga (%)", -100.0, 100.0, p.min_weight * 100, 1.0)
            max_w = st.number_input("Maksymalna waga (%)", 1.0, 200.0, p.max_weight * 100, 1.0)
            lo, hi = pf.min_return_feasible(mom.mu), pf.max_return_feasible(mom.mu)
            tr_default = p.target_return if p.target_return is not None else (lo + hi) / 2
            target = st.number_input("Docelowa oczekiwana stopa (rocznie, %)", -50.0, 100.0, round(tr_default * 100, 2), 0.25,
                                     help=f"Zakres osiągalny bez krótkiej sprzedaży: {fmt_pct(lo, 2)} – {fmt_pct(hi, 2)}.")
        c1.caption(METHOD_HELP.get(method, ""))
        hist_growth = (1 + ctx.est_returns).cumprod()
        editor_df = pd.DataFrame({
            "Aktywo": assets,
            "Nazwa": [catalog.display_name(a, None) for a in assets],
            "Klasa": [labels.get(ctx.classes[a], ctx.classes[a]) for a in assets],
            "Waga (%)": np.round(np.array(p.weights) * 100, 2),
            "Historia": [hist_growth[a].iloc[::3].round(4).tolist() for a in assets],
        })
        edited = st.data_editor(
            editor_df, hide_index=True, width="stretch", key="ui_weights_editor", disabled=["Aktywo", "Nazwa", "Klasa", "Historia"],
            column_config={
                "Waga (%)": st.column_config.NumberColumn("Waga (%)", min_value=-100.0, max_value=200.0, step=0.5, format="localized",
                                                          help="Używane przy metodzie ręcznej. Suma musi wynosić 100%."),
                "Historia": st.column_config.LineChartColumn("Wzrost w okresie estymacji", width="medium"),
            },
        )
        normalize = st.toggle("Automatycznie przeskaluj wagi ręczne do 100%", value=False)
        go_btn = st.form_submit_button("Wyznacz i zatwierdź portfel", type="primary", icon=":material/calculate:")

    if go_btn:
        manual = edited["Waga (%)"].to_numpy(dtype=float) / 100
        if method == OptimizationMethod.MANUAL and manual.sum() != 0 and (normalize or abs(manual.sum() - 1) < 5e-4):
            manual = manual / manual.sum()  # explicit request, or rounding of the 2-decimal editor
        res = pf.optimize_portfolio(method, mom, rf, min_w / 100, max_w / 100, allow_short, target / 100, manual)
        if not res.success:
            st.error(f"Optymalizacja nie powiodła się: {res.message}", icon=":material/error:")
        else:
            try:
                if method == OptimizationMethod.MANUAL:
                    pf.check_weights(res.weights, allow_short)
                new = s.model_copy(deep=True)
                new.portfolio = new.portfolio.model_validate({
                    **new.portfolio.model_dump(), "weights": [float(x) for x in res.weights], "method": method,
                    "allow_short": allow_short, "min_weight": min_w / 100, "max_weight": max_w / 100,
                    "target_return": target / 100 if method == OptimizationMethod.TARGET_RETURN else p.target_return})
                state.save(new, f"Zastosowano portfel: {METHOD_LABELS[method]} (iteracje: {res.iterations}).")
                st.rerun()
            except ValidationError as exc:
                for msg in polish_errors(exc):
                    st.error(f"Niepoprawne wagi – {msg}", icon=":material/error:")
            except pf.OptimizationError as exc:
                st.error(f"Niepoprawne wagi: {exc}", icon=":material/error:")

# ------------------------------------------------------------------ KPIs
w = ctx.weights
st_ = ctx.stats
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Oczekiwana stopa (arytm.)", fmt_pct(st_["expected_arithmetic"], 2),
        delta=f"geometryczna {fmt_pct(st_['geometric_nominal'], 2)}", delta_color="off",
        help="Średnia arytmetyczna ×12 z okresu estymacji. Do projekcji długoterminowych używana jest stopa geometryczna.")
with c2:
    kpi("Zmienność roczna", fmt_pct(st_["volatility"], 2), help="Odchylenie standardowe miesięcznych stóp × √12.",
        chart=list(pd.Series(st_["hist_monthly"]).rolling(12).std().dropna() * np.sqrt(12)), chart_type="line")
with c3:
    kpi("Sharpe Ratio", fmt_num(st_["sharpe"], 2), help=f"(E[R] − r_f) / σ przy r_f = {fmt_pct(rf, 2)}.")
with c4:
    kpi("Realna stopa netto", fmt_pct(st_["geometric_real_net"], 2), help="Geometryczna, po inflacji i kosztach – używana w planie FIRE.")
st.caption(f"Estymacja: {mom.n_obs} mies. ({mom.start:%m.%Y} – {mom.end:%m.%Y}), kowariancja: "
           f"{'Ledoit–Wolf (intensywność ' + fmt_num(mom.shrinkage_intensity, 2) + ')' if mom.method == 'ledoit_wolf' else 'próbkowa'}. "
           "Parametry z przeszłości nie gwarantują przyszłych wyników.")

# ------------------------------------------------------------------ 3. analyses
tab_front, tab_struct, tab_corr, tab_capm = st.tabs([
    ":material/scatter_plot: Granica efektywna", ":material/donut_large: Struktura i ryzyko",
    ":material/grid_on: Korelacje i kowariancje", ":material/show_chart: CAPM i SML"])

min_b, max_b, short = p.min_weight, p.max_weight, p.allow_short


@st.dialog("Szczegóły portfela", width="large", icon=":material/donut_large:")
def portfolio_details(label: str, weights: np.ndarray) -> None:
    ret, vol = pf.portfolio_return(weights, mom.mu), pf.portfolio_volatility(weights, mom.cov)
    rc = pf.risk_contributions(weights, mom.cov)
    c1, c2, c3 = st.columns(3)
    c1.metric("Oczekiwana stopa", fmt_pct(ret, 2), border=True)
    c2.metric("Zmienność", fmt_pct(vol, 2), border=True)
    c3.metric("Sharpe", fmt_num(pf.sharpe(weights, mom.mu, mom.cov, rf), 2), border=True)
    df = pd.DataFrame({"Aktywo": names, "Waga": weights, "Udział w ryzyku": rc / vol if vol > 0 else 0})
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "Waga": st.column_config.ProgressColumn("Waga", min_value=min(0.0, float(weights.min())), max_value=1, format="percent"),
        "Udział w ryzyku": st.column_config.ProgressColumn("Udział w ryzyku", min_value=0, max_value=1, format="percent")})
    st.caption(label)


@st.fragment
def frontier_section() -> None:
    try:
        fr = analytics.frontier(mom.mu, mom.cov, min_b, max_b, short)
    except pf.OptimizationError as exc:
        st.error(f"Nie udało się wyznaczyć granicy efektywnej: {exc}")
        return
    if fr.failures:
        st.warning(f"{fr.failures} punktów granicy nie osiągnęło zbieżności i zostało pominiętych.")
    cw, cr, cv = analytics.random_cloud(mom.mu, mom.cov, max(min_b, 0.0), max_b)
    specials = {"Twój portfel": w}
    for m in (OptimizationMethod.MIN_VARIANCE, OptimizationMethod.MAX_SHARPE, OptimizationMethod.RISK_PARITY, OptimizationMethod.EQUAL_WEIGHT):
        r = pf.optimize_portfolio(m, mom, rf, min_b, max_b, short)
        if r.success:
            specials[METHOD_LABELS[m]] = r.weights
        else:
            st.caption(f":material/error: {METHOD_LABELS[m]}: {r.message}")
    registry: list[tuple[str, np.ndarray]] = []

    def add_points(ws_, label_fn):
        start = len(registry)
        for i, ww in enumerate(ws_):
            registry.append((label_fn(i), ww))
        return list(range(start, len(registry)))

    fig = go.Figure()
    ids = add_points(cw, lambda i: f"Portfel losowy #{i + 1}")
    sharpe_cloud = (cr - rf) / cv
    fig.add_trace(go.Scatter(x=cv, y=cr, mode="markers", name="Portfele losowe (long-only)", customdata=ids,
                             marker=dict(size=5, color=sharpe_cloud, colorscale=[[0, "#DDE3EA"], [1, TEAL]], opacity=0.55,
                                         colorbar=dict(title="Sharpe", thickness=10)),
                             hovertemplate="σ %{x:.2%}<br>E[R] %{y:.2%}<extra>Losowy</extra>"))
    ids = add_points(fr.weights, lambda i: f"Granica efektywna – punkt {i + 1}")
    fig.add_trace(go.Scatter(x=fr.vols, y=fr.returns, mode="lines+markers", name="Granica efektywna", customdata=ids,
                             line=dict(color=NAVY, width=3), marker=dict(size=6),
                             hovertemplate="σ %{x:.2%}<br>E[R] %{y:.2%}<extra>Granica</extra>"))
    if "Maximum Sharpe" in specials and not short:
        tw = specials["Maximum Sharpe"]
        tv, tr = pf.portfolio_volatility(tw, mom.cov), pf.portfolio_return(tw, mom.mu)
        if tr > rf:
            xs = np.linspace(0, max(fr.vols.max(), tv) * 1.1, 20)
            fig.add_trace(go.Scatter(x=xs, y=rf + (tr - rf) / tv * xs, mode="lines", name="Capital Market Line",
                                     line=dict(color=ORANGE, dash="dash", width=2), hoverinfo="skip"))
    for i, a in enumerate(assets):
        e = np.zeros(len(assets))
        e[i] = 1
        if e[i] <= max_b + 1e-9:
            idx = add_points([e], lambda _: f"100% {names[i]}")
        else:
            idx = [None]
        fig.add_trace(go.Scatter(x=[np.sqrt(mom.cov[i, i])], y=[mom.mu[i]], mode="markers+text", text=[names[i]],
                                 textposition="top center", name=names[i], showlegend=False, customdata=idx, cliponaxis=False,
                                 marker=dict(size=10, symbol="diamond", color=ASSET_CLASS_COLORS.get(ctx.classes[a], GREY)),
                                 hovertemplate=f"{names[i]}<br>σ %{{x:.2%}}<br>E[R] %{{y:.2%}}<extra></extra>"))
    sym = {"Twój portfel": ("star", NAVY, 20), "Minimum Variance": ("square", "#1D4E89", 12),
           "Maximum Sharpe": ("triangle-up", "#7A5EA8", 14), "Risk Parity": ("hexagon", "#6FA37A", 13),
           "Equal Weight": ("x", "#5B8DD6", 12)}
    for label, ww in specials.items():
        idx = add_points([ww], lambda _: label)
        s_, c_, z_ = sym.get(label, ("circle", NAVY, 12))
        fig.add_trace(go.Scatter(x=[pf.portfolio_volatility(ww, mom.cov)], y=[pf.portfolio_return(ww, mom.mu)], mode="markers",
                                 name=label, customdata=idx, marker=dict(size=z_, symbol=s_, color=c_, line=dict(width=1, color="white")),
                                 hovertemplate=f"{label}<br>σ %{{x:.2%}}<br>E[R] %{{y:.2%}}<extra></extra>"))
    fig.update_xaxes(title="Zmienność roczna (σ)", tickformat=".0%")
    fig.update_yaxes(title="Oczekiwana stopa zwrotu (arytm., rocznie)", tickformat=".0%")
    style(fig, 520, hovermode="closest", dragmode="select")
    event = st.plotly_chart(fig, key="frontier_chart", on_select="rerun", selection_mode=("points", "box", "lasso"),
                            config={"displaylogo": False})
    st.caption("Kliknij punkt (lub zaznacz obszar) – szczegóły wybranego portfela pojawią się poniżej. "
               "CML zakłada możliwość pożyczania i lokowania po stopie wolnej od ryzyka i jest rysowana tylko bez krótkiej sprzedaży.")
    pts = event.selection.points if event and event.selection else []
    chosen = None
    for pt in pts:
        cd = pt.get("customdata")
        if isinstance(cd, list):
            cd = cd[0] if cd else None
        if cd is not None and 0 <= int(cd) < len(registry):
            chosen = registry[int(cd)]
            break
    if chosen is None:
        st.info("Nie wybrano portfela na wykresie – kliknij dowolny punkt.", icon=":material/touch_app:")
        return
    label, cw_ = chosen
    with st.container(border=True):
        st.markdown(f"**Wybrany portfel: {label}**")
        ret, vol = pf.portfolio_return(cw_, mom.mu), pf.portfolio_volatility(cw_, mom.cov)
        c1, c2, c3 = st.columns(3)
        c1.metric("Oczekiwana stopa", fmt_pct(ret, 2), delta=fmt_pct(ret - st_["expected_arithmetic"], 2, signed=True), border=True)
        c2.metric("Zmienność", fmt_pct(vol, 2), delta=fmt_pct(vol - st_["volatility"], 2, signed=True), delta_color="inverse", border=True)
        c3.metric("Sharpe", fmt_num(pf.sharpe(cw_, mom.mu, mom.cov, rf), 2), border=True)
        bars = go.Figure(go.Bar(x=names, y=cw_, marker_color=[ASSET_CLASS_COLORS.get(ctx.classes[a], GREY) for a in assets],
                                hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        bars.update_yaxes(tickformat=".0%", title="Waga")
        st.plotly_chart(style(bars, 260, hovermode="closest"), key="sel_weights", config={"displaylogo": False})
        b1, b2 = st.columns(2)
        if b1.button("Szczegóły portfela", icon=":material/open_in_full:", key="sel_details"):
            portfolio_details(label, cw_)
        if b2.button("Zastosuj ten portfel do całego scenariusza FIRE", type="primary", icon=":material/done_all:", key="sel_apply"):
            new = s.model_copy(deep=True)
            new.portfolio = new.portfolio.model_copy(update={"weights": [float(x) for x in pf.clean_weights(cw_, short)],
                                                             "method": OptimizationMethod.MANUAL})
            state.save(new, f"Zastosowano portfel „{label}” – zależne analizy oznaczono jako nieaktualne.")
            st.rerun(scope="app")


with tab_front:
    frontier_section()

with tab_struct:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Struktura portfela**")
        st.plotly_chart(allocation_donut(assets, w, ctx.classes), key="donut", config={"displaylogo": False})
        ca = class_allocation(assets, w, ctx.classes)
        st.dataframe(pd.DataFrame({"Klasa aktywów": ca.index, "Udział": ca.values}), hide_index=True, width="stretch",
                     column_config={"Udział": st.column_config.ProgressColumn("Udział", min_value=0, max_value=1, format="percent")})
    with c2:
        st.markdown("**Kontrybucja do ryzyka portfela**", help="RC_i = w_i (Σw)_i / σ_p – udziały sumują się do 100% zmienności.")
        rc = pf.risk_contributions(w, mom.cov)
        vol = pf.portfolio_volatility(w, mom.cov)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=names, y=w, name="Waga", marker_color=hex_to_rgba(NAVY, 0.55)))
        fig.add_trace(go.Bar(x=names, y=rc / vol if vol > 0 else rc, name="Udział w ryzyku",
                             marker_color=[RED if x > y + 0.05 else TEAL for x, y in zip(rc / vol, w)]))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 320, barmode="group", hovermode="x"), key="risk_contrib", config={"displaylogo": False})
        st.caption("Czerwony słupek: aktywo wnosi wyraźnie więcej ryzyka niż wynosi jego waga.")
    st.markdown("**Oczekiwana stopa zwrotu vs zmienność aktywów**")
    recent = ctx.est_returns.iloc[-24:]
    table = pd.DataFrame({
        "Aktywo": names,
        "Waga": w,
        "E[R] rocznie (arytm.)": mom.mu,
        "Zmienność": mom.vol,
        "Udział w ryzyku": (rc / vol) if vol > 0 else 0,
        "Ostatnie 24 mies.": [recent[a].round(4).tolist() for a in assets],
    })
    st.dataframe(polish_frame(table, pct=["E[R] rocznie (arytm.)", "Zmienność"]), hide_index=True, width="stretch", column_config={
        "Waga": st.column_config.ProgressColumn("Waga", min_value=min(0.0, float(w.min())), max_value=1, format="percent"),
        "Udział w ryzyku": st.column_config.ProgressColumn("Udział w ryzyku", min_value=min(0.0, float((rc / vol).min())), max_value=1,
                                                           format="percent"),
        "Ostatnie 24 mies.": st.column_config.BarChartColumn("Miesięczne stopy (24 mies.)", y_min=-0.15, y_max=0.15),
    })
    export = pd.DataFrame({"ticker": assets, "nazwa": [catalog.display_name(a) for a in assets],
                           "klasa": [ctx.classes[a] for a in assets], "waga": w, "oczekiwana_stopa_arytm": mom.mu,
                           "zmiennosc": mom.vol, "udzial_w_ryzyku": rc / vol if vol > 0 else 0})
    st.download_button("Eksportuj strukturę portfela (CSV)", export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       f"fire_lab_portfel_{s.name}.csv", "text/csv", icon=":material/download:")

with tab_corr:
    mode = st.segmented_control("Macierz", ["corr", "cov"], default="corr", required=True, key="ui_matrix_mode",
                                format_func={"corr": "Korelacje", "cov": "Kowariancje (roczne)"}.get)
    if mode == "corr":
        st.plotly_chart(matrix_heatmap(mom.corr, names, ".2f"), key="corr", config={"displaylogo": False})
    else:
        st.plotly_chart(matrix_heatmap(mom.cov, names, ".4f", diverging=False), key="cov", config={"displaylogo": False})
    eig = np.linalg.eigvalsh(mom.cov)
    st.caption(f"Kontrola PSD: najmniejsza wartość własna = {fmt_num(eig.min(), 6)} "
               f"({'macierz dodatnio półokreślona' if eig.min() >= -1e-12 else 'wymagana korekta'}). "
               f"Wskaźnik uwarunkowania: {fmt_num(eig.max() / max(eig.min(), 1e-16), 0)}.")

with tab_capm:
    show_capm = st.toggle("Pokaż analizę CAPM i linię SML", value=True, key="ui_capm")
    if show_capm:
        bench_opts = assets
        bench = st.selectbox("Portfel rynkowy (proxy)", bench_opts,
                             index=bench_opts.index(s.market.benchmark) if s.market.benchmark in bench_opts else 0,
                             format_func=catalog.short_name, key="ui_capm_bench",
                             help="CAPM wymaga portfela rynkowego; przybliżamy go wybranym indeksem akcji.")
        capm = pf.capm_analysis(ctx.est_returns, ctx.est_returns[bench], rf)
        fig = go.Figure()
        xs = np.linspace(min(0, capm.betas.min()) - 0.1, capm.betas.max() + 0.2, 20)
        fig.add_trace(go.Scatter(x=xs, y=rf + xs * (capm.market_return - rf), name="Security Market Line",
                                 line=dict(color=ORANGE, dash="dash")))
        fig.add_trace(go.Scatter(x=capm.betas, y=capm.expected, mode="markers+text", text=names, textposition="top center",
                                 name="Aktywa", marker=dict(size=11, color=TEAL),
                                 hovertemplate="β %{x:.2f}<br>E[R] %{y:.2%}<extra>%{text}</extra>"))
        fig.update_xaxes(title="Beta względem portfela rynkowego")
        fig.update_yaxes(title="Średnia stopa zwrotu (rocznie)", tickformat=".0%")
        st.plotly_chart(style(fig, 400, hovermode="closest"), key="sml", config={"displaylogo": False})
        capm_df = pd.DataFrame({"Aktywo": names, "Beta": capm.betas, "E[R] historyczna": capm.expected,
                                "E[R] wg CAPM": capm.sml, "Alfa Jensena": capm.alphas})
        st.dataframe(polish_frame(capm_df, pct=["E[R] historyczna", "E[R] wg CAPM", "Alfa Jensena"], num={"Beta": 2}),
                     hide_index=True, width="stretch")
        with st.expander("Założenia CAPM", icon=":material/menu_book:"):
            st.latex(r"E[R_i] = r_f + \beta_i\,(E[R_M]-r_f), \qquad \beta_i = \frac{\mathrm{Cov}(R_i,R_M)}{\mathrm{Var}(R_M)}")
            st.markdown("Model zakłada m.in. efektywny rynek, jednorodne oczekiwania i możliwość pożyczania po stopie wolnej od ryzyka. "
                        "Alfa wyznaczona na danych historycznych jest obarczona dużym błędem estymacji.")

st.page_link("views/strategy_lab.py", label="Przetestuj portfel historycznie (walk-forward)", icon=":material/science:")
