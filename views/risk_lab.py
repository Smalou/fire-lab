"""Strona 5 – Laboratorium ryzyka (Monte Carlo, VaR/ES, stres testy, wrażliwość)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import cdf_chart, fan_chart, heatmap, histogram
from components.formatting import fmt_age, fmt_money, fmt_num, fmt_pct, polish_frame
from components.kpi import kpi
from components.layout import context_guard, data_badge, page_header, stale_notice
from components.theme import GREY, ICONS, NAVY, ORANGE, RED, TEAL, style
from core.fire import build_plan
from core.lifecycle import deflate_returns, run_lifecycle, summarize
from core.models import RetirementRule, SimulationModel, WithdrawalStrategy
from core.risk import (HISTORICAL_SCENARIOS, HYPOTHETICAL_SCENARIOS, apply_stress, historical_stress_path,
                       hypothetical_stress_path, sensitivity_analysis, success_heatmap)
from services import analytics, state

state.init()
s = state.active()
page_header("Laboratorium ryzyka", "Symulacja Monte Carlo całego cyklu: akumulacja, przejście na FIRE i wypłaty do końca horyzontu.")
ctx = analytics.get_context(s)
data_badge(ctx)
if not context_guard(ctx):
    st.stop()

sim = s.simulation
PATHS = [100, 250, 500, 1000, 2000, 5000, 10000]
CONF = [0.90, 0.95, 0.975, 0.99]

with st.form("mc_form"):
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    model = c1.segmented_control("Model stóp zwrotu", list(SimulationModel), default=sim.model, required=True,
                                 format_func={SimulationModel.PARAMETRIC: "Parametryczny",
                                              SimulationModel.BOOTSTRAP: "Block bootstrap"}.get,
                                 help="Parametryczny: rozkład normalny/t-Studenta logarytmicznych stóp aktywów z macierzą kowariancji. "
                                      "Bootstrap: losowanie bloków historycznych miesięcy (zachowuje korelacje i autokorelację).")
    n_paths = c2.select_slider("Liczba ścieżek", PATHS, value=min(PATHS, key=lambda x: abs(x - sim.n_paths)),
                               format_func=lambda x: fmt_num(x))
    seed = c3.number_input("Ziarno losowe", 0, 2**31 - 1, sim.seed, 1, help="Te same parametry i ziarno = identyczne wyniki.")
    horizon = c4.number_input("Horyzont (wiek)", s.profile.fire_age + 1, 120, s.profile.life_horizon_age, 1)
    c1, c2, c3 = st.columns(3)
    fat = c1.toggle("Grube ogony (t-Studenta)", value=sim.fat_tails, disabled=False,
                    help="Tylko model parametryczny: rozkład t z tą samą macierzą kowariancji – częstsze skrajne miesiące.")
    shrink = c2.toggle("Shrinkage kowariancji (Ledoit–Wolf)", value=sim.shrinkage)
    conf = c3.select_slider("Poziom ufności VaR / ES", CONF, value=min(CONF, key=lambda x: abs(x - sim.var_confidence)),
                            format_func=lambda x: fmt_pct(x, 1))
    with st.popover("Ustawienia zaawansowane", icon=":material/tune:"):
        dof = st.number_input("Stopnie swobody t-Studenta", 2.5, 100.0, sim.t_dof, 0.5)
        block = st.number_input("Długość bloku bootstrap (mies.)", 1, 60, sim.block_months, 1)
        haircut = st.number_input("Korekta oczekiwanej stopy (p.p. rocznie, w dół)", -5.0, 10.0, sim.return_haircut * 100, 0.25,
                                  help="Ostrożnościowe obniżenie historycznej średniej (np. 1 p.p.).")
        var_h = st.number_input("Horyzont VaR / ES (lata)", 1, 30, sim.var_horizon_years, 1)
    run = st.form_submit_button("Uruchom symulację Monte Carlo", type="primary", icon=":material/play_arrow:")

if run:
    new = s.model_copy(deep=True)
    new.simulation = new.simulation.model_validate({**sim.model_dump(), "model": model, "n_paths": n_paths, "seed": int(seed),
                                                    "fat_tails": fat, "shrinkage": shrink, "var_confidence": conf, "t_dof": dof,
                                                    "block_months": int(block), "return_haircut": haircut / 100,
                                                    "var_horizon_years": int(var_h)})
    new.profile = new.profile.model_validate({**new.profile.model_dump(), "life_horizon_age": int(horizon)})
    try:
        state.save(new, None)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Niepoprawne parametry: {exc}")
        st.stop()
    s = state.active()
    ctx = analytics.get_context(s)
    bar = st.progress(0.0, text="Start symulacji")
    with st.status("Symulacja Monte Carlo", expanded=True) as status:
        st.write(f"Estymacja na {ctx.moments.n_obs} mies. · {fmt_num(n_paths)} ścieżek · "
                 f"{(s.profile.life_horizon_age - s.profile.current_age) * 12} mies. horyzontu")
        result = analytics.run_monte_carlo(ctx, lambda f, t: bar.progress(min(f, 1.0), text=t))
        status.update(label="Symulacja zakończona", state="complete", expanded=False)
    state.store_result(s, "montecarlo", result)
    st.toast("Zapisano wyniki Monte Carlo.", icon=":material/check_circle:")

res, fresh, at = state.get_result(s, "montecarlo")
if res is None:
    st.info("Ustaw parametry i uruchom symulację. Prawdopodobieństwa pojawią się po obliczeniach – aplikacja nie pokazuje "
            "wartości szacunkowych bez symulacji.", icon=":material/info:")
    st.stop()
stale_notice(at, fresh)
if res.get("note"):
    st.info(res["note"])
summ = res["summary"]
var = res["var"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("P(FIRE w docelowym wieku)", fmt_pct(summ["p_fire_by_target_age"], 1),
        help=f"Odsetek ścieżek, na których realny majątek osiąga cel do wieku {s.profile.fire_age} lat (przy kontynuacji akumulacji).",
        delta=f"kiedykolwiek do wieku {s.fire.max_work_age}: {fmt_pct(summ['p_fire_ever'], 0)}", delta_color="off")
with c2:
    kpi("Mediana wieku FIRE", fmt_age(summ["median_fire_age"]), help="Mediana wieku pierwszego osiągnięcia celu (ścieżki, które go osiągają).")
with c3:
    kpi("P(utrzymania kapitału)", fmt_pct(summ["p_not_depleted"], 1),
        help=f"Odsetek ścieżek, na których portfel nie wyczerpuje się do wieku {s.profile.life_horizon_age} lat.")
with c4:
    kpi("P(sukcesu planu)", fmt_pct(summ["p_success"], 1),
        help="FIRE osiągnięte, kapitał niewyczerpany i wydatki pokryte co najmniej w progu minimalnym w każdym miesiącu.",
        delta=f"pokrycie wydatków: {fmt_pct(summ['p_spending_covered'], 0)}", delta_color="off")
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi(f"VaR {fmt_pct(var.confidence, 1)} / {var.horizon_years} r.", fmt_pct(var.var, 1), delta=f"{fmt_money(-var.var_amount, compact=True)}",
        delta_color="normal", help="Strata stopy zwrotu portfela (bez przepływów) przekroczona z prawd. 1−α w horyzoncie H. "
                                    "Kwota odniesiona do obecnego kapitału.")
with c2:
    kpi(f"Expected Shortfall {fmt_pct(var.confidence, 1)}", fmt_pct(var.es, 1), delta=f"{fmt_money(-var.es_amount, compact=True)}",
        delta_color="normal", help="Średnia strata w najgorszych (1−α) scenariuszach – miara ogona rozkładu.")
with c3:
    kpi("Mediana majątku końcowego", fmt_money(summ["median_terminal_wealth"], compact=True), help="Realnie, w dzisiejszych złotych.")
with c4:
    kpi("Ścieżki / model", f"{fmt_num(res['n_paths'])}", delta=f"{'bootstrap' if res['model'] == 'bootstrap' else 'parametryczny'} · ziarno {res['seed']}",
        delta_color="off")

if s.household.enabled:
    k1, k2 = st.columns([1, 3], vertical_alignment="center")
    with k1:
        kpi("P(luki płynnościowej)", fmt_pct(summ.get("p_illiquid_shortfall", 0.0), 1), icon=":material/lock_clock:",
            help="Odsetek ścieżek, na których po FIRE zabrakło pieniędzy na wydatki, choć część majątku była zablokowana "
                 "na IKE, IKZE, PPK lub PPE przed wiekiem dostępu.")
    with k2:
        st.markdown("Majątek liczony **po podatkach** (Belka, IKZE 10%, OKI, spread kruszców) z blokadami wieku dostępu "
                    "do kont emerytalnych.")
        st.page_link("views/household.py", label="Szczegóły: Majątek i konta", icon=ICONS["household"])
    for msg in res.get("capped", []):
        st.caption(f":material/warning: {msg}")

tabs = st.tabs([":material/ssid_chart: Wachlarz", ":material/bar_chart: Rozkłady", ":material/table: Percentyle",
                ":material/grid_on: Mapa sukcesu", ":material/tune: Wrażliwość", ":material/bolt: Scenariusze kryzysowe"])

with tabs[0]:
    log_scale = st.toggle("Skala logarytmiczna", key="ui_mc_log", help="Ułatwia odczyt dolnych percentyli przy szerokim rozkładzie.")
    fan = fan_chart(res["ages"], np.maximum(res["wealth_percentiles"], 1.0) if log_scale else res["wealth_percentiles"],
                    res["target"], s.profile.fire_age)
    if log_scale:
        fan.update_yaxes(type="log")
    st.plotly_chart(fan, key="mc_fan", config={"displaylogo": False})
    if st.toggle("Pokaż przykładowe ścieżki", key="ui_mc_paths"):
        fig = go.Figure()
        for i, path in enumerate(res["sample_paths"]):
            fig.add_trace(go.Scatter(x=res["ages"], y=path, mode="lines", line=dict(width=1, color=TEAL), opacity=0.35,
                                     showlegend=False, hovertemplate="%{y:,.0f} zł"))
        fig.add_hline(y=res["target"], line=dict(color=ORANGE, dash="dash"))
        fig.update_yaxes(tickformat=",.0f", ticksuffix=" zł")
        st.plotly_chart(style(fig, 360, hovermode="closest"), key="mc_paths", config={"displaylogo": False})

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rozkład wieku osiągnięcia FIRE**")
        fa = res["fire_ages"]
        st.plotly_chart(histogram(fa, "Wiek", ORANGE, nbins=40, money=False,
                                  vlines={"Docelowy wiek": s.profile.fire_age}), key="mc_fire_age", config={"displaylogo": False})
        st.caption(f"{fmt_pct(np.mean(np.isnan(fa)), 1)} ścieżek nie osiąga celu do wieku {s.fire.max_work_age} lat.")
    with c2:
        st.markdown("**Rozkład majątku w docelowym wieku FIRE**")
        st.plotly_chart(histogram(res["wealth_at_fire"], "Majątek realny", TEAL, vlines={"Cel FIRE": res["target"]}),
                        key="mc_w_fire", config={"displaylogo": False})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dystrybuanta majątku końcowego**")
        st.plotly_chart(cdf_chart(res["terminal_wealth"], "Majątek realny na końcu horyzontu"), key="mc_cdf",
                        config={"displaylogo": False})
        st.caption(f"P(wyczerpania kapitału) = {fmt_pct(1 - summ['p_not_depleted'], 1)}")
    with c2:
        st.markdown("**Rozkład wieku wyczerpania kapitału**")
        d = res["depletion_ages"]
        if np.isfinite(d).any():
            st.plotly_chart(histogram(d, "Wiek wyczerpania", RED, nbins=30, money=False), key="mc_depl", config={"displaylogo": False})
        else:
            st.success("Na żadnej ścieżce kapitał się nie wyczerpał.")
    st.markdown(f"**Rozkład {var.horizon_years}-letniej stopy zwrotu portfela (bez przepływów)**")
    hr = res["horizon_returns"]
    fig = histogram(hr, f"Stopa zwrotu w {var.horizon_years} r.", NAVY, money=False, vlines={f"−VaR {fmt_pct(var.confidence, 1)}": -var.var})
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, key="mc_hr", config={"displaylogo": False})

with tabs[2]:
    ages = res["ages"]
    pick = [a for a in ages if (a - ages[0]) % 5 == 0 or a in (s.profile.fire_age, ages[-1])]
    idx = [int(a - ages[0]) for a in pick]
    pct = res["wealth_percentiles"]
    ptab = pd.DataFrame({"Wiek": pick, **{f"P{p}": pct[i, idx] for i, p in enumerate((5, 25, 50, 75, 95))}})
    st.dataframe(polish_frame(ptab, money=[f"P{p}" for p in (5, 25, 50, 75, 95)]), hide_index=True, width="stretch")
    full = pd.DataFrame({"wiek": ages, **{f"P{p}": pct[i] for i, p in enumerate((5, 25, 50, 75, 95))}})
    paths_df = pd.DataFrame({"majatek_koncowy_realny": res["terminal_wealth"], "wiek_fire": res["fire_ages"],
                             "wiek_wyczerpania": res["depletion_ages"], "majatek_w_wieku_fire": res["wealth_at_fire"]})
    c1, c2 = st.columns(2)
    c1.download_button("Percentyle majątku (CSV)", full.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       "fire_lab_mc_percentyle.csv", "text/csv", icon=":material/download:")
    c2.download_button("Wyniki wszystkich ścieżek (CSV)", paths_df.to_csv(index_label="sciezka", sep=";", decimal=",").encode("utf-8-sig"),
                       "fire_lab_mc_sciezki.csv", "text/csv", icon=":material/download:")
    with st.expander("Definicje VaR i ES", icon=":material/menu_book:"):
        st.latex(r"\mathrm{VaR}_\alpha = -q_{1-\alpha}\left(R_H\right), \qquad \mathrm{ES}_\alpha = -\mathbb{E}\left[R_H \mid R_H \le q_{1-\alpha}(R_H)\right]")
        st.markdown(f"R_H to skumulowana stopa zwrotu portfela w horyzoncie **H = {var.horizon_years} r.** bez wpłat i wypłat, "
                    f"poziom ufności **α = {fmt_pct(var.confidence, 1)}**, rozkład z {fmt_num(res['n_paths'])} symulowanych ścieżek.")

sub_n = res["nominal_subset"].shape[0]


@st.fragment
def heatmap_section() -> None:
    st.caption(f"Siatka: wiek FIRE × wydatki po FIRE. Stały moment przejścia na FIRE, te same {fmt_num(sub_n)} ścieżki "
               "dla każdej komórki (common random numbers).")
    metric = st.segmented_control("Miara", ["p_success", "p_not_depleted"], default="p_success", required=True, key="ui_hm_metric",
                                  format_func={"p_success": "Sukces planu", "p_not_depleted": "Utrzymanie kapitału"}.get)
    data, fresh_h, at_h = state.get_result(s, "montecarlo", ":heatmap")
    if st.button("Oblicz mapę prawdopodobieństwa", icon=":material/grid_on:", key="hm_run"):
        ages = [a for a in range(s.profile.fire_age - 6, s.profile.fire_age + 7, 3)
                if s.profile.current_age < a < min(s.profile.life_horizon_age, 100)]
        mults = [0.8, 0.9, 1.0, 1.1, 1.2]
        with st.spinner("Przeliczanie siatki scenariuszy…"):
            out = success_heatmap(res["nominal_subset"], s, ctx.stats["geometric_real_net"], ages, mults,
                                  analytics.income_yield_or_none(ctx), res.get("gold_nominal_subset"))
        data = {"ages": ages, "mults": mults, **out}
        state.store_result(s, "montecarlo", data, ":heatmap")
        fresh_h = True
    if data is None:
        st.info("Kliknij przycisk, aby policzyć mapę (ok. kilku sekund).")
        return
    if not fresh_h:
        st.warning("Mapa jest nieaktualna względem bieżących parametrów.")
    yl = [fmt_money(s.profile.retirement_monthly_expenses * m) + "/mies." for m in data["mults"]]
    st.plotly_chart(heatmap(data[metric], [str(a) for a in data["ages"]], yl, "Wiek przejścia na FIRE", "Wydatki po FIRE"),
                    key="mc_heat", config={"displaylogo": False})


@st.fragment
def sensitivity_section() -> None:
    st.caption("Analiza „jeden czynnik naraz” na tych samych ścieżkach losowych. Długość słupka = wpływ parametru na wynik.")
    data, fresh_s, _ = state.get_result(s, "montecarlo", ":sens")
    metric = st.segmented_control("Miara", ["p_success", "p_not_depleted"], default="p_success", required=True, key="ui_sens_metric",
                                  format_func={"p_success": "Sukces planu", "p_not_depleted": "Utrzymanie kapitału"}.get)
    if st.button("Uruchom analizę wrażliwości", icon=":material/tune:", key="sens_run"):
        with st.spinner("Przeliczanie wariantów…"):
            df = sensitivity_analysis(res["nominal_subset"], s, ctx.stats["geometric_real_net"],
                                      analytics.income_yield_or_none(ctx), metric, res.get("gold_nominal_subset"))
        data = {"df": df, "metric": metric}
        state.store_result(s, "montecarlo", data, ":sens")
        fresh_s = True
    if data is None:
        st.info("Kliknij przycisk, aby uruchomić analizę.")
        return
    if not fresh_s:
        st.warning("Analiza jest nieaktualna względem bieżących parametrów.")
    df = data["df"]
    if data["metric"] != metric:
        st.caption("Wynik dotyczy poprzednio wybranej miary – uruchom ponownie, aby przeliczyć.")
    base = df["bazowy"].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["parametr"], x=df["niekorzystny"] - base, base=base, orientation="h", name="Wariant niekorzystny",
                         marker_color=RED, hovertemplate="%{x:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(y=df["parametr"], x=df["korzystny"] - base, base=base, orientation="h", name="Wariant korzystny",
                         marker_color=TEAL, hovertemplate="%{x:.1%}<extra></extra>"))
    fig.add_vline(x=base, line=dict(color=NAVY, dash="dot"), annotation_text=f"Bazowo {fmt_pct(base, 1)}")
    fig.update_xaxes(tickformat=".0%", title="Prawdopodobieństwo")
    st.plotly_chart(style(fig, 380, barmode="overlay", hovermode="closest"), key="mc_tornado", config={"displaylogo": False})


@st.fragment
def stress_section() -> None:
    kind = st.segmented_control("Rodzaj scenariusza", ["hyp", "hist"], default="hyp", required=True, key="ui_stress_kind",
                                format_func={"hyp": "Hipotetyczne", "hist": "Historyczne"}.get)
    names = list(HYPOTHETICAL_SCENARIOS) if kind == "hyp" else list(HISTORICAL_SCENARIOS)
    name = st.selectbox("Scenariusz", names, key="ui_stress_name")
    when = st.radio("Moment wystąpienia szoku", ["now", "fire"], horizontal=True, key="ui_stress_when",
                    format_func={"now": "Teraz (start symulacji)", "fire": "Tuż po przejściu na FIRE"}.get)
    if kind == "hyp":
        sc = HYPOTHETICAL_SCENARIOS[name]
        st.caption(sc["description"])
        path = hypothetical_stress_path(sc, ctx.assets, ctx.classes, ctx.currencies, ctx.weights, s.market.base_currency)
    else:
        if ctx.md.is_synthetic:
            st.warning("Scenariusze historyczne wymagają rzeczywistych notowań – dane demonstracyjne są syntetyczne. "
                       "Przełącz źródło na Yahoo Finance lub migawkę.", icon=":material/warning:")
            return
        a, b = HISTORICAL_SCENARIOS[name]
        path, msg = historical_stress_path(ctx.md.returns[ctx.assets], a, b, ctx.weights)
        if path is None:
            st.warning(f"Scenariusz niedostępny dla obecnego portfela: {msg}", icon=":material/warning:")
            return
    loss = float(np.prod(1 + path) - 1)
    c1, c2 = st.columns(2)
    c1.metric("Skumulowana stopa portfela w scenariuszu", fmt_pct(loss, 1), border=True)
    c2.metric("Zmiana wartości obecnego kapitału", fmt_money(loss * s.profile.initial_capital, compact=True), border=True)
    if st.button("Przelicz prawdopodobieństwa ze szokiem", icon=":material/bolt:", key="stress_run"):
        start = 0 if when == "now" else (s.profile.fire_age - s.profile.current_age) * 12
        base_n = res["nominal_subset"]
        stressed = apply_stress(base_n, path, start)
        variant = s
        if when == "fire":
            variant = s.model_copy(update={"fire": s.fire.model_copy(update={"retirement_rule": RetirementRule.FIXED_AGE})})
        plan = build_plan(variant, ctx.stats["geometric_real_net"])
        ws = s.withdrawal
        inc = analytics.income_yield_or_none(ctx)
        if ws.strategy == WithdrawalStrategy.INCOME_ONLY and inc is None:
            ws = ws.model_copy(update={"strategy": WithdrawalStrategy.CONSTANT_REAL})
        gn = res.get("gold_nominal_subset")
        gold = deflate_returns(gn[:, : plan.n_months], s.profile.inflation) if gn is not None else None
        before = summarize(run_lifecycle(deflate_returns(base_n[:, : plan.n_months], s.profile.inflation), plan, ws, inc, gold))
        after = summarize(run_lifecycle(deflate_returns(stressed[:, : plan.n_months], s.profile.inflation), plan, ws, inc, gold))
        rows = [("P(utrzymania kapitału)", before["p_not_depleted"], after["p_not_depleted"]),
                ("P(pokrycia wydatków)", before["p_spending_covered"], after["p_spending_covered"]),
                ("P(sukcesu planu)", before["p_success"], after["p_success"])]
        st.dataframe(pd.DataFrame([{"Miara": r, "Bez szoku": fmt_pct(b, 1), "Ze szokiem": fmt_pct(a, 1),
                                    "Zmiana": fmt_pct(a - b, 1, signed=True)} for r, b, a in rows]), hide_index=True, width="stretch")
        st.caption(f"Obliczono na {fmt_num(base_n.shape[0])} ścieżkach; szok nadpisuje stopy zwrotu wszystkich ścieżek w oknie scenariusza.")


with tabs[3]:
    heatmap_section()
with tabs[4]:
    sensitivity_section()
with tabs[5]:
    stress_section()

with st.expander("Założenia symulacji Monte Carlo", icon=":material/menu_book:"):
    st.markdown(
        "- Stopy zwrotu aktywów symulowane są **łącznie** (korelacje zachowane), portfel rebalansowany co miesiąc do wag docelowych.\n"
        "- Parametry estymowane na oknie estymacji; opcjonalna korekta w dół i rozkład t-Studenta.\n"
        "- Inflacja stała; stopy nominalne deflowane dokładnie raz. Wyniki realne w dzisiejszych złotych.\n"
        "- Każda ścieżka korzysta z tych samych zasad wpłat, wypłat, kosztów i podatków co ścieżka deterministyczna.\n"
        "- Historyczne stopy zwrotu nie są gwarancją przyszłych wyników; model nie uwzględnia zmian reżimów i ryzyka długowieczności "
        "poza przyjętym horyzontem."
    )
    st.latex(r"\mathbf{x}_t \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \;\;\text{lub}\;\; t_\nu, \qquad R_{p,t} = \mathbf{w}^\top\left(e^{\mathbf{x}_t}-1\right)")
