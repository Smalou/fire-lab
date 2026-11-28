"""Sidebar: active scenario, data source settings dialog, data refresh."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from components.formatting import fmt_datetime
from components.theme import ICONS
from services import catalog, state
from services.analytics import clear_market_cache
from services.market_data import DataUnavailableError, parse_price_csv, snapshot_available
from services.scenarios import remap_portfolio_to_source

SOURCE_LABELS = {
    "demo": "Demo – dane syntetyczne (offline)",
    "yfinance": "Yahoo Finance – pobieranie na żywo",
    "snapshot": "Yahoo Finance – zapisana migawka (offline)",
    "csv": "Import własnego pliku CSV",
}


def _on_scenario_change() -> None:
    sid = st.session_state["ui_active_scenario"]
    state.set_active(sid)
    st.query_params["scenariusz"] = state.active().name


@st.dialog("Ustawienia danych rynkowych", width="large", icon=ICONS["data"])
def data_settings_dialog() -> None:
    s = state.active()
    m = s.market
    st.caption("Zmiana tych ustawień oznacza wcześniejsze wyniki jako nieaktualne.")
    source = st.radio("Źródło danych", list(SOURCE_LABELS), index=list(SOURCE_LABELS).index(m.data_source),
                      format_func=SOURCE_LABELS.get, horizontal=False, key="ui_ds_source",
                      help="W trybie demo dane są syntetyczne i wyraźnie oznaczone. Yahoo Finance wymaga Internetu.")
    if source == "snapshot" and not snapshot_available():
        st.warning("Brak migawki. Uruchom `python scripts/fetch_snapshot.py` na komputerze z dostępem do Internetu.")
    if source == "csv":
        _csv_import()

    with st.form("data_settings_form", border=False):
        c1, c2 = st.columns(2)
        base = c1.selectbox("Waluta bazowa", ["PLN", "USD", "EUR"], index=["PLN", "USD", "EUR"].index(m.base_currency),
                            disabled=source in ("demo", "csv"),
                            help="Ceny instrumentów są przeliczane po historycznych kursach FX z tego samego dnia.")
        rf = c2.number_input("Stopa wolna od ryzyka (rocznie, %)", -5.0, 30.0, m.risk_free_rate * 100, 0.25,
                             help="Nominalna stopa w walucie bazowej, używana w Sharpe, Sortino, CML i jako oprocentowanie bufora gotówki.")
        longest = st.toggle("Najdłuższa wspólna historia aktywów", value=m.start_date is None,
                            help="Jeżeli wyłączysz, możesz wskazać własny okres – aplikacja sprawdzi dostępność danych.")
        c3, c4 = st.columns(2)
        start = c3.date_input("Początek okresu", value=m.start_date or date(2005, 1, 1), format="DD.MM.YYYY",
                              min_value=date(1990, 1, 1), max_value=date.today(), disabled=longest)
        end = c4.date_input("Koniec okresu (puste = dziś)", value=m.end_date, format="DD.MM.YYYY",
                            min_value=date(1990, 1, 1), max_value=date.today())
        c5, c6 = st.columns(2)
        est = c5.slider("Okres estymacji parametrów (lata)", 3, 30, m.estimation_years,
                        help="Ostatnie N lat okresu analizy służy do estymacji średnich i kowariancji (optymalizacja, Monte Carlo).")
        cov = c6.segmented_control("Estymator kowariancji", ["sample", "ledoit_wolf"], default=m.covariance_method,
                                   format_func={"sample": "Próbkowy", "ledoit_wolf": "Ledoit–Wolf"}.get, required=True,
                                   help="Ledoit–Wolf (shrinkage) stabilizuje macierz przy krótkiej historii lub wielu aktywach.")
        submitted = st.form_submit_button("Zastosuj ustawienia", type="primary", icon=ICONS["save"])
    if submitted:
        new = s.model_copy(deep=True)
        if source != m.data_source:
            new.portfolio = remap_portfolio_to_source(new, source)
            if source == "csv" and st.session_state.get("csv_dataset"):
                cols = list(st.session_state["csv_dataset"]["prices"].columns)
                new.portfolio = new.portfolio.model_copy(update={"assets": cols, "weights": [1 / len(cols)] * len(cols)})
            new.market.benchmark = new.portfolio.assets[0] if new.portfolio.assets else None
            if source == "demo":
                new.market.benchmark = "SYN_GLOBAL" if "SYN_GLOBAL" in new.portfolio.assets else new.market.benchmark
        new.market.data_source = source
        new.market.base_currency = "PLN" if source == "demo" else base
        new.market.risk_free_rate = rf / 100
        new.market.start_date = None if longest else start
        new.market.end_date = end
        new.market.estimation_years = est
        new.market.covariance_method = cov
        try:
            state.save(new, "Zapisano ustawienia danych.")
        except Exception as exc:  # noqa: BLE001 - validation message for the user
            st.error(f"Niepoprawne ustawienia: {exc}")
            return
        st.rerun()


def _csv_import() -> None:
    st.markdown("**Import CSV** – pierwsza kolumna: data, kolejne: ceny całkowite (z dywidendami) w walucie bazowej.")
    up = st.file_uploader("Plik CSV z cenami", type=["csv"], key="ui_csv_upload",
                          help="Separator , lub ; (wtedy przecinek dziesiętny). Minimum 37 miesięcy wspólnej historii.")
    if up is not None:
        try:
            prices = parse_price_csv(up.getvalue())
        except DataUnavailableError as exc:
            st.error(str(exc))
            return
        labels = catalog.asset_class_labels()
        meta = pd.DataFrame({"Kolumna": prices.columns, "Klasa aktywów": ["global_equity"] * prices.shape[1]})
        edited = st.data_editor(
            meta, hide_index=True, disabled=["Kolumna"], key="ui_csv_classes",
            column_config={"Klasa aktywów": st.column_config.SelectboxColumn("Klasa aktywów", options=list(labels),
                                                                             format_func=labels.get, required=True)},
        )
        if st.button("Wczytaj dane z pliku", icon=ICONS["upload"]):
            st.session_state["csv_dataset"] = {
                "name": up.name,
                "prices": prices,
                "classes": dict(zip(edited["Kolumna"], edited["Klasa aktywów"])),
            }
            clear_market_cache()
            st.toast(f"Wczytano {prices.shape[1]} serii, {len(prices)} miesięcy.", icon=":material/check_circle:")
    ds = st.session_state.get("csv_dataset")
    if ds:
        st.caption(f"Aktualnie wczytany plik: **{ds['name']}** ({ds['prices'].shape[1]} serii).")


def render_sidebar() -> None:
    sc = state.scenarios()
    s = state.active()
    with st.sidebar:
        st.markdown("### :material/local_fire_department: FIRE Lab")
        st.caption("Od akumulacji do niezależności finansowej.")
        ids = list(sc)
        st.selectbox("Aktywny scenariusz", ids, index=ids.index(s.id), format_func=lambda i: sc[i].name,
                     key="ui_active_scenario", on_change=_on_scenario_change,
                     help="Wszystkie strony korzystają z jednego, wspólnego scenariusza.")
        src = SOURCE_LABELS[s.market.data_source]
        st.caption(f":material/database: {src}")
        if s.market.data_source == "demo":
            st.badge("Dane syntetyczne", icon=":material/science:", color="orange")
        with st.container(horizontal=True, gap="small"):
            if st.button("Dane", icon=ICONS["settings"], help="Źródło danych, okres, waluta, stopa wolna od ryzyka"):
                data_settings_dialog()
            if st.button("Odśwież", icon=ICONS["refresh"], help="Wyczyść cache i pobierz dane ponownie"):
                clear_market_cache()
                st.session_state["data_refreshed_at"] = pd.Timestamp.now()
                st.toast("Wyczyszczono cache danych – zostaną pobrane ponownie.", icon=ICONS["refresh"])
        if st.session_state.get("data_refreshed_at") is not None:
            st.caption(f"Ostatnie ręczne odświeżenie: {fmt_datetime(st.session_state['data_refreshed_at'])}")
        st.divider()
        st.caption(
            ":material/info: Scenariusze są przechowywane wyłącznie w bieżącej sesji przeglądarki i mogą zostać utracone "
            "po jej zamknięciu. Eksportuj je do JSON na stronie Porównywarki scenariuszy."
        )
