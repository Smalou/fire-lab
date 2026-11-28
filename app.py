"""FIRE Lab – entrypoint and router.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys

import streamlit as st

MIN_STREAMLIT = (1, 67)
_st_version = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
if _st_version < MIN_STREAMLIT:
    st.error(
        f"FIRE Lab wymaga Streamlit ≥ {'.'.join(map(str, MIN_STREAMLIT))}, a uruchomiono wersję {st.__version__} "
        f"(Python: `{sys.executable}`). Aktywuj środowisko projektu, zainstaluj zależności "
        "(`pip install -r requirements.txt`) i uruchom aplikację poleceniem `python -m streamlit run app.py`."
    )
    st.stop()

from components.theme import ICONS  # noqa: E402
from services import state  # noqa: E402

st.set_page_config(
    page_title="FIRE Lab – od akumulacji do niezależności finansowej",
    page_icon=":material/local_fire_department:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "FIRE Lab – akademicka aplikacja do analizy strategii FIRE (Zarządzanie portfelem)."},
)

state.init()

# Non-sensitive query parameter: active scenario by name (shareable within a session)
qp_name = st.query_params.get("scenariusz")
if qp_name:
    match = next((s for s in state.scenarios().values() if s.name == qp_name), None)
    if match and match.id != state.active().id:
        state.set_active(match.id)

pages = {
    "Przegląd": [
        st.Page("views/command_center.py", title="Centrum dowodzenia", icon=ICONS["home"], default=True),
    ],
    "Planowanie": [
        st.Page("views/planner.py", title="Planer FIRE", icon=ICONS["planner"], url_path="planer"),
        st.Page("views/portfolio_builder.py", title="Konstruktor portfela", icon=ICONS["portfolio"], url_path="portfel"),
    ],
    "Analizy": [
        st.Page("views/strategy_lab.py", title="Laboratorium strategii", icon=ICONS["strategy"], url_path="strategie"),
        st.Page("views/risk_lab.py", title="Laboratorium ryzyka", icon=ICONS["risk"], url_path="ryzyko"),
        st.Page("views/post_fire.py", title="Życie po FIRE", icon=ICONS["post_fire"], url_path="po-fire"),
        st.Page("views/scenarios.py", title="Porównywarka scenariuszy", icon=ICONS["scenarios"], url_path="scenariusze"),
    ],
    "Wiedza": [
        st.Page("views/methodology.py", title="Metodologia", icon=ICONS["methodology"], url_path="metodologia"),
    ],
}
nav = st.navigation(pages, position="sidebar")

from components.sidebar import render_sidebar  # noqa: E402  (after set_page_config)

render_sidebar()
nav.run()
