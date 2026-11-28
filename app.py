"""FIRE Lab – entrypoint and router.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from components.theme import ICONS
from services import state

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
    "Konto": [
        st.Page("views/profile.py", title="Profil i dane", icon=ICONS["profile"], url_path="profil"),
    ],
}
nav = st.navigation(pages, position="sidebar")

from components.sidebar import render_sidebar  # noqa: E402  (after set_page_config)

render_sidebar()
nav.run()
