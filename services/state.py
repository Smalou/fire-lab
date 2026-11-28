"""Session state: scenarios (committed configuration), results with fingerprints, UI state.

Layers
------
* draft parameters     – live inside ``st.form`` widgets (not stored here)
* committed scenario   – ``st.session_state["scenarios"][id]`` (Pydantic ``Scenario``)
* results              – ``st.session_state["results"][(scenario_id, kind)]`` with the fingerprint
                         of the scenario sections they depend on; a mismatch == stale result
* UI state             – widget keys prefixed ``ui_`` and ``st.query_params``
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from core.models import RESULT_DEPENDENCIES, Scenario
from services.scenarios import default_scenarios

SCENARIOS = "scenarios"
ACTIVE = "active_scenario_id"
RESULTS = "results"


def init() -> None:
    ss = st.session_state
    if SCENARIOS not in ss:
        ss[SCENARIOS] = {s.id: s for s in default_scenarios()}
    if RESULTS not in ss:
        ss[RESULTS] = {}
    if ACTIVE not in ss or ss[ACTIVE] not in ss[SCENARIOS]:
        ss[ACTIVE] = next(iter(ss[SCENARIOS]))
    ss.setdefault("csv_dataset", None)
    ss.setdefault("celebrated", set())


def scenarios() -> dict[str, Scenario]:
    return st.session_state[SCENARIOS]


def active() -> Scenario:
    return scenarios()[st.session_state[ACTIVE]]


def set_active(scenario_id: str) -> None:
    if scenario_id in scenarios():
        st.session_state[ACTIVE] = scenario_id


def save(s: Scenario, message: str | None = "Zapisano ustawienia scenariusza.") -> None:
    """Commit a validated scenario. Dependent results become stale automatically (fingerprints)."""
    s = Scenario.model_validate(s.model_dump())  # full validation incl. nested models
    s.touch()
    scenarios()[s.id] = s
    if message:
        st.toast(message, icon=":material/check_circle:")


def add(s: Scenario, make_active: bool = True) -> None:
    scenarios()[s.id] = s
    if make_active:
        set_active(s.id)


def delete(scenario_id: str) -> bool:
    sc = scenarios()
    if len(sc) <= 1 or scenario_id not in sc:
        return False
    del sc[scenario_id]
    for key in [k for k in st.session_state[RESULTS] if k[0] == scenario_id]:
        del st.session_state[RESULTS][key]
    if st.session_state[ACTIVE] == scenario_id:
        st.session_state[ACTIVE] = next(iter(sc))
    return True


def fingerprint(s: Scenario, kind: str) -> str:
    fp = s.fingerprint(*RESULT_DEPENDENCIES[kind])
    if s.market.data_source == "csv" and st.session_state.get("csv_dataset"):
        fp += "|" + st.session_state["csv_dataset"]["name"]
    return fp


def store_result(s: Scenario, kind: str, data: Any, sub: str = "") -> None:
    st.session_state[RESULTS][(s.id, kind + sub)] = {
        "fingerprint": fingerprint(s, kind),
        "data": data,
        "computed_at": datetime.now().replace(microsecond=0),
    }


def get_result(s: Scenario, kind: str, sub: str = "") -> tuple[Any | None, bool, datetime | None]:
    """Return (data, is_fresh, computed_at). Stale results are returned but flagged."""
    entry = st.session_state[RESULTS].get((s.id, kind + sub))
    if not entry:
        return None, False, None
    return entry["data"], entry["fingerprint"] == fingerprint(s, kind), entry["computed_at"]
