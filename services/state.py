"""Session state: scenarios (committed configuration), results with fingerprints, UI state.

Layers
------
* draft parameters     – live inside ``st.form`` widgets (not stored here)
* committed scenario   – ``st.session_state["scenarios"][id]`` (Pydantic ``Scenario``)
* results              – ``st.session_state["results"][(scenario_id, kind)]`` with the fingerprint
                         of the scenario sections they depend on; a mismatch == stale result
* UI state             – widget keys prefixed ``ui_`` and ``st.query_params``
* persistence          – for a signed-in / local-profile user every committed change is written
                         through to SQLite (``services.db``); guests keep the old session-only behaviour
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Any, Callable

import streamlit as st

from core.models import RESULT_DEPENDENCIES, Scenario
from services import auth, persistence
from services.auth import CurrentUser
from services.scenarios import default_scenarios

SCENARIOS = "scenarios"
ACTIVE = "active_scenario_id"
RESULTS = "results"
OWNER = "scenarios_owner"  # id of the user whose scenarios are loaded (None = guest)
LOAD_WARNINGS = "load_warnings"

log = logging.getLogger(__name__)


def init() -> None:
    ss = st.session_state
    user = auth.current_user()
    if SCENARIOS not in ss or ss.get(OWNER, "unset") != user.id:
        _load_for(user)
    if RESULTS not in ss:
        ss[RESULTS] = {}
    if ACTIVE not in ss or ss[ACTIVE] not in ss[SCENARIOS]:
        ss[ACTIVE] = next(iter(ss[SCENARIOS]))
    ss.setdefault("csv_dataset", None)
    ss.setdefault("celebrated", set())


def user() -> CurrentUser:
    return auth.current_user()


def _persist(fn: Callable[[auth.Database, int], Any]) -> Any:
    """Run a write for the current persisted user; a database failure never breaks the session."""
    u = auth.current_user()
    if not u.persisted:
        return None
    try:
        return fn(auth.database(), u.id)
    except sqlite3.Error as exc:
        log.exception("database write failed")
        st.toast(f"Nie udało się zapisać w bazie danych: {exc}", icon=":material/error:")
        return None


def _load_for(u: CurrentUser) -> None:
    ss = st.session_state
    ss[RESULTS] = {}
    ss["csv_dataset"] = None
    ss[LOAD_WARNINGS] = []
    ss.pop(ACTIVE, None)
    ss[OWNER] = u.id
    if not u.persisted:
        ss[SCENARIOS] = {s.id: s for s in default_scenarios()}
        return
    db = auth.database()
    res = persistence.load_scenarios(db, u.id)
    ss[LOAD_WARNINGS] = [f"Nie wczytano scenariusza „{n}” ({why})." for n, why in res.broken]
    items = res.scenarios
    if not items:
        items = default_scenarios()
        for pos, s in enumerate(items):
            persistence.save_scenario(db, u.id, s, pos)
    ss[SCENARIOS] = {s.id: s for s in items}
    prefs = db.settings(u.id)
    if prefs.get("active_scenario_id") in ss[SCENARIOS]:
        ss[ACTIVE] = prefs["active_scenario_id"]
    if prefs.get("active_dataset_id"):
        ds = db.get_dataset(u.id, int(prefs["active_dataset_id"]))
        if ds is not None:
            try:
                activate_dataset(ds, remember=False)
            except Exception as exc:  # noqa: BLE001 - corrupted file must not block the app
                ss[LOAD_WARNINGS].append(f"Nie wczytano zapisanego pliku danych „{ds.name}”: {exc}")


def reload() -> None:
    """Drop the in-memory copy and load the current user's data again (e.g. after switching profile)."""
    st.session_state.pop(SCENARIOS, None)
    init()


def activate_dataset(ds, remember: bool = True) -> None:
    """Put a stored price file into the session (``csv_dataset``)."""
    from services.market_data import parse_price_csv

    prices = parse_price_csv(ds.content)
    classes = {c: ds.meta.get("classes", {}).get(c, "global_equity") for c in prices.columns}
    st.session_state["csv_dataset"] = {"name": ds.name, "prices": prices, "classes": classes, "db_id": ds.id}
    if remember:
        _persist(lambda db, uid: db.set_setting(uid, "active_dataset_id", ds.id))


def store_dataset(name: str, content: bytes, classes: dict[str, str]) -> int | None:
    """Save an imported price file in the user's profile and remember it as the active one."""
    def _write(db, uid: int) -> int:
        ds_id = db.save_dataset(uid, name, content, {"classes": classes})
        db.set_setting(uid, "active_dataset_id", ds_id)
        return ds_id

    return _persist(_write)


def scenarios() -> dict[str, Scenario]:
    return st.session_state[SCENARIOS]


def active() -> Scenario:
    return scenarios()[st.session_state[ACTIVE]]


def set_active(scenario_id: str) -> None:
    if scenario_id in scenarios() and st.session_state.get(ACTIVE) != scenario_id:
        st.session_state[ACTIVE] = scenario_id
        _persist(lambda db, uid: db.set_setting(uid, "active_scenario_id", scenario_id))


def save(s: Scenario, message: str | None = "Zapisano ustawienia scenariusza.") -> None:
    """Commit a validated scenario. Dependent results become stale automatically (fingerprints)."""
    s = Scenario.model_validate(s.model_dump())  # full validation incl. nested models
    s.touch()
    scenarios()[s.id] = s
    _persist(lambda db, uid: persistence.save_scenario(db, uid, s))
    if message:
        st.toast(message, icon=":material/check_circle:")


def add(s: Scenario, make_active: bool = True) -> None:
    scenarios()[s.id] = s
    _persist(lambda db, uid: persistence.save_scenario(db, uid, s))
    if make_active:
        set_active(s.id)


def delete(scenario_id: str) -> bool:
    sc = scenarios()
    if len(sc) <= 1 or scenario_id not in sc:
        return False
    del sc[scenario_id]
    _persist(lambda db, uid: db.delete_scenario(uid, scenario_id))
    for key in [k for k in st.session_state[RESULTS] if k[0] == scenario_id]:
        del st.session_state[RESULTS][key]
    if st.session_state[ACTIVE] == scenario_id:
        set_active(next(iter(sc)))
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
