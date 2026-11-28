"""Scenarios survive a new browser session (local-profile mode) and profiles are isolated."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRE_LAB_DB", str(tmp_path / "app.db"))
    return tmp_path / "app.db"


def _new_session(page: str = "views/scenarios.py") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.switch_page(page).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _names(at: AppTest) -> list[str]:
    return [s.name for s in at.session_state["scenarios"].values()]


def test_scenarios_persist_across_sessions(fresh_db):
    at = _new_session()
    assert _names(at) == ["Zrównoważony", "Agresywny", "Konserwatywny"]
    next(b for b in at.button if b.label == "Klonuj").click()
    at.run()
    assert len(_names(at)) == 4
    at.selectbox(key="ui_sc_pick").set_value(list(at.session_state["scenarios"])[2]).run()
    next(b for b in at.button if b.label == "Ustaw jako aktywny").click()
    at.run()
    active = at.session_state["active_scenario_id"]

    again = _new_session()
    assert _names(again) == _names(at)
    assert again.session_state["active_scenario_id"] == active
    assert fresh_db.exists()


def test_local_profiles_are_isolated(fresh_db):
    from services.auth import _database

    db = _database(str(fresh_db))
    other = db.upsert_user("local", "druga-osoba", "Druga osoba")
    at = _new_session()
    default_id = at.session_state["local_profile_id"]
    assert default_id != other.id
    at.selectbox(key="ui_local_profile").set_value(other.id).run()
    assert not at.exception
    assert at.session_state["local_profile_id"] == other.id
    sid = next(iter(at.session_state["scenarios"]))
    at.session_state["ui_sc_pick"] = sid
    at.run()
    next(b for b in at.button if b.label == "Klonuj").click()
    at.run()
    assert len(db.list_scenarios(other.id)) == 4
    assert len(db.list_scenarios(default_id)) == 3


def test_google_mode_guest_sees_login_and_nothing_is_saved(fresh_db):
    at = AppTest.from_file(APP, default_timeout=300)
    at.secrets["auth"] = {"redirect_uri": "http://localhost:8501/oauth2callback", "cookie_secret": "x" * 32,
                          "google": {"client_id": "id", "client_secret": "secret",
                                     "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration"}}
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any(b.label == "Zaloguj przez Google" for b in at.sidebar.button)
    from services.auth import _database

    assert _database(str(fresh_db)).list_users() == []


class _FakeUser(dict):
    def __getattr__(self, key):
        return self[key]


def _google_app(monkeypatch, **claims) -> AppTest:
    import streamlit

    monkeypatch.setattr(streamlit, "user", _FakeUser(is_logged_in=True, **claims))
    at = AppTest.from_file(APP, default_timeout=300)
    at.secrets["auth"] = {"redirect_uri": "http://localhost:8501/oauth2callback", "cookie_secret": "x" * 32,
                          "google": {"client_id": "id", "client_secret": "secret",
                                     "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration"}}
    return at


def test_google_user_gets_persistent_profile(fresh_db, monkeypatch):
    from services.auth import _database

    at = _google_app(monkeypatch, sub="g-123", email="ala@example.com", name="Ala", email_verified=True).run()
    assert not at.exception, [e.message for e in at.exception]
    assert any(b.label == "Wyloguj" for b in at.sidebar.button)
    users = _database(str(fresh_db)).list_users("google")
    assert [(u.subject, u.email) for u in users] == [("g-123", "ala@example.com")]
    assert len(_database(str(fresh_db)).list_scenarios(users[0].id)) == 3


def test_google_allow_list_blocks_other_accounts(fresh_db, monkeypatch):
    from services.auth import _database

    at = _google_app(monkeypatch, sub="g-9", email="obcy@example.com", name="Obcy", email_verified=True)
    at.secrets["fire_lab"] = {"allowed_emails": ["ala@example.com"]}
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("nie ma dostępu" in w.value for w in at.sidebar.warning)
    assert _database(str(fresh_db)).list_users() == []
