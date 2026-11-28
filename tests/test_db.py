import sqlite3

import pytest

from core.models import SCHEMA_VERSION
from services import persistence
from services.db import Database, slugify
from services.scenarios import clone, default_scenarios


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "t.db")


def test_migrations_are_idempotent(tmp_path):
    a = Database(tmp_path / "x.db")
    b = Database(tmp_path / "x.db")
    with b.connect() as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == a.migrate()
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "user_settings", "scenarios", "datasets"} <= tables


def test_refuses_newer_database(tmp_path):
    db = Database(tmp_path / "x.db")
    with db.connect() as con:
        con.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="nowszy schemat"):
        Database(tmp_path / "x.db")


def test_user_upsert_is_keyed_by_provider_and_subject(db):
    u1 = db.upsert_user("google", "sub-1", "Ala", "ala@example.com")
    u2 = db.upsert_user("google", "sub-1", "Ala Nowak", "ala@example.com")
    u3 = db.upsert_user("local", "sub-1", "Lokalny")
    assert u1.id == u2.id != u3.id
    assert db.get_user(u1.id).display_name == "Ala Nowak"
    with pytest.raises(sqlite3.IntegrityError):
        db.upsert_user("facebook", "x", "X")


def test_settings_roundtrip_json(db):
    u = db.upsert_user("local", "a", "A")
    db.set_setting(u.id, "active_scenario_id", "abc")
    db.set_setting(u.id, "active_scenario_id", "def")
    db.set_setting(u.id, "ui", {"theme": "dark", "n": [1, 2]})
    assert db.get_setting(u.id, "active_scenario_id") == "def"
    assert db.settings(u.id)["ui"] == {"theme": "dark", "n": [1, 2]}
    assert db.get_setting(u.id, "missing", 42) == 42


def test_scenarios_roundtrip_order_and_isolation(db):
    a = db.upsert_user("local", "a", "A")
    b = db.upsert_user("local", "b", "B")
    sc = default_scenarios()
    for s in sc:
        persistence.save_scenario(db, a.id, s)
    persistence.save_scenario(db, b.id, sc[0])  # same scenario id in another profile is independent
    loaded = persistence.load_scenarios(db, a.id)
    assert [s.model_dump() for s in loaded.scenarios] == [s.model_dump() for s in sc]
    assert not loaded.broken
    edited = sc[0].model_copy(deep=True)
    edited.profile.monthly_contribution = 9_999
    persistence.save_scenario(db, a.id, edited)  # update keeps position
    again = persistence.load_scenarios(db, a.id).scenarios
    assert [s.id for s in again] == [s.id for s in sc]
    assert again[0].profile.monthly_contribution == 9_999
    assert persistence.load_scenarios(db, b.id).scenarios[0].profile.monthly_contribution == sc[0].profile.monthly_contribution
    assert db.delete_scenario(a.id, sc[1].id) and not db.delete_scenario(a.id, sc[1].id)
    assert len(db.list_scenarios(a.id)) == 2


def test_broken_or_newer_scenarios_are_reported_not_fatal(db):
    u = db.upsert_user("local", "a", "A")
    good = default_scenarios()[0]
    persistence.save_scenario(db, u.id, good)
    bad = clone(good, "Zepsuty").model_dump(mode="json")
    bad["profile"]["fire_age"] = 5
    db.save_scenario(u.id, bad)
    newer = clone(good, "Z przyszłości").model_dump(mode="json")
    newer["schema_version"] = SCHEMA_VERSION + 1
    db.save_scenario(u.id, newer)
    res = persistence.load_scenarios(db, u.id)
    assert [s.name for s in res.scenarios] == [good.name]
    assert {n for n, _ in res.broken} == {"Zepsuty", "Z przyszłości"}


def test_old_documents_without_new_sections_load_with_defaults(db):
    u = db.upsert_user("local", "a", "A")
    data = default_scenarios()[0].model_dump(mode="json")
    del data["withdrawal"]  # a section added later to the model
    db.save_scenario(u.id, data)
    s = persistence.load_scenarios(db, u.id).scenarios[0]
    assert s.withdrawal.initial_rate > 0


def test_datasets_dedupe_and_cascade_delete(db):
    u = db.upsert_user("local", "a", "A")
    i1 = db.save_dataset(u.id, "ceny.csv", b"date,A\n2020-01-31,1\n", {"classes": {"A": "bonds"}})
    i2 = db.save_dataset(u.id, "ceny_v2.csv", b"date,A\n2020-01-31,1\n", {"classes": {"A": "gold"}})
    assert i1 == i2
    ds = db.get_dataset(u.id, i1)
    assert ds.name == "ceny_v2.csv" and ds.meta["classes"]["A"] == "gold" and ds.content.startswith(b"date")
    persistence.save_scenario(db, u.id, default_scenarios()[0])
    db.set_setting(u.id, "k", 1)
    export = db.export_user(u.id)
    assert len(export["scenarios"]) == 1 and export["datasets"][0]["size_bytes"] > 0
    db.delete_user(u.id)
    assert db.list_datasets(u.id) == [] and db.list_scenarios(u.id) == [] and db.settings(u.id) == {}


def test_slugify_polish():
    assert slugify("Łukasz – Plan Główny!") == "lukasz-plan-glowny"
    assert slugify("   ") == "profil"
