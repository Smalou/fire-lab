"""SQLite persistence: user profiles, per-user settings, scenarios and imported datasets (no Streamlit).

Design
------
* One file database (default ``data/fire_lab.db``, override with ``FIRE_LAB_DB``).
* A short-lived connection per operation: safe with Streamlit's thread-per-session model.
* Scenarios are stored as a JSON document of the Pydantic ``Scenario`` model plus a few
  indexed columns (name, schema_version, timestamps). Adding fields to the model therefore
  needs no database migration; ``schema_version`` lets old documents be upgraded on load.
* Schema migrations of the database itself are tracked with ``PRAGMA user_version``.

Tables
------
users          – one row per identity (``provider`` = google | local, ``subject`` = OIDC ``sub`` / local slug)
user_settings  – key/value JSON settings of a user (active scenario, active dataset, ...)
scenarios      – the user's scenarios (JSON document), ordered by ``position``
datasets       – user-imported price files (raw CSV bytes + asset-class mapping), deduplicated by SHA-256
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "fire_lab.db"

MIGRATIONS: list[str] = [
    # 1 – initial schema
    """
    CREATE TABLE users (
        id            INTEGER PRIMARY KEY,
        provider      TEXT NOT NULL CHECK (provider IN ('google', 'local')),
        subject       TEXT NOT NULL,
        email         TEXT,
        display_name  TEXT NOT NULL,
        picture_url   TEXT,
        created_at    TEXT NOT NULL,
        last_login_at TEXT NOT NULL,
        UNIQUE (provider, subject)
    );
    CREATE TABLE user_settings (
        user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        key        TEXT NOT NULL,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (user_id, key)
    );
    CREATE TABLE scenarios (
        user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        id             TEXT NOT NULL,
        name           TEXT NOT NULL,
        description    TEXT NOT NULL DEFAULT '',
        schema_version INTEGER NOT NULL,
        data_json      TEXT NOT NULL,
        position       INTEGER NOT NULL DEFAULT 0,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL,
        PRIMARY KEY (user_id, id)
    );
    CREATE INDEX scenarios_by_user ON scenarios(user_id, position);
    CREATE TABLE datasets (
        id         INTEGER PRIMARY KEY,
        user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        name       TEXT NOT NULL,
        kind       TEXT NOT NULL CHECK (kind IN ('price_csv')),
        content    BLOB NOT NULL,
        meta_json  TEXT NOT NULL DEFAULT '{}',
        sha256     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (user_id, sha256)
    );
    """,
]
SCHEMA_VERSION = len(MIGRATIONS)


def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def slugify(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text.replace("ł", "l").replace("Ł", "L"))
    ascii_ = norm.encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_).strip("-") or "profil"


@dataclass(frozen=True)
class User:
    id: int
    provider: str
    subject: str
    email: str | None
    display_name: str
    picture_url: str | None
    created_at: str
    last_login_at: str


@dataclass(frozen=True)
class StoredScenario:
    id: str
    name: str
    schema_version: int
    data: dict[str, Any]
    updated_at: str


@dataclass(frozen=True)
class Dataset:
    id: int
    name: str
    kind: str
    content: bytes
    meta: dict[str, Any]
    created_at: str


class Database:
    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path or os.environ.get("FIRE_LAB_DB") or DEFAULT_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            with con:  # commit on success, rollback on error
                yield con
        finally:
            con.close()

    def migrate(self) -> int:
        with self.connect() as con:
            con.execute("PRAGMA journal_mode = WAL")
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"Baza {self.path} ma nowszy schemat ({version}) niż obsługiwany ({SCHEMA_VERSION}).")
            for i in range(version, SCHEMA_VERSION):
                con.executescript(MIGRATIONS[i])
                con.execute(f"PRAGMA user_version = {i + 1}")
        return SCHEMA_VERSION

    # ------------------------------------------------------------------ users
    def upsert_user(self, provider: str, subject: str, display_name: str,
                    email: str | None = None, picture_url: str | None = None) -> User:
        now = _now()
        with self.connect() as con:
            con.execute(
                """INSERT INTO users (provider, subject, email, display_name, picture_url, created_at, last_login_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (provider, subject) DO UPDATE SET
                       email = excluded.email, display_name = excluded.display_name,
                       picture_url = excluded.picture_url, last_login_at = excluded.last_login_at""",
                (provider, subject, email, display_name, picture_url, now, now),
            )
            row = con.execute("SELECT * FROM users WHERE provider = ? AND subject = ?", (provider, subject)).fetchone()
        return User(**dict(row))

    def get_user(self, user_id: int) -> User | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return User(**dict(row)) if row else None

    def list_users(self, provider: str | None = None) -> list[User]:
        sql, args = "SELECT * FROM users", ()
        if provider:
            sql, args = sql + " WHERE provider = ?", (provider,)
        with self.connect() as con:
            rows = con.execute(sql + " ORDER BY display_name COLLATE NOCASE", args).fetchall()
        return [User(**dict(r)) for r in rows]

    def rename_user(self, user_id: int, display_name: str) -> None:
        with self.connect() as con:
            con.execute("UPDATE users SET display_name = ? WHERE id = ?", (display_name.strip(), user_id))

    def delete_user(self, user_id: int) -> None:
        """Delete the profile and everything that belongs to it (cascade)."""
        with self.connect() as con:
            con.execute("DELETE FROM users WHERE id = ?", (user_id,))

    # ------------------------------------------------------------------ settings
    def get_setting(self, user_id: int, key: str, default: Any = None) -> Any:
        with self.connect() as con:
            row = con.execute("SELECT value_json FROM user_settings WHERE user_id = ? AND key = ?", (user_id, key)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, user_id: int, key: str, value: Any) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO user_settings (user_id, key, value_json, updated_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT (user_id, key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at""",
                (user_id, key, json.dumps(value, ensure_ascii=False), _now()),
            )

    def settings(self, user_id: int) -> dict[str, Any]:
        with self.connect() as con:
            rows = con.execute("SELECT key, value_json FROM user_settings WHERE user_id = ?", (user_id,)).fetchall()
        return {r["key"]: json.loads(r["value_json"]) for r in rows}

    # ------------------------------------------------------------------ scenarios
    def save_scenario(self, user_id: int, data: dict[str, Any], position: int | None = None) -> None:
        """Insert or replace one scenario. ``data`` is ``Scenario.model_dump(mode="json")``."""
        with self.connect() as con:
            if position is None:
                row = con.execute("SELECT position FROM scenarios WHERE user_id = ? AND id = ?", (user_id, data["id"])).fetchone()
                position = row[0] if row else con.execute(
                    "SELECT COALESCE(MAX(position) + 1, 0) FROM scenarios WHERE user_id = ?", (user_id,)).fetchone()[0]
            con.execute(
                """INSERT INTO scenarios (user_id, id, name, description, schema_version, data_json, position, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (user_id, id) DO UPDATE SET
                       name = excluded.name, description = excluded.description,
                       schema_version = excluded.schema_version, data_json = excluded.data_json,
                       position = excluded.position, updated_at = excluded.updated_at""",
                (user_id, data["id"], data["name"], data.get("description", ""), int(data.get("schema_version", 1)),
                 json.dumps(data, ensure_ascii=False), position,
                 str(data.get("created_at") or _now()), str(data.get("updated_at") or _now())),
            )

    def list_scenarios(self, user_id: int) -> list[StoredScenario]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT id, name, schema_version, data_json, updated_at FROM scenarios WHERE user_id = ? ORDER BY position, created_at",
                (user_id,),
            ).fetchall()
        return [StoredScenario(r["id"], r["name"], r["schema_version"], json.loads(r["data_json"]), r["updated_at"]) for r in rows]

    def delete_scenario(self, user_id: int, scenario_id: str) -> bool:
        with self.connect() as con:
            cur = con.execute("DELETE FROM scenarios WHERE user_id = ? AND id = ?", (user_id, scenario_id))
        return cur.rowcount > 0

    # ------------------------------------------------------------------ datasets
    def save_dataset(self, user_id: int, name: str, content: bytes, meta: dict[str, Any] | None = None,
                     kind: str = "price_csv") -> int:
        """Store an imported file; re-importing identical bytes updates name/meta and returns the same id."""
        digest = hashlib.sha256(content).hexdigest()
        with self.connect() as con:
            con.execute(
                """INSERT INTO datasets (user_id, name, kind, content, meta_json, sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (user_id, sha256) DO UPDATE SET name = excluded.name, meta_json = excluded.meta_json""",
                (user_id, name, kind, content, json.dumps(meta or {}, ensure_ascii=False), digest, _now()),
            )
            return con.execute("SELECT id FROM datasets WHERE user_id = ? AND sha256 = ?", (user_id, digest)).fetchone()[0]

    def list_datasets(self, user_id: int) -> list[Dataset]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM datasets WHERE user_id = ? ORDER BY created_at DESC, id DESC", (user_id,)).fetchall()
        return [self._dataset(r) for r in rows]

    def get_dataset(self, user_id: int, dataset_id: int) -> Dataset | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM datasets WHERE user_id = ? AND id = ?", (user_id, dataset_id)).fetchone()
        return self._dataset(row) if row else None

    def delete_dataset(self, user_id: int, dataset_id: int) -> bool:
        with self.connect() as con:
            cur = con.execute("DELETE FROM datasets WHERE user_id = ? AND id = ?", (user_id, dataset_id))
        return cur.rowcount > 0

    @staticmethod
    def _dataset(r: sqlite3.Row) -> Dataset:
        return Dataset(r["id"], r["name"], r["kind"], bytes(r["content"]), json.loads(r["meta_json"]), r["created_at"])

    # ------------------------------------------------------------------ export
    def export_user(self, user_id: int) -> dict[str, Any]:
        """Everything stored for a user (GDPR-style export); dataset contents are omitted, only metadata."""
        user = self.get_user(user_id)
        return {
            "user": {k: v for k, v in (user.__dict__ if user else {}).items() if k != "id"},
            "settings": self.settings(user_id),
            "scenarios": [s.data for s in self.list_scenarios(user_id)],
            "datasets": [{"name": d.name, "kind": d.kind, "meta": d.meta, "created_at": d.created_at,
                          "size_bytes": len(d.content)} for d in self.list_datasets(user_id)],
        }
