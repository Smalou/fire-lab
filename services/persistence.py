"""Bridge between the database and the domain models (no Streamlit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from core.models import SCHEMA_VERSION, Scenario
from services.db import Database

# Upgraders for stored scenario documents: {from_version: fn(data) -> data at from_version + 1}.
# Sections added to ``Scenario`` with defaults need no upgrader (missing keys take defaults);
# add one only when a field is renamed, moved or changes meaning, and bump core.models.SCHEMA_VERSION.
UPGRADERS: dict[int, Any] = {}


def upgrade(data: dict[str, Any]) -> dict[str, Any]:
    version = int(data.get("schema_version", 1))
    if version > SCHEMA_VERSION:
        raise ValueError(f"nowsza wersja schematu ({version}) niż obsługiwana ({SCHEMA_VERSION})")
    while version < SCHEMA_VERSION:
        fn = UPGRADERS.get(version)
        data = fn(dict(data)) if fn else dict(data)
        version += 1
        data["schema_version"] = version
    return data


@dataclass
class LoadResult:
    scenarios: list[Scenario] = field(default_factory=list)
    broken: list[tuple[str, str]] = field(default_factory=list)  # (name, reason) – left untouched in the DB


def load_scenarios(db: Database, user_id: int) -> LoadResult:
    out = LoadResult()
    for row in db.list_scenarios(user_id):
        try:
            out.scenarios.append(Scenario.model_validate(upgrade(row.data)))
        except (ValidationError, ValueError) as exc:
            msg = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            out.broken.append((row.name, msg))
    return out


def save_scenario(db: Database, user_id: int, s: Scenario, position: int | None = None) -> None:
    db.save_scenario(user_id, s.model_dump(mode="json"), position)
