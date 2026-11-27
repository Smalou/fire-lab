"""Configurable instrument catalogue (data/instruments.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "instruments.json"


@dataclass(frozen=True)
class Instrument:
    id: str
    name: str
    asset_class: str
    currency: str
    listed_since: str
    substitute: str | None
    source: str


@lru_cache(maxsize=1)
def _raw() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def asset_class_labels() -> dict[str, str]:
    return dict(_raw()["asset_classes"])


def instruments(source: str | None = None) -> list[Instrument]:
    items = [Instrument(**i) for i in _raw()["instruments"]]
    if source is None:
        return items
    src = "demo" if source == "demo" else "yfinance"
    return [i for i in items if i.source == src]


def get(asset_id: str) -> Instrument | None:
    return next((i for i in instruments() if i.id == asset_id), None)


def default_portfolio(source: str) -> dict[str, float]:
    key = "demo" if source == "demo" else "yfinance"
    return dict(_raw()["default_portfolios"][key])


def fx_ticker(currency: str, base: str) -> str:
    return f"{currency}{base}=X"


def catalog_frame(source: str | None = None) -> pd.DataFrame:
    labels = asset_class_labels()
    rows = [
        {
            "Ticker": i.id,
            "Nazwa": i.name,
            "Klasa aktywów": labels.get(i.asset_class, i.asset_class),
            "Waluta": i.currency,
            "Notowany od (ok.)": i.listed_since,
            "Zamiennik": i.substitute or "—",
        }
        for i in instruments(source)
    ]
    return pd.DataFrame(rows)


def classes_for(asset_ids: list[str], custom: dict[str, str] | None = None) -> dict[str, str]:
    out = {}
    for a in asset_ids:
        if custom and a in custom:
            out[a] = custom[a]
        else:
            inst = get(a)
            out[a] = inst.asset_class if inst else "global_equity"
    return out


def currencies_for(asset_ids: list[str], base: str, custom: dict[str, str] | None = None) -> dict[str, str]:
    out = {}
    for a in asset_ids:
        inst = get(a)
        out[a] = inst.currency if inst else (custom or {}).get(a, base)
    return out


def display_name(asset_id: str, custom_names: dict[str, str] | None = None) -> str:
    if custom_names and asset_id in custom_names:
        return custom_names[asset_id]
    inst = get(asset_id)
    return f"{inst.id} – {inst.name}" if inst else asset_id


def short_name(asset_id: str) -> str:
    inst = get(asset_id)
    if inst and inst.source == "demo":
        return inst.name.replace(" (demo)", "").replace("Syntetyczne ", "").replace("Syntetyczna ", "").capitalize()
    return asset_id
