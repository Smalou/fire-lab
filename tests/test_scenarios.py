import json

import pytest

from core.models import Scenario
from services.scenarios import ImportErrorPL, clone, default_scenarios, from_json, remap_portfolio_to_source, to_json


def test_defaults_are_valid_and_distinct():
    sc = default_scenarios()
    assert [s.name for s in sc] == ["Zrównoważony", "Agresywny", "Konserwatywny", "Rodzina z kontami"]
    for s in sc:
        assert abs(sum(s.portfolio.weights) - 1) < 1e-9
    assert len({s.id for s in sc}) == 4
    assert sc[3].household.enabled and not sc[0].household.enabled


def test_json_roundtrip_preserves_everything():
    sc = default_scenarios()
    back = from_json(to_json(sc))
    assert [b.model_dump() for b in back] == [s.model_dump() for s in sc]


def test_import_single_scenario_and_id_conflict():
    s = default_scenarios()[0]
    raw = json.dumps(s.model_dump(mode="json"))
    out = from_json(raw, existing_ids={s.id})
    assert out[0].id != s.id and out[0].name == s.name


def test_import_errors_are_polish():
    with pytest.raises(ImportErrorPL, match="JSON"):
        from_json("{not json")
    bad = default_scenarios()[0].model_dump(mode="json")
    bad["profile"]["fire_age"] = 10
    with pytest.raises(ImportErrorPL, match="Scenariusz 1"):
        from_json(json.dumps({"scenarios": [bad]}))
    bad = default_scenarios()[0].model_dump(mode="json")
    bad["portfolio"]["weights"] = [0.5] * len(bad["portfolio"]["weights"])
    with pytest.raises(ImportErrorPL):
        from_json(json.dumps(bad))


def test_clone_and_fingerprint():
    s = default_scenarios()[0]
    c = clone(s)
    assert c.id != s.id and c.name.endswith("(kopia)")
    assert c.fingerprint("portfolio", "market") == s.fingerprint("portfolio", "market")
    c.profile.monthly_contribution += 1
    assert c.fingerprint("profile") != s.fingerprint("profile")
    assert c.fingerprint("portfolio") == s.fingerprint("portfolio")  # presentation-independent


def test_weights_must_sum_to_one():
    s = default_scenarios()[0].model_dump()
    s["portfolio"]["weights"][0] += 0.1
    with pytest.raises(ValueError):
        Scenario.model_validate(s)


def test_remap_to_yfinance_keeps_classes():
    s = default_scenarios()[0]
    p = remap_portfolio_to_source(s, "yfinance")
    assert abs(sum(p.weights) - 1) < 1e-9
    assert all(not a.startswith("SYN_") for a in p.assets)
