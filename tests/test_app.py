"""Streamlit AppTest smoke and interaction tests (within what the framework supports)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["views/command_center.py", "views/planner.py", "views/portfolio_builder.py", "views/strategy_lab.py",
         "views/risk_lab.py", "views/post_fire.py", "views/scenarios.py", "views/methodology.py", "views/profile.py"]


def _click(at: AppTest, label: str) -> AppTest:
    for b in at.button:
        if b.label == label:
            b.click()
            return at.run()
    raise AssertionError(f"Brak przycisku „{label}”: {[b.label for b in at.button]}")


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=300).run()


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page):
    app.switch_page(page).run()
    _ok(app)
    assert app.title, "strona bez tytułu"


def test_polish_navigation_titles(app):
    app.switch_page("views/command_center.py").run()
    assert app.title[0].value.startswith("Centrum dowodzenia")
    labels = [m.label for m in app.metric]
    assert "Wymagany kapitał FIRE" in labels and "Zmienność portfela" in labels


def test_probabilities_hidden_until_computed(app):
    app.switch_page("views/command_center.py").run()
    probs = {m.label: m.value for m in app.metric if m.label.startswith("P(")}
    assert all(v == "—" for v in probs.values())


def test_full_workflow(app):
    app.switch_page("views/planner.py").run()
    _ok(_click(app, "Zatwierdź parametry planu"))
    app.switch_page("views/strategy_lab.py").run()
    _ok(_click(app, "Uruchom backtest"))
    assert app.dataframe, "brak tabeli wyników backtestu"
    app.switch_page("views/risk_lab.py").run()
    _ok(_click(app, "Uruchom symulację Monte Carlo"))
    vals = {m.label: m.value for m in app.metric}
    assert vals["P(utrzymania kapitału)"].endswith("%")
    app.switch_page("views/post_fire.py").run()
    _ok(_click(app, "Zapisz i porównaj strategie"))
    app.switch_page("views/command_center.py").run()
    _ok(app)
    probs = {m.label: m.value for m in app.metric if m.label.startswith("P(")}
    assert all(v != "—" for v in probs.values())


def test_portfolio_change_marks_results_stale(app):
    app.switch_page("views/portfolio_builder.py").run()
    seg = next(x for x in app.button_group if x.key == "ui_pf_method")
    seg.set_value("min_variance")
    _ok(_click(app, "Wyznacz i zatwierdź portfel"))
    app.switch_page("views/command_center.py").run()
    probs = {m.label: m.value for m in app.metric if m.label.startswith("P(")}
    assert all(v == "—" for v in probs.values()), "wyniki MC powinny być nieaktualne po zmianie portfela"
    app.switch_page("views/risk_lab.py").run()
    assert any("nieaktualne" in w.value for w in app.warning)


def test_presentation_toggle_does_not_invalidate(app):
    app.switch_page("views/risk_lab.py").run()
    _ok(_click(app, "Uruchom symulację Monte Carlo"))
    app.switch_page("views/planner.py").run()
    app.toggle(key="ui_planner_nominal").set_value(True).run()
    _ok(app)
    app.switch_page("views/risk_lab.py").run()
    assert not any("nieaktualne" in w.value for w in app.warning)


def test_scenario_clone_and_export(app):
    app.switch_page("views/scenarios.py").run()
    n = len(app.session_state["scenarios"])
    _ok(_click(app, "Klonuj"))
    assert len(app.session_state["scenarios"]) == n + 1
