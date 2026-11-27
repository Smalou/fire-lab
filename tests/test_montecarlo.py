import numpy as np
import pytest

from core.lifecycle import deflate_returns, run_lifecycle, summarize
from core.fire import build_plan
from core.models import InvestorProfile, Scenario, SimulationModel
from core.risk import apply_stress, hypothetical_stress_path, var_es, HYPOTHETICAL_SCENARIOS
from core.simulation import ReturnModelSpec, horizon_returns, simulate_portfolio_returns


@pytest.mark.parametrize("model", [SimulationModel.PARAMETRIC, SimulationModel.BOOTSTRAP])
def test_reproducible_with_seed(toy_returns, model):
    spec = ReturnModelSpec(model=model, n_paths=700, n_months=120, seed=11)
    w = np.full(4, 0.25)
    a = simulate_portfolio_returns(toy_returns, w, spec)
    b = simulate_portfolio_returns(toy_returns, w, spec)
    np.testing.assert_array_equal(a, b)
    c = simulate_portfolio_returns(toy_returns, w, ReturnModelSpec(model=model, n_paths=700, n_months=120, seed=12))
    assert not np.array_equal(a, c)


def test_progress_callback_reports_real_progress(toy_returns):
    seen = []
    simulate_portfolio_returns(toy_returns, np.full(4, 0.25),
                               ReturnModelSpec(SimulationModel.PARAMETRIC, 1200, 24, 1), progress=seen.append)
    assert seen == pytest.approx([500 / 1200, 1000 / 1200, 1.0])


def test_parametric_preserves_correlation(toy_returns):
    """Simulate single-asset portfolios and verify cross-asset correlation is retained."""
    spec = ReturnModelSpec(SimulationModel.PARAMETRIC, 2000, 60, 3)
    a = simulate_portfolio_returns(toy_returns, np.array([1, 0, 0, 0.0]), spec).ravel()
    b = simulate_portfolio_returns(toy_returns, np.array([0, 1, 0, 0.0]), spec).ravel()
    hist = np.corrcoef(toy_returns["EQ1"], toy_returns["EQ2"])[0, 1]
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(hist, abs=0.03)


def test_parametric_mean_close_to_history(toy_returns):
    spec = ReturnModelSpec(SimulationModel.PARAMETRIC, 4000, 120, 5)
    sim = simulate_portfolio_returns(toy_returns, np.full(4, 0.25), spec)
    assert sim.mean() == pytest.approx(toy_returns.mean().mean(), abs=4e-4)


def test_fat_tails_increase_kurtosis(toy_returns):
    w = np.array([1, 0, 0, 0.0])
    n = simulate_portfolio_returns(toy_returns, w, ReturnModelSpec(SimulationModel.PARAMETRIC, 2000, 60, 3)).ravel()
    t = simulate_portfolio_returns(toy_returns, w, ReturnModelSpec(SimulationModel.PARAMETRIC, 2000, 60, 3, fat_tails=True, t_dof=4)).ravel()
    kurt = lambda x: np.mean((x - x.mean()) ** 4) / np.var(x) ** 2
    assert kurt(t) > kurt(n) + 0.5
    assert np.std(t) == pytest.approx(np.std(n), rel=0.08)


def test_bootstrap_uses_only_historical_values(toy_returns):
    w = np.full(4, 0.25)
    hist = set(np.round(toy_returns.to_numpy() @ w, 12))
    sim = simulate_portfolio_returns(toy_returns, w, ReturnModelSpec(SimulationModel.BOOTSTRAP, 200, 60, 2, block_months=6))
    assert set(np.round(sim.ravel(), 12)) <= hist


def test_var_es_definitions():
    r = np.linspace(-0.5, 0.5, 1001)
    out = var_es(r, 0.95, 100_000, 1)
    assert out.var == pytest.approx(0.45, abs=1e-3)
    assert out.es > out.var
    assert out.var_amount == pytest.approx(out.var * 100_000)


def test_horizon_returns_compound():
    r = np.full((3, 24), 0.01)
    assert horizon_returns(r, 12) == pytest.approx(np.full(3, 1.01**12 - 1))


def test_lifecycle_mc_probabilities_in_range(toy_returns):
    s = Scenario(profile=InvestorProfile(current_age=35, fire_age=50, life_horizon_age=85))
    nominal = simulate_portfolio_returns(toy_returns, np.full(4, 0.25), ReturnModelSpec(SimulationModel.PARAMETRIC, 500, 600, 9))
    plan = build_plan(s, 0.03)
    out = summarize(run_lifecycle(deflate_returns(nominal[:, : plan.n_months], 0.03), plan, s.withdrawal))
    for k in ("p_fire_by_target_age", "p_not_depleted", "p_spending_covered", "p_success"):
        assert 0.0 <= out[k] <= 1.0
    assert out["p_success"] <= min(out["p_not_depleted"], out["p_spending_covered"]) + 1e-12


def test_stress_injection():
    base = np.zeros((2, 100))
    path = hypothetical_stress_path(HYPOTHETICAL_SCENARIOS["Globalny krach akcyjny"], ["A", "B"],
                                    {"A": "global_equity", "B": "bonds"}, {}, np.array([1.0, 0.0]))
    out = apply_stress(base, path, 10)
    assert np.prod(1 + out[0, 10:28]) == pytest.approx(0.55)
    assert out[0, :10].sum() == 0 and out[0, 28:].sum() == 0
