import numpy as np
import pytest

from core import portfolio as pf


def test_moments_annualised_and_psd(toy_returns):
    m = pf.estimate_moments(toy_returns)
    assert m.mu == pytest.approx(toy_returns.mean().to_numpy() * 12)
    assert pf.is_psd(m.cov)
    lw = pf.estimate_moments(toy_returns, "ledoit_wolf")
    assert pf.is_psd(lw.cov) and 0 <= lw.shrinkage_intensity <= 1


def test_nearest_psd_repairs_matrix():
    bad = np.array([[1.0, 0.99, -0.9], [0.99, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert not pf.is_psd(bad)
    assert pf.is_psd(pf.nearest_psd(bad))


def test_weights_sum_and_bounds(toy_returns):
    m = pf.estimate_moments(toy_returns)
    for method in ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]:
        res = pf.optimize_portfolio(method, m, rf=0.02, min_w=0.0, max_w=0.6)
        assert res.success, res.message
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert (res.weights >= -1e-8).all() and (res.weights <= 0.6 + 1e-6).all()


def test_min_variance_is_minimal(toy_returns):
    m = pf.estimate_moments(toy_returns)
    res = pf.min_variance(m.cov)
    v_opt = pf.portfolio_volatility(res.weights, m.cov)
    rng = np.random.default_rng(5)
    for w in rng.dirichlet(np.ones(4), 2000):
        assert pf.portfolio_volatility(w, m.cov) >= v_opt - 1e-7


def test_min_variance_two_asset_closed_form():
    cov = np.array([[0.04, 0.006], [0.006, 0.01]])
    w1 = (cov[1, 1] - cov[0, 1]) / (cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    res = pf.min_variance(cov)
    assert res.weights[0] == pytest.approx(w1, abs=1e-5)


def test_max_sharpe_beats_random(toy_returns):
    m = pf.estimate_moments(toy_returns)
    res = pf.max_sharpe(m.mu, m.cov, 0.01)
    s_opt = pf.sharpe(res.weights, m.mu, m.cov, 0.01)
    rng = np.random.default_rng(6)
    for w in rng.dirichlet(np.ones(4), 2000):
        assert pf.sharpe(w, m.mu, m.cov, 0.01) <= s_opt + 1e-6


def test_max_sharpe_undefined_when_all_below_rf(toy_returns):
    m = pf.estimate_moments(toy_returns)
    res = pf.max_sharpe(m.mu, m.cov, rf=1.0)
    assert not res.success


def test_risk_parity_equal_contributions(toy_returns):
    m = pf.estimate_moments(toy_returns)
    res = pf.risk_parity(m.cov)
    rc = pf.risk_contributions(res.weights, m.cov)
    assert rc.sum() == pytest.approx(pf.portfolio_volatility(res.weights, m.cov))
    assert np.allclose(rc / rc.sum(), 0.25, atol=1e-4)


def test_target_return_and_infeasible(toy_returns):
    m = pf.estimate_moments(toy_returns)
    mid = float(np.mean(m.mu))
    res = pf.target_return_portfolio(m.mu, m.cov, mid)
    assert res.success and pf.portfolio_return(res.weights, m.mu) == pytest.approx(mid, abs=1e-6)
    bad = pf.target_return_portfolio(m.mu, m.cov, 5.0)
    assert not bad.success and "poza zakresem" in bad.message


def test_infeasible_bounds():
    cov = np.eye(3) * 0.01
    assert not pf.min_variance(cov, 0.0, 0.2).success  # 3 * 0.2 < 1


def test_frontier_monotone(toy_returns):
    m = pf.estimate_moments(toy_returns)
    fr = pf.efficient_frontier(m.mu, m.cov, n_points=15)
    assert np.all(np.diff(fr.returns) >= -1e-9)
    assert np.all(np.diff(fr.vols) >= -1e-7)
    assert np.allclose(fr.weights.sum(axis=1), 1.0)


def test_short_selling_allowed(toy_returns):
    m = pf.estimate_moments(toy_returns)
    res = pf.min_variance(m.cov, min_w=-0.5, max_w=1.5, allow_short=True)
    assert res.success and res.weights.sum() == pytest.approx(1.0)
    long_only = pf.min_variance(m.cov)
    assert pf.portfolio_volatility(res.weights, m.cov) <= pf.portfolio_volatility(long_only.weights, m.cov) + 1e-9


def test_capm_beta_of_market_is_one(toy_returns):
    res = pf.capm_analysis(toy_returns, toy_returns["EQ1"], 0.02)
    assert res.betas[0] == pytest.approx(1.0)
    assert res.alphas[0] == pytest.approx(0.0, abs=1e-12)
