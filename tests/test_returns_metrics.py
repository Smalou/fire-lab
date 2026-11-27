import numpy as np
import pandas as pd
import pytest

from core import metrics, returns


def test_monthly_annual_roundtrip():
    assert returns.monthly_to_annual(returns.annual_to_monthly(0.07)) == pytest.approx(0.07)


def test_fisher_exact_not_approximate():
    # (1.08 / 1.03) - 1, not 0.08 - 0.03
    assert returns.nominal_to_real(0.08, 0.03) == pytest.approx(1.08 / 1.03 - 1)
    assert returns.real_to_nominal(returns.nominal_to_real(0.08, 0.03), 0.03) == pytest.approx(0.08)


def test_zero_and_negative_returns():
    assert returns.annualized_geometric_return(np.zeros(24)) == pytest.approx(0.0)
    r = np.full(12, -0.01)
    assert returns.annualized_geometric_return(r) == pytest.approx(0.99**12 - 1)
    assert returns.cumulative_return([-0.5, 1.0]) == pytest.approx(0.0)
    assert returns.annualized_geometric_return([-1.0, 0.1]) == -1.0


def test_arithmetic_vs_geometric():
    g = returns.arithmetic_to_geometric(0.08, 0.20)
    assert g < 0.08
    assert g == pytest.approx(0.08 - 0.5 * 0.2**2 / 1.08, abs=0.005)


def test_max_drawdown_known_path():
    r = [0.10, -0.20, 0.05, -0.10, 0.30]
    idx = np.cumprod([1] + [1 + x for x in r])
    expected = (idx[4] / idx[1]) - 1  # peak after first month, trough after fourth
    assert metrics.max_drawdown(r) == pytest.approx(expected)
    assert metrics.max_drawdown([0.01, 0.02]) == 0.0


def test_sharpe_sortino_annualisation():
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.04, 600)
    rf_m = (1.02) ** (1 / 12) - 1
    expected = (r - rf_m).mean() / (r - rf_m).std(ddof=1) * np.sqrt(12)
    assert metrics.sharpe_ratio(r, 0.02) == pytest.approx(expected)
    assert metrics.sortino_ratio(r, 0.02) > metrics.sharpe_ratio(r, 0.02)


def test_beta_alpha_te_ir():
    rng = np.random.default_rng(1)
    b = rng.normal(0.006, 0.04, 240)
    r = 0.001 + 1.5 * b
    st = metrics.relative_stats(r, b, 0.0)
    assert st.beta == pytest.approx(1.5)
    assert st.alpha == pytest.approx(0.001 * 12, rel=1e-6)
    assert st.tracking_error > 0
    same = metrics.relative_stats(b, b, 0.0)
    assert same.tracking_error == pytest.approx(0.0)


def test_xirr_simple_cases():
    d = [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]
    assert metrics.xirr([-100, 110], d) == pytest.approx(0.10, abs=1e-3)
    assert metrics.xirr([-100, 100], d) == pytest.approx(0.0, abs=1e-9)
    assert metrics.xirr([-100, 90], d) == pytest.approx(-0.10, abs=1e-3)
    assert np.isnan(metrics.xirr([100, 100], d))


def test_monthly_table_shape():
    s = pd.Series(np.arange(24) / 100, index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    t = metrics.monthly_return_table(s)
    assert t.shape == (2, 12)
    assert t.loc[2021, 12] == pytest.approx(0.23)
