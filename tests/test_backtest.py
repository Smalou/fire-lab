import numpy as np
import pandas as pd
import pytest

from core.backtest import run_backtest
from core.models import BacktestSettings


def settings(**kw):
    base = dict(rebalance="annual", initial_investment=10_000, monthly_contribution=0, commission_rate=0.0,
                spread_bps=0.0, estimation_months=36)
    base.update(kw)
    return BacktestSettings(**base)


@pytest.mark.parametrize("strategy", ["buy_hold", "scenario", "equal_weight", "sixty_forty", "min_variance", "max_sharpe", "risk_parity", "glide_path"])
def test_all_strategies_run(toy_returns, toy_classes, strategy):
    res = run_backtest(toy_returns, strategy, toy_classes, [0.4, 0.2, 0.3, 0.1], settings(), rf=0.02,
                       benchmark=toy_returns["EQ1"])
    assert len(res.values) == len(toy_returns) - 36
    assert np.allclose(res.weights.sum(axis=1), 1.0)
    assert np.isfinite(res.metrics["cagr"])


def test_no_lookahead(toy_returns, toy_classes):
    """Changing returns after month k must not change any decision taken up to month k."""
    k = 100
    base = run_backtest(toy_returns, "min_variance", toy_classes, [0.25] * 4, settings(rebalance="monthly"), rf=0.02)
    altered = toy_returns.copy()
    altered.iloc[k:] = altered.iloc[k:] * -3 + 0.05
    alt = run_backtest(altered, "min_variance", toy_classes, [0.25] * 4, settings(rebalance="monthly"), rf=0.02)
    # decisions for months <= k use data < k only
    pd.testing.assert_frame_equal(base.weights.iloc[: k - 36 + 1], alt.weights.iloc[: k - 36 + 1])
    assert not base.weights.iloc[k - 36 + 5 :].equals(alt.weights.iloc[k - 36 + 5 :])


def test_buy_hold_without_flows_matches_asset_growth(toy_returns, toy_classes):
    w = np.array([0.5, 0.0, 0.5, 0.0])
    res = run_backtest(toy_returns, "buy_hold", toy_classes, w, settings(), rf=0.0)
    r = toy_returns.iloc[36:]
    expected = 10_000 * (0.5 * (1 + r["EQ1"]).prod() + 0.5 * (1 + r["BOND"]).prod())
    assert res.values.iloc[-1] == pytest.approx(expected)


def test_contributions_separated_from_performance(toy_returns, toy_classes):
    a = run_backtest(toy_returns, "equal_weight", toy_classes, [0.25] * 4, settings(monthly_contribution=0), rf=0.0)
    b = run_backtest(toy_returns, "equal_weight", toy_classes, [0.25] * 4, settings(monthly_contribution=1_000), rf=0.0)
    assert b.values.iloc[-1] > a.values.iloc[-1]
    # With zero costs and monthly rebalancing TWR is independent of contributions
    a_m = run_backtest(toy_returns, "equal_weight", toy_classes, [0.25] * 4, settings(rebalance="monthly"), rf=0.0)
    b_m = run_backtest(toy_returns, "equal_weight", toy_classes, [0.25] * 4, settings(rebalance="monthly", monthly_contribution=1_000), rf=0.0)
    np.testing.assert_allclose(a_m.twr_returns, b_m.twr_returns, atol=1e-12)
    assert b.metrics["investment_gain"] == pytest.approx(b.values.iloc[-1] - b.invested.iloc[-1])


def test_costs_reduce_performance(toy_returns, toy_classes):
    free = run_backtest(toy_returns, "equal_weight", toy_classes, [0.25] * 4, settings(rebalance="monthly"), rf=0.0)
    costly = run_backtest(toy_returns, "equal_weight", toy_classes, [0.25] * 4, settings(rebalance="monthly", commission_rate=0.01), rf=0.0)
    assert costly.values.iloc[-1] < free.values.iloc[-1]
    assert costly.metrics["total_costs"] > 0


def test_xirr_equals_cagr_without_contributions(toy_returns, toy_classes):
    res = run_backtest(toy_returns, "equal_weight", toy_classes, [0.25] * 4, settings(rebalance="monthly"), rf=0.0)
    assert res.metrics["xirr"] == pytest.approx(res.metrics["cagr"], abs=2e-3)


def test_too_short_history_raises(toy_returns, toy_classes):
    with pytest.raises(ValueError):
        run_backtest(toy_returns.iloc[:40], "min_variance", toy_classes, [0.25] * 4, settings(), rf=0.0)
