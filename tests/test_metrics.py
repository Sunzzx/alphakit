import numpy as np
import pandas as pd
import pytest

from alphakit.metrics import (
    compute_ic,
    calculate_ic_stats,
    calculate_turnover,
    backtest_long_short,
)


@pytest.fixture
def market_panels():
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    assets = [f"TICKER_{i}" for i in range(20)]
    rng = np.random.default_rng(123)
    alpha_vals = rng.normal(size=(50, 20))
    # correlated forward returns
    returns_vals = alpha_vals * 0.15 + rng.normal(scale=0.8, size=(50, 20))
    alpha = pd.DataFrame(alpha_vals, index=dates, columns=assets)
    forward_returns = pd.DataFrame(returns_vals, index=dates, columns=assets)
    return alpha, forward_returns


def test_compute_ic(market_panels):
    alpha, forward_returns = market_panels
    ic_series = compute_ic(alpha, forward_returns, method="spearman")
    assert len(ic_series) == 50
    assert not ic_series.isna().all()
    stats = calculate_ic_stats(ic_series)
    assert "mean_ic" in stats
    assert "ic_ir" in stats
    assert stats["mean_ic"] > 0  # Constructed with positive correlation


def test_calculate_turnover():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    assets = ["A", "B"]
    # Day 0: [0.5, -0.5]
    # Day 1: [0.5, -0.5] (no change -> turnover 0)
    # Day 2: [-0.5, 0.5] (diff is [1.0, 1.0] -> sum 2.0 -> turnover 1.0)
    w = pd.DataFrame([[0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]], index=dates, columns=assets)
    t = calculate_turnover(w)
    assert np.isnan(t.iloc[0]) or t.iloc[0] == 0.0
    assert np.isclose(t.iloc[1], 0.0)
    assert np.isclose(t.iloc[2], 1.0)


def test_backtest_long_short(market_panels):
    alpha, forward_returns = market_panels
    bt = backtest_long_short(alpha, forward_returns, top_q=0.2, bottom_q=0.2)
    assert "annualized_return" in bt
    assert "annualized_volatility" in bt
    assert "sharpe_ratio" in bt
    assert "max_drawdown" in bt
    assert "mean_daily_turnover" in bt
    assert isinstance(bt["returns"], pd.Series)
    assert len(bt["returns"]) == len(alpha)
