import numpy as np
import pandas as pd
import pytest

from alphakit.operators import (
    cs_rank,
    cs_zscore,
    cs_demean,
    cs_winsorize,
    ts_delay,
    ts_delta,
    ts_mean,
    ts_std,
    ts_zscore,
    ts_decay_linear,
    ts_corr,
)


@pytest.fixture
def sample_panel():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    assets = ["A", "B", "C", "D"]
    data = np.arange(40, dtype=float).reshape(10, 4)
    return pd.DataFrame(data, index=dates, columns=assets)


def test_cs_rank(sample_panel):
    ranked = cs_rank(sample_panel, pct=True)
    assert ranked.shape == sample_panel.shape
    # For any row, values should be between 0 and 1
    assert (ranked.iloc[0].values == np.array([0.25, 0.5, 0.75, 1.0])).all()


def test_cs_demean(sample_panel):
    demeaned = cs_demean(sample_panel)
    # Mean across columns should be zero for each row
    assert np.allclose(demeaned.mean(axis=1), 0.0)


def test_cs_zscore(sample_panel):
    z = cs_zscore(sample_panel)
    assert np.allclose(z.mean(axis=1), 0.0)
    assert np.allclose(z.std(axis=1), 1.0)


def test_cs_winsorize():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    assets = [f"A_{i}" for i in range(100)]
    data = np.tile(np.arange(100, dtype=float), (2, 1))
    df = pd.DataFrame(data, index=dates, columns=assets)
    winsorized = cs_winsorize(df, limits=(0.05, 0.05))
    assert winsorized.iloc[0].min() >= 4.0
    assert winsorized.iloc[0].max() <= 95.0


def test_ts_delay_and_delta(sample_panel):
    delayed = ts_delay(sample_panel, 1)
    assert delayed.iloc[0].isna().all()
    assert np.allclose(delayed.iloc[1], sample_panel.iloc[0])

    delta = ts_delta(sample_panel, 1)
    assert delta.iloc[0].isna().all()
    assert np.allclose(delta.iloc[1:], 4.0)


def test_ts_mean_and_std(sample_panel):
    m = ts_mean(sample_panel, 3)
    assert m.iloc[0].isna().all()
    assert m.iloc[1].isna().all()
    assert not m.iloc[2].isna().any()

    s = ts_std(sample_panel, 3)
    assert not s.iloc[2].isna().any()


def test_ts_decay_linear(sample_panel):
    decayed = ts_decay_linear(sample_panel, 3)
    assert decayed.iloc[0].isna().all()
    assert decayed.iloc[1].isna().all()
    # For row 2 (indices 0, 1, 2 for col A: 0, 4, 8):
    # weights: 1/6, 2/6, 3/6 -> 0*1/6 + 4*2/6 + 8*3/6 = 8/6 + 24/6 = 32/6 = 5.3333333333
    assert np.isclose(decayed.loc[decayed.index[2], "A"], 32.0 / 6.0)


def test_ts_corr(sample_panel):
    panel2 = sample_panel * 2.0 + 5.0
    corr = ts_corr(sample_panel, panel2, 5)
    assert np.allclose(corr.dropna(), 1.0)
