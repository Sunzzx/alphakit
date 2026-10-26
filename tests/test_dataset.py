import pandas as pd
import pytest
from alphakit.dataset import generate_synthetic_panel


def test_generate_synthetic_panel():
    prices, returns, volume = generate_synthetic_panel(n_days=100, n_assets=15, seed=99)
    assert isinstance(prices, pd.DataFrame)
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(volume, pd.DataFrame)
    assert prices.shape == (100, 15)
    assert returns.shape == (100, 15)
    assert volume.shape == (100, 15)
    assert (prices > 0).all().all()
    assert (volume > 0).all().all()
