"""Synthetic market data generation for alpha prototyping and benchmarking."""
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def generate_synthetic_panel(
    n_days: int = 252,
    n_assets: int = 50,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic market panel of (prices, forward_returns, volume).

    Simulates multi-asset prices using a one-factor market model with idiosyncratic
    stochastic volatility and log-normal volume distributions.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    tickers = [f"ASSET_{i:03d}" for i in range(n_assets)]

    # Common market shock + idiosyncratic shocks
    market_shock = rng.normal(loc=0.0004, scale=0.012, size=(n_days, 1))
    betas = rng.uniform(0.6, 1.4, size=(1, n_assets))
    idio_shocks = rng.normal(loc=0.0, scale=0.018, size=(n_days, n_assets))

    log_returns = market_shock * betas + idio_shocks
    # Cumulative prices
    initial_prices = rng.uniform(20.0, 150.0, size=(1, n_assets))
    price_paths = initial_prices * np.exp(np.cumsum(log_returns, axis=0))

    close_prices = pd.DataFrame(price_paths, index=dates, columns=tickers)
    # Forward 1-day returns
    forward_returns = close_prices.pct_change().shift(-1)

    # Realistic volume (log-normal with price correlation)
    base_vol = rng.lognormal(mean=13.0, sigma=0.8, size=(n_days, n_assets))
    volume = pd.DataFrame(base_vol, index=dates, columns=tickers)

    return close_prices, forward_returns, volume
