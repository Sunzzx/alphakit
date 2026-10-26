"""Core time-series and cross-sectional alpha formula operators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cs_rank(df: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Cross-sectional percentile rank across assets (columns) for each timestamp (row)."""
    return df.rank(axis=1, pct=pct)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional standard score (Z-Score) across assets at each timestamp."""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional market neutralization (subtracting cross-sectional mean)."""
    return df.sub(df.mean(axis=1), axis=0)


def cs_winsorize(df: pd.DataFrame, limits: tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
    """Winsorize cross-sectional outliers at given quantile boundaries."""
    lower_q = df.quantile(limits[0], axis=1)
    upper_q = df.quantile(1.0 - limits[1], axis=1)
    return df.clip(lower=lower_q, upper=upper_q, axis=0)


def ts_delay(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Lagged values of the time series by N periods."""
    return df.shift(periods)


def ts_delta(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Difference of values from N periods ago: df - ts_delay(df, periods)."""
    return df.diff(periods)


def ts_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series mean over window."""
    return df.rolling(window=window, min_periods=window).mean()


def ts_std(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series standard deviation over window."""
    return df.rolling(window=window, min_periods=window).std()


def ts_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling time-series z-score over window."""
    r = df.rolling(window=window, min_periods=window)
    mean = r.mean()
    std = r.std().replace(0, np.nan)
    return (df - mean) / std


def ts_decay_linear(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Linearly decaying weighted average over past window days (weights: 1, 2, ..., d)."""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def _weighted_avg(x: np.ndarray) -> float:
        if len(x) < window or np.isnan(x).any():
            return np.nan
        return float(np.dot(x, weights))

    return df.rolling(window=window, min_periods=window).apply(_weighted_avg, raw=True)


def ts_corr(df1: pd.DataFrame, df2: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling pairwise correlation between two panels over window."""
    return df1.rolling(window=window, min_periods=window).corr(df2)
