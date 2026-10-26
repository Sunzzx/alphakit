"""Performance evaluation metrics for quantitative alpha factors."""
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def compute_ic(
    alpha: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Calculate cross-sectional Information Coefficient (IC / Rank IC) across time."""
    aligned_alpha, aligned_returns = alpha.align(forward_returns, join="inner")
    ic_values = {}
    
    for dt in aligned_alpha.index:
        a_row = aligned_alpha.loc[dt]
        r_row = aligned_returns.loc[dt]
        valid = ~(a_row.isna() | r_row.isna())
        if valid.sum() > 2:
            corr = a_row[valid].corr(r_row[valid], method=method)
            ic_values[dt] = corr
        else:
            ic_values[dt] = np.nan
            
    return pd.Series(ic_values, name="ic")


def calculate_ic_stats(ic_series: pd.Series) -> Dict[str, float]:
    """Calculate aggregate statistics on an IC series."""
    clean_ic = ic_series.dropna()
    if len(clean_ic) == 0:
        return {"mean_ic": 0.0, "std_ic": 0.0, "ic_ir": 0.0, "positive_ratio": 0.0}
    mean = float(clean_ic.mean())
    std = float(clean_ic.std(ddof=1)) if len(clean_ic) > 1 else 0.0
    ir = (mean / std * np.sqrt(252)) if std > 0 else 0.0
    pos_ratio = float((clean_ic > 0).mean())
    return {
        "mean_ic": mean,
        "std_ic": std,
        "ic_ir": ir,
        "positive_ratio": pos_ratio,
    }


def calculate_turnover(weights: pd.DataFrame) -> pd.Series:
    """Calculate day-over-day portfolio turnover: 0.5 * sum(|w_t - w_{t-1}|)."""
    w = weights.fillna(0.0)
    delta = w.diff().abs()
    return delta.sum(axis=1) * 0.5


def backtest_long_short(
    alpha: pd.DataFrame,
    forward_returns: pd.DataFrame,
    top_q: float = 0.2,
    bottom_q: float = 0.2,
) -> Dict[str, Any]:
    """Simulate a dollar-neutral long/short quantile portfolio based on factor scores."""
    aligned_alpha, aligned_returns = alpha.align(forward_returns, join="inner")
    dates = aligned_alpha.index
    
    pnl_records = []
    weights_list = []
    
    for dt in dates:
        a_row = aligned_alpha.loc[dt]
        r_row = aligned_returns.loc[dt]
        valid = ~(a_row.isna() | r_row.isna())
        
        if valid.sum() < 4:
            pnl_records.append(0.0)
            weights_list.append(pd.Series(0.0, index=a_row.index))
            continue
            
        a_valid = a_row[valid]
        top_thresh = a_valid.quantile(1.0 - top_q)
        bot_thresh = a_valid.quantile(bottom_q)
        
        longs = a_valid >= top_thresh
        shorts = a_valid <= bot_thresh
        
        w = pd.Series(0.0, index=a_row.index)
        if longs.sum() > 0:
            w[longs[longs].index] = 1.0 / longs.sum()
        if shorts.sum() > 0:
            w[shorts[shorts].index] = -1.0 / shorts.sum()
            
        daily_return = float((w * r_row.fillna(0.0)).sum())
        pnl_records.append(daily_return)
        weights_list.append(w)
        
    returns_series = pd.Series(pnl_records, index=dates, name="strategy_return")
    weights_df = pd.DataFrame(weights_list, index=dates)
    turnover_series = calculate_turnover(weights_df)
    
    cum_return = (1.0 + returns_series).cumprod() - 1.0
    peak = (1.0 + returns_series).cumprod().cummax()
    drawdowns = ((1.0 + returns_series).cumprod() - peak) / peak
    max_drawdown = float(drawdowns.min())
    
    ann_mean = float(returns_series.mean() * 252)
    ann_vol = float(returns_series.std(ddof=1) * np.sqrt(252)) if len(returns_series) > 1 else 0.0
    sharpe = (ann_mean / ann_vol) if ann_vol > 0 else 0.0
    
    return {
        "returns": returns_series,
        "annualized_return": ann_mean,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "mean_daily_turnover": float(turnover_series.mean()),
    }
