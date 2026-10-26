"""AlphaKit: High-performance quantitative alpha factor research, calculation, and evaluation toolkit."""

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
from alphakit.metrics import (
    compute_ic,
    calculate_ic_stats,
    calculate_turnover,
    backtest_long_short,
)
from alphakit.dataset import generate_synthetic_panel

__version__ = "0.1.0"

__all__ = [
    "cs_rank",
    "cs_zscore",
    "cs_demean",
    "cs_winsorize",
    "ts_delay",
    "ts_delta",
    "ts_mean",
    "ts_std",
    "ts_zscore",
    "ts_decay_linear",
    "ts_corr",
    "compute_ic",
    "calculate_ic_stats",
    "calculate_turnover",
    "backtest_long_short",
    "generate_synthetic_panel",
    "__version__",
]
