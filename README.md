# AlphaKit

[![CI](https://github.com/Sunzzx/alphakit/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunzzx/alphakit/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**AlphaKit** is a high-performance quantitative alpha factor research, calculation, and evaluation toolkit designed for quantitative researchers, portfolio managers, and algorithmic trading engineers.

It offers high-speed vectorised expressions for standard cross-sectional and time-series operators, robust performance diagnostics (Information Coefficient, Rank IC, IC-IR, turnover, drawdown), and synthetic market generation.

---

## Key Features

- **Cross-Sectional Alpha Operators**: Ranking (`cs_rank`), Z-score normalization (`cs_zscore`), market neutralization (`cs_demean`), and outlier clipping (`cs_winsorize`).
- **Time-Series Alpha Operators**: Rolling delays (`ts_delay`), momentum deltas (`ts_delta`), rolling statistics (`ts_mean`, `ts_std`, `ts_zscore`), linear decay filtering (`ts_decay_linear`), and rolling pairwise cross-asset correlation (`ts_corr`).
- **Performance & Risk Diagnostics**: Cross-sectional Spearman/Pearson Information Coefficient (`compute_ic`), summary statistics (`calculate_ic_stats`), portfolio turnover estimation (`calculate_turnover`), and dollar-neutral long/short quantile backtesting (`backtest_long_short`).
- **Synthetic Market Simulation**: Realistic multi-asset panel data generator (`generate_synthetic_panel`) modeling common factor beta shocks and idiosyncratic price paths.

---

## Installation

```bash
pip install alphakit
```

For local development:
```bash
git clone https://github.com/Sunzzx/alphakit.git
cd alphakit
pip install -e ".[dev]"
```

---

## Quick Start

```python
from alphakit import (
    generate_synthetic_panel,
    cs_rank,
    ts_delta,
    compute_ic,
    calculate_ic_stats,
    backtest_long_short,
)

# 1. Generate synthetic market data
prices, forward_returns, volume = generate_synthetic_panel(n_days=252, n_assets=50)

# 2. Formulate a mean-reversion alpha: -1 * cs_rank(ts_delta(prices, 5))
alpha = -1.0 * cs_rank(ts_delta(prices, 5))

# 3. Evaluate Information Coefficient (IC)
ic_series = compute_ic(alpha, forward_returns, method="spearman")
ic_stats = calculate_ic_stats(ic_series)
print(f"Mean Rank IC: {ic_stats['mean_ic']:.4f}")
print(f"IC-IR: {ic_stats['ic_ir']:.2f}")

# 4. Run Long-Short quantile backtest
results = backtest_long_short(alpha, forward_returns, top_q=0.2, bottom_q=0.2)
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
```

---

## Testing

Run the comprehensive test suite with `pytest`:

```bash
pytest -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
