"""
端到端集成测试（使用模拟数据，无需网络）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from stock_strategy.strategies.signals import STRATEGY_MAP
from stock_strategy.backtest.engine import run_backtest
from stock_strategy.backtest.metrics import calc_metrics, print_metrics
from stock_strategy.backtest.optimizer import grid_search, print_top_params
from stock_strategy.visualization.plotter import plot_strategy

# ── 构造模拟数据 ──
np.random.seed(42)
dates = pd.date_range("2021-01-01", periods=600, freq="B")
close = 10 + np.cumsum(np.random.randn(600) * 0.25)
open_ = close * (1 + np.random.randn(600) * 0.005)
high  = close * (1 + np.abs(np.random.randn(600) * 0.01))
low   = close * (1 - np.abs(np.random.randn(600) * 0.01))
vol   = np.random.randint(100000, 800000, 600).astype(float)
df = pd.DataFrame(
    {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
    index=dates
)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)

print("=" * 50)
print("  所有策略回测结果")
print("=" * 50)

for name, fn in STRATEGY_MAP.items():
    df_sig = fn(df.copy())
    result = run_backtest(df_sig, init_cash=100000)
    m = calc_metrics(result["portfolio"])
    strat_label = df_sig["strategy_name"].iloc[0] if "strategy_name" in df_sig.columns else name
    print_metrics(m, strat_label)

    # 保存图表（无弹窗）
    plot_strategy(
        df_sig, result["portfolio"],
        symbol="TEST", strategy_name=strat_label,
        save_dir=output_dir, show=False
    )
    print(f"  图表已保存至 output/")

# ── 参数优化测试 ──
print("\n[INFO] 测试 MA 策略参数优化 ...")
from stock_strategy.strategies.signals import ma_cross_strategy
results_df = grid_search(
    df.copy(), ma_cross_strategy,
    {"fast": [3, 5, 10], "slow": [15, 20, 30]},
    opt_metric="夏普比率", verbose=False
)
print_top_params(results_df, n=3, metric="夏普比率")

print("\n[PASS] 所有测试完成！")
