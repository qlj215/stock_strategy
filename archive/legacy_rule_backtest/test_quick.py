"""
快速测试脚本（验证所有模块）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from strategies.signals import STRATEGY_MAP
from backtest.engine import run_backtest
from backtest.metrics import calc_metrics

# 构造模拟数据
np.random.seed(42)
dates = pd.date_range("2022-01-01", periods=500, freq="B")
close = 10 + np.cumsum(np.random.randn(500) * 0.2)
open_ = close * (1 + np.random.randn(500) * 0.005)
high  = close * (1 + np.abs(np.random.randn(500) * 0.01))
low   = close * (1 - np.abs(np.random.randn(500) * 0.01))
vol   = np.random.randint(100000, 500000, 500).astype(float)
df = pd.DataFrame(
    {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
    index=dates
)

for name, fn in STRATEGY_MAP.items():
    df_sig = fn(df.copy())
    result = run_backtest(df_sig)
    m = calc_metrics(result["portfolio"])
    sharpe = m["夏普比率"]
    total  = m["总收益率"]
    print(f"[OK] {name}: Sharpe={sharpe}, TotalReturn={total}%")

print("\n所有模块测试通过！")
