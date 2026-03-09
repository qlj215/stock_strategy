"""
参数优化模块
- 对指定策略进行网格搜索
- 返回所有参数组合的绩效指标 DataFrame
"""
import itertools
import pandas as pd
from typing import Dict, List, Callable, Any

from .engine import run_backtest
from .metrics import calc_metrics


def grid_search(
    raw_df: pd.DataFrame,
    strategy_fn: Callable,
    param_grid: Dict[str, List[Any]],
    opt_metric: str = "夏普比率",
    init_cash: float = 100_000.0,
    commission: float = 0.0003,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    对策略进行网格参数搜索

    参数:
        raw_df       : 原始 OHLCV DataFrame（不含指标）
        strategy_fn  : 策略函数，接受 (df, **kwargs) 返回含 signal 列的 df
        param_grid   : 参数网格，如 {"fast": [5, 10], "slow": [20, 30]}
        opt_metric   : 优化目标指标（绩效指标的中文键）
        init_cash    : 初始资金
        commission   : 手续费
        verbose      : 是否打印进度

    返回:
        包含所有参数组合及其绩效指标的 DataFrame，按 opt_metric 降序排列
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    records = []
    total = len(combinations)
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        try:
            df_sig = strategy_fn(raw_df.copy(), **params)
            result = run_backtest(df_sig, init_cash=init_cash, commission=commission)
            metrics = calc_metrics(result["portfolio"])
            row = {**params, **metrics}
            records.append(row)
        except Exception as e:
            if verbose:
                print(f"  [跳过] 参数 {params} 出错: {e}")
            continue

        if verbose and (i + 1) % max(1, total // 10) == 0:
            print(f"  进度: {i + 1}/{total}")

    results_df = pd.DataFrame(records)
    if opt_metric in results_df.columns:
        results_df = results_df.sort_values(opt_metric, ascending=False).reset_index(drop=True)
    return results_df


def print_top_params(results_df: pd.DataFrame, n: int = 5, metric: str = "夏普比率") -> None:
    """打印最优参数组合"""
    print(f"\n{'='*50}")
    print(f"  参数优化结果 TOP {n} ({metric})")
    print(f"{'='*50}")
    cols = [c for c in results_df.columns if c not in ["strategy_name"]]
    print(results_df[cols].head(n).to_string(index=False))
    print(f"{'='*50}")
