"""
绩效指标计算模块
"""
import pandas as pd
import numpy as np
from typing import Dict


def calc_metrics(portfolio: pd.DataFrame, annual_trading_days: int = 252) -> Dict[str, float]:
    """
    计算常用绩效指标

    参数:
        portfolio            : 回测引擎返回的 portfolio DataFrame
        annual_trading_days  : 年化交易日数

    返回:
        指标字典
    """
    tv = portfolio["total_value"]
    ret = portfolio["returns"].dropna()

    # ── 总收益 ──
    total_return = (tv.iloc[-1] / tv.iloc[0]) - 1

    # ── 买入持有总收益 ──
    bh_total = (portfolio["bh_value"].iloc[-1] / portfolio["bh_value"].iloc[0]) - 1

    # ── 年化收益率 ──
    n_days = len(tv)
    annual_return = (1 + total_return) ** (annual_trading_days / n_days) - 1

    # ── 年化波动率 ──
    annual_vol = ret.std() * np.sqrt(annual_trading_days)

    # ── 夏普比率（无风险利率 2.5% 年化，近似 A 股国债） ──
    rf_daily = 0.025 / annual_trading_days
    sharpe = (ret.mean() - rf_daily) / ret.std() * np.sqrt(annual_trading_days) if ret.std() > 0 else 0.0

    # ── 最大回撤 ──
    cum_max = tv.cummax()
    drawdown = (tv - cum_max) / cum_max
    max_drawdown = drawdown.min()

    # ── 卡玛比率 ──
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # ── 胜率（按交易对计算） ──
    win_rate = _calc_win_rate(portfolio)

    return {
        "总收益率":        round(total_return * 100, 2),
        "买入持有收益率":  round(bh_total * 100, 2),
        "年化收益率":      round(annual_return * 100, 2),
        "年化波动率":      round(annual_vol * 100, 2),
        "夏普比率":        round(sharpe, 3),
        "最大回撤":        round(max_drawdown * 100, 2),
        "卡玛比率":        round(calmar, 3),
        "胜率(%)":         round(win_rate * 100, 2),
    }


def _calc_win_rate(portfolio: pd.DataFrame) -> float:
    """基于持仓变化，统计盈利交易次数 / 总交易次数"""
    tv = portfolio["total_value"]
    sig = portfolio["signal"]

    buy_indices  = portfolio.index[sig == 1].tolist()
    sell_indices = portfolio.index[sig == -1].tolist()

    if not buy_indices or not sell_indices:
        return 0.0

    wins = 0
    total = 0
    for b in buy_indices:
        # 找下一个卖出信号
        later_sells = [s for s in sell_indices if s > b]
        if not later_sells:
            continue
        s = later_sells[0]
        buy_val  = portfolio.loc[b, "total_value"]
        sell_val = portfolio.loc[s, "total_value"]
        if sell_val > buy_val:
            wins += 1
        total += 1

    return wins / total if total > 0 else 0.0


def print_metrics(metrics: Dict[str, float], strategy_name: str = "") -> None:
    """格式化打印绩效指标"""
    title = f"策略绩效: {strategy_name}" if strategy_name else "策略绩效"
    print("\n" + "=" * 40)
    print(f"  {title}")
    print("=" * 40)
    for k, v in metrics.items():
        unit = "%" if "%" in k or "收益率" in k or "回撤" in k or "波动率" in k else ""
        print(f"  {k:<15}: {v}{unit}")
    print("=" * 40)
