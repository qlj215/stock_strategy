"""
回测引擎模块
- 支持全仓/固定份额交易
- 计算每日持仓价值、现金变化
- 返回完整的交易记录和每日净值序列
"""
import pandas as pd
import numpy as np
from typing import Optional


def run_backtest(
    df: pd.DataFrame,
    init_cash: float = 100_000.0,
    commission: float = 0.0003,
    slippage: float = 0.0,
    trade_unit: int = 100,          # A股最小交易单位：100股
    position_pct: float = 1.0,      # 每次买入使用资金比例 (0~1)
) -> dict:
    """
    执行策略回测

    参数:
        df            : 含 close / signal 列的 DataFrame
        init_cash     : 初始资金
        commission    : 手续费率（双边）
        slippage      : 滑点（相对价格的比例）
        trade_unit    : 最小交易手数（股数），A股为100
        position_pct  : 买入时使用的资金比例

    返回:
        {
            "portfolio": DataFrame  每日净值/持仓/现金,
            "trades":    DataFrame  交易记录,
            "signals":   Series     原始信号
        }
    """
    df = df.copy().dropna(subset=["close", "signal"])

    cash = init_cash
    shares = 0
    portfolio_records = []
    trade_records = []

    for date, row in df.iterrows():
        price = row["close"]
        sig   = row["signal"]

        # 买入
        if sig == 1 and shares == 0:
            buy_cash = cash * position_pct
            # 计算可买手数
            price_with_slip = price * (1 + slippage)
            max_shares = int(buy_cash / (price_with_slip * (1 + commission)))
            lots = max_shares // trade_unit
            if lots > 0:
                buy_shares = lots * trade_unit
                cost = buy_shares * price_with_slip * (1 + commission)
                cash -= cost
                shares += buy_shares
                trade_records.append({
                    "date": date,
                    "action": "BUY",
                    "price": round(price_with_slip, 4),
                    "shares": buy_shares,
                    "cost": round(cost, 2),
                    "cash_after": round(cash, 2)
                })

        # 卖出
        elif sig == -1 and shares > 0:
            price_with_slip = price * (1 - slippage)
            revenue = shares * price_with_slip * (1 - commission)
            cash += revenue
            trade_records.append({
                "date": date,
                "action": "SELL",
                "price": round(price_with_slip, 4),
                "shares": shares,
                "revenue": round(revenue, 2),
                "cash_after": round(cash, 2)
            })
            shares = 0

        total_value = cash + shares * price
        portfolio_records.append({
            "date": date,
            "close": price,
            "signal": sig,
            "shares": shares,
            "cash": round(cash, 2),
            "total_value": round(total_value, 2),
        })

    portfolio = pd.DataFrame(portfolio_records).set_index("date")
    portfolio["returns"] = portfolio["total_value"].pct_change()
    portfolio["cum_returns"] = portfolio["total_value"] / init_cash - 1

    # 买入持有基准
    portfolio["bh_value"] = (df["close"] / df["close"].iloc[0]) * init_cash
    portfolio["bh_returns"] = portfolio["bh_value"] / init_cash - 1

    trades = pd.DataFrame(trade_records)
    if not trades.empty:
        trades = trades.set_index("date")

    return {
        "portfolio": portfolio,
        "trades": trades,
        "signals": df["signal"],
    }
