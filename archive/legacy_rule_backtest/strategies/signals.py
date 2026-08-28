"""
策略信号生成模块
每个策略函数接收带有技术指标的 DataFrame，返回添加了 signal 列的 DataFrame
signal:  1 = 买入,  -1 = 卖出,  0 = 观望
"""
import pandas as pd
import numpy as np
from .indicators import add_all_indicators


def _shift_signal(signal: pd.Series) -> pd.Series:
    """将信号向后移一天，避免当天信号当天成交的未来数据泄露"""
    return signal.shift(1).fillna(0)


# ─────────────────────────────────────────────
# 1. MA 双均线金叉死叉策略
# ─────────────────────────────────────────────
def ma_cross_strategy(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """
    快均线上穿慢均线 → 买入
    快均线下穿慢均线 → 卖出
    """
    df = df.copy()
    df[f"ma{fast}"] = df["close"].rolling(fast).mean()
    df[f"ma{slow}"] = df["close"].rolling(slow).mean()

    signal = pd.Series(0, index=df.index)
    cross_up = (df[f"ma{fast}"] > df[f"ma{slow}"]) & (df[f"ma{fast}"].shift(1) <= df[f"ma{slow}"].shift(1))
    cross_dn = (df[f"ma{fast}"] < df[f"ma{slow}"]) & (df[f"ma{fast}"].shift(1) >= df[f"ma{slow}"].shift(1))
    signal[cross_up] = 1
    signal[cross_dn] = -1
    df["signal"] = _shift_signal(signal)
    df["strategy_name"] = f"MA({fast},{slow})"
    return df


# ─────────────────────────────────────────────
# 2. RSI 超买超卖策略
# ─────────────────────────────────────────────
def rsi_strategy(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0
) -> pd.DataFrame:
    """
    RSI < oversold → 买入
    RSI > overbought → 卖出
    """
    from .indicators import calc_rsi
    df = df.copy()
    df["rsi"] = calc_rsi(df["close"], period)

    signal = pd.Series(0, index=df.index)
    buy  = (df["rsi"] < oversold)  & (df["rsi"].shift(1) >= oversold)
    sell = (df["rsi"] > overbought) & (df["rsi"].shift(1) <= overbought)
    signal[buy]  = 1
    signal[sell] = -1
    df["signal"] = _shift_signal(signal)
    df["strategy_name"] = f"RSI({period},{oversold},{overbought})"
    return df


# ─────────────────────────────────────────────
# 3. MACD 策略
# ─────────────────────────────────────────────
def macd_strategy(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    DIF 上穿 DEA（金叉）→ 买入
    DIF 下穿 DEA（死叉）→ 卖出
    """
    from .indicators import calc_macd
    df = df.copy()
    macd = calc_macd(df["close"], fast, slow, signal_period)
    df["macd_dif"] = macd["dif"]
    df["macd_dea"] = macd["dea"]
    df["macd_hist"] = macd["hist"]

    signal = pd.Series(0, index=df.index)
    cross_up = (macd["dif"] > macd["dea"]) & (macd["dif"].shift(1) <= macd["dea"].shift(1))
    cross_dn = (macd["dif"] < macd["dea"]) & (macd["dif"].shift(1) >= macd["dea"].shift(1))
    signal[cross_up] = 1
    signal[cross_dn] = -1
    df["signal"] = _shift_signal(signal)
    df["strategy_name"] = f"MACD({fast},{slow},{signal_period})"
    return df


# ─────────────────────────────────────────────
# 4. 布林带策略
# ─────────────────────────────────────────────
def bollinger_strategy(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    收盘价跌破下轨 → 买入
    收盘价突破上轨 → 卖出
    """
    from .indicators import calc_bollinger
    df = df.copy()
    boll = calc_bollinger(df["close"], period, std_dev)
    df["boll_upper"] = boll["upper"]
    df["boll_mid"]   = boll["mid"]
    df["boll_lower"] = boll["lower"]

    signal = pd.Series(0, index=df.index)
    buy  = (df["close"] < boll["lower"]) & (df["close"].shift(1) >= boll["lower"].shift(1))
    sell = (df["close"] > boll["upper"]) & (df["close"].shift(1) <= boll["upper"].shift(1))
    signal[buy]  = 1
    signal[sell] = -1
    df["signal"] = _shift_signal(signal)
    df["strategy_name"] = f"Bollinger({period},{std_dev})"
    return df


# ─────────────────────────────────────────────
# 策略注册表
# ─────────────────────────────────────────────
STRATEGY_MAP = {
    "ma":        ma_cross_strategy,
    "rsi":       rsi_strategy,
    "macd":      macd_strategy,
    "bollinger": bollinger_strategy,
}


def get_strategy(name: str):
    """根据名称获取策略函数"""
    if name not in STRATEGY_MAP:
        raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_MAP.keys())}")
    return STRATEGY_MAP[name]
