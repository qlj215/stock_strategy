"""
技术指标计算模块
包含：MA, EMA, RSI, MACD, 布林带
"""
import pandas as pd
import numpy as np


def calc_ma(close: pd.Series, period: int) -> pd.Series:
    """简单移动平均线"""
    return close.rolling(window=period).mean()


def calc_ema(close: pd.Series, period: int) -> pd.Series:
    """指数移动平均线"""
    return close.ewm(span=period, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    相对强弱指数 RSI
    返回值范围 0~100
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    MACD 指标
    返回 DataFrame，列: dif, dea, hist
        dif  = EMA(fast) - EMA(slow)
        dea  = EMA(dif, signal)
        hist = (dif - dea) * 2   (A股常用乘2)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


def calc_bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    布林带
    返回 DataFrame，列: mid, upper, lower
    """
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def add_all_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    一次性计算所有指标并附加到 df

    params 示例:
    {
        "ma_fast": 5, "ma_slow": 20,
        "ema_fast": 12, "ema_slow": 26,
        "rsi_period": 14,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "boll_period": 20, "boll_std": 2.0
    }
    """
    if params is None:
        params = {}

    close = df["close"]

    # MA
    ma_fast = params.get("ma_fast", 5)
    ma_slow = params.get("ma_slow", 20)
    df[f"ma{ma_fast}"] = calc_ma(close, ma_fast)
    df[f"ma{ma_slow}"] = calc_ma(close, ma_slow)

    # EMA
    ema_fast = params.get("ema_fast", 12)
    ema_slow = params.get("ema_slow", 26)
    df[f"ema{ema_fast}"] = calc_ema(close, ema_fast)
    df[f"ema{ema_slow}"] = calc_ema(close, ema_slow)

    # RSI
    rsi_period = params.get("rsi_period", 14)
    df["rsi"] = calc_rsi(close, rsi_period)

    # MACD
    macd = calc_macd(
        close,
        fast=params.get("macd_fast", 12),
        slow=params.get("macd_slow", 26),
        signal=params.get("macd_signal", 9)
    )
    df["macd_dif"] = macd["dif"]
    df["macd_dea"] = macd["dea"]
    df["macd_hist"] = macd["hist"]

    # 布林带
    boll = calc_bollinger(
        close,
        period=params.get("boll_period", 20),
        std_dev=params.get("boll_std", 2.0)
    )
    df["boll_mid"] = boll["mid"]
    df["boll_upper"] = boll["upper"]
    df["boll_lower"] = boll["lower"]

    return df
