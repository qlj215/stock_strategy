"""
数据获取模块 - 使用AKShare获取A股历史数据（增强版）
改进点：
1) 自动重试（指数退避）
2) 临时禁用系统代理再试一次（应对 ProxyError）
3) 自动缓存到本地，网络失败时可回退读取缓存
"""
import os
import time
from contextlib import contextmanager
from typing import Optional

import akshare as ak
import pandas as pd


PROXY_ENV_KEYS = [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]


@contextmanager
def _temp_disable_proxies():
    """临时清理代理环境变量，退出后恢复。"""
    backup = {k: os.environ.get(k) for k in PROXY_ENV_KEYS}
    try:
        for k in PROXY_ENV_KEYS:
            if k in os.environ:
                del os.environ[k]
        yield
    finally:
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _normalize_akshare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("返回数据为空")

    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "price_chg",
        "换手率": "turnover",
    }
    df = df.rename(columns=col_map)

    if "date" not in df.columns:
        raise ValueError("缺少日期列")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    core_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not core_cols:
        raise ValueError("缺少OHLCV核心列")
    df = df[core_cols].copy().astype(float)
    return df


def _to_tx_symbol(symbol: str) -> str:
    if symbol.startswith(("sh", "sz", "bj")):
        return symbol
    if symbol.startswith(("6", "9")):
        return f"sh{symbol}"
    if symbol.startswith(("0", "2", "3")):
        return f"sz{symbol}"
    return symbol


def _fetch_once(symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    # 主通道：东方财富 hist
    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    return _normalize_akshare_df(raw)


def _fetch_tx_once(symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    # 备通道：腾讯 hist_tx，symbol需带交易所前缀（sh/sz）
    tx_symbol = _to_tx_symbol(symbol)
    raw = ak.stock_zh_a_hist_tx(
        symbol=tx_symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    # hist_tx 常见列：date/open/close/high/low/amount
    if raw is None or raw.empty:
        raise ValueError("hist_tx 返回空数据")
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()
    keep = [c for c in ["open", "high", "low", "close", "amount", "volume"] if c in raw.columns]
    if "amount" in keep and "volume" not in keep:
        raw = raw.rename(columns={"amount": "volume"})
        keep = ["open", "high", "low", "close", "volume"]
    else:
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    return raw[keep].astype(float)


def fetch_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    retries: int = 3,
    retry_sleep: float = 1.5,
    cache_dir: str = "output/cache",
) -> pd.DataFrame:
    """
    获取股票日线数据，支持重试与缓存回退。

    流程：
    A. 常规请求重试
    B. 若失败，禁用代理后再重试
    C. 若仍失败，尝试读取本地缓存
    """
    last_err: Optional[Exception] = None

    # A) 常规重试
    for i in range(retries):
        try:
            df = _fetch_once(symbol, start_date, end_date, adjust)
            save_data(df, f"{symbol}_{start_date}_{end_date}", data_dir=cache_dir)
            return df
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(retry_sleep * (i + 1))

    # B) 禁代理再试
    for i in range(max(1, retries - 1)):
        try:
            with _temp_disable_proxies():
                df = _fetch_once(symbol, start_date, end_date, adjust)
            save_data(df, f"{symbol}_{start_date}_{end_date}", data_dir=cache_dir)
            return df
        except Exception as e:
            last_err = e
            if i < max(1, retries - 1) - 1:
                time.sleep(retry_sleep * (i + 1))

    # C) 备通道：腾讯 hist_tx
    for i in range(retries):
        try:
            df = _fetch_tx_once(symbol, start_date, end_date, adjust)
            save_data(df, f"{symbol}_{start_date}_{end_date}", data_dir=cache_dir)
            return df
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(retry_sleep * (i + 1))

    # D) 缓存回退
    cache_path = os.path.join(cache_dir, f"{symbol}_{start_date}_{end_date}.csv")
    if os.path.exists(cache_path):
        df_cache = load_data(cache_path)
        if not df_cache.empty:
            return df_cache

    raise RuntimeError(f"获取股票 {symbol} 数据失败（主通道/禁代理/备通道/缓存均失败）: {last_err}")


def save_data(df: pd.DataFrame, symbol: str, data_dir: str = "output") -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{symbol}.csv")
    df.to_csv(path)
    return path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.name is None:
        df.index.name = "date"
    return df
