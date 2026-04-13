"""
数据获取模块 - 使用 MiniQMT + xtdata 获取 A 股数据

说明：
1) 不再依赖 AKShare 的公网抓取链路，避免 Eastmoney 限流/风控导致的不稳定。
2) 行情由本机 MiniQMT 提供；本模块可按配置自动尝试拉起 MiniQMT。
3) 保留原 fetch_stock_data 接口签名，尽量减少上层调用改动。
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


_MINIQMT_LAUNCH_ATTEMPTED = False
_MINIQMT_LAUNCH_RESULT: Dict[str, Any] = {
    "attempted": False,
    "success": False,
    "path": "",
    "detail": "",
}


def _is_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME")) or ("microsoft" in os.uname().release.lower())


def _wsl_to_windows_path(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    if re.match(r"^[A-Za-z]:\\", p):
        return p
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", p)
    if not m:
        return p
    drive = m.group(1).upper()
    rest = m.group(2).replace("/", "\\")
    return f"{drive}:\\{rest}"


def _windows_to_wsl_path(path: str) -> str:
    p = str(path or "").strip()
    m = re.match(r"^([A-Za-z]):\\(.*)$", p)
    if not m:
        return p
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _path_exists_cross_platform(path: str) -> bool:
    p = str(path or "").strip()
    if not p:
        return False
    if os.path.exists(p):
        return True
    alt = _windows_to_wsl_path(p)
    return alt != p and os.path.exists(alt)


def _candidate_miniqmt_paths() -> List[str]:
    candidates = [
        os.environ.get("MINIQMT_EXE", ""),
        os.environ.get("XTMINIQMT_EXE", ""),
        os.environ.get("QMT_EXE", ""),
        r"C:\Users\24333\国金证券QMT交易端\bin.x64\XtMiniQmt.exe",
    ]

    out: List[str] = []
    seen = set()
    for p in candidates:
        p = str(p or "").strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _launch_miniqmt_if_needed() -> None:
    global _MINIQMT_LAUNCH_ATTEMPTED, _MINIQMT_LAUNCH_RESULT

    auto_start = str(os.environ.get("MINIQMT_AUTO_START", "1")).strip().lower() not in {"0", "false", "no", "off"}
    if not auto_start or _MINIQMT_LAUNCH_ATTEMPTED:
        return

    _MINIQMT_LAUNCH_ATTEMPTED = True
    _MINIQMT_LAUNCH_RESULT = {
        "attempted": True,
        "success": False,
        "path": "",
        "detail": "未找到可用 MiniQMT 路径",
    }

    for candidate in _candidate_miniqmt_paths():
        if not _path_exists_cross_platform(candidate):
            continue

        try:
            if os.name == "nt":
                subprocess.Popen([candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif _is_wsl():
                win_path = _wsl_to_windows_path(candidate)
                subprocess.Popen(
                    ["cmd.exe", "/C", "start", "", win_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # 非 Windows/WSL 环境不强行启动，避免误调用 wine 等未知环境
                _MINIQMT_LAUNCH_RESULT = {
                    "attempted": True,
                    "success": False,
                    "path": candidate,
                    "detail": "当前环境不是 Windows/WSL，跳过 MiniQMT 自动启动",
                }
                return

            wait_sec = float(os.environ.get("MINIQMT_STARTUP_WAIT_SEC", "3"))
            time.sleep(max(0.0, wait_sec))
            _MINIQMT_LAUNCH_RESULT = {
                "attempted": True,
                "success": True,
                "path": candidate,
                "detail": "已尝试自动启动 MiniQMT",
            }
            return
        except Exception as e:
            _MINIQMT_LAUNCH_RESULT = {
                "attempted": True,
                "success": False,
                "path": candidate,
                "detail": str(e),
            }


def get_miniqmt_launch_status() -> Dict[str, Any]:
    return dict(_MINIQMT_LAUNCH_RESULT)


def _load_xtdata():
    _launch_miniqmt_if_needed()
    try:
        from xtquant import xtdata  # type: ignore

        return xtdata
    except Exception as e:
        launch_msg = ""
        if _MINIQMT_LAUNCH_RESULT.get("attempted"):
            launch_msg = (
                f" 自动启动状态: success={_MINIQMT_LAUNCH_RESULT.get('success')},"
                f" path={_MINIQMT_LAUNCH_RESULT.get('path')},"
                f" detail={_MINIQMT_LAUNCH_RESULT.get('detail')}。"
            )
        raise RuntimeError(
            "未检测到可用 xtquant/xtdata。请使用 Python 3.6~3.13 安装 xtquant，"
            "并确认 MiniQMT 客户端可用。"
            + launch_msg
        ) from e


def _to_xt_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s:
        raise ValueError("symbol 不能为空")

    if "." in s:
        code, market = s.split(".", 1)
        return f"{code.zfill(6)}.{market}"

    if s.startswith(("SH", "SZ", "BJ")) and s[2:].isdigit():
        return f"{s[2:].zfill(6)}.{s[:2]}"

    code = "".join(ch for ch in s if ch.isdigit())
    if not code:
        raise ValueError(f"无法识别 symbol: {symbol}")
    code = code.zfill(6)

    if code.startswith(("6", "9")):
        market = "SH"
    elif code.startswith(("8", "4")):
        market = "BJ"
    else:
        market = "SZ"

    return f"{code}.{market}"


def _to_plain_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if "." in s:
        return s.split(".", 1)[0].zfill(6)
    if s.startswith(("SH", "SZ", "BJ")) and s[2:].isdigit():
        return s[2:].zfill(6)
    if s.isdigit():
        return s.zfill(6)
    return s


def _map_adjust_to_dividend_type(adjust: Optional[str]) -> str:
    a = (adjust or "qfq").lower().strip()
    mapping = {
        "qfq": "front",
        "hfq": "back",
        "none": "none",
        "front": "front",
        "back": "back",
        "front_ratio": "front_ratio",
        "back_ratio": "back_ratio",
    }
    return mapping.get(a, "front")


def _normalize_xt_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("返回数据为空")

    out = df.copy()

    if "time" in out.columns:
        dt = pd.to_datetime(
            pd.to_numeric(out["time"], errors="coerce"),
            unit="ms",
            errors="coerce",
            utc=True,
        )
        # xtdata 时间戳按 UTC 纪元给出，这里统一转为 Asia/Shanghai 本地时间
        dt = dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        out = out.assign(_dt=dt).dropna(subset=["_dt"]).set_index("_dt")
    else:
        # 兼容极少数情况下 time 列缺失，尝试使用索引
        idx = pd.to_datetime(out.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("Asia/Shanghai").tz_localize(None)
        out = out.assign(_dt=idx).dropna(subset=["_dt"]).set_index("_dt")

    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
    }

    # 仅保留存在列
    keep_cols = [c for c in rename_map if c in out.columns]
    out = out[keep_cols].rename(columns=rename_map)

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "close" not in out.columns:
        raise ValueError("返回数据缺少 close 列")

    out = out.sort_index()
    out.index.name = "date"
    return out


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
    获取股票日线数据（MiniQMT + xtdata）。

    返回格式与旧版保持一致：
    index=date, columns 至少包含 open/high/low/close/volume。
    """
    xtdata = _load_xtdata()
    xt_symbol = _to_xt_symbol(symbol)
    plain_symbol = _to_plain_symbol(symbol)
    dividend_type = _map_adjust_to_dividend_type(adjust)

    last_err: Optional[Exception] = None

    for i in range(max(1, retries)):
        try:
            # 先确保本地有历史数据
            xtdata.download_history_data(
                xt_symbol,
                period="1d",
                start_time=start_date,
                end_time=end_date,
                incrementally=True,
            )

            data = xtdata.get_market_data_ex(
                ["time", "open", "high", "low", "close", "volume", "amount"],
                [xt_symbol],
                period="1d",
                start_time=start_date,
                end_time=end_date,
                dividend_type=dividend_type,
            )

            df = _normalize_xt_kline_df(data.get(xt_symbol))

            core_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(core_cols) < 5:
                raise ValueError("返回数据缺少 OHLCV 核心列")

            out = df[core_cols].copy().astype(float)
            save_data(out, f"{plain_symbol}_{start_date}_{end_date}", data_dir=cache_dir)
            return out

        except Exception as e:
            last_err = e
            if i < max(1, retries) - 1:
                time.sleep(retry_sleep * (i + 1))

    # 缓存回退
    cache_path = os.path.join(cache_dir, f"{plain_symbol}_{start_date}_{end_date}.csv")
    if os.path.exists(cache_path):
        df_cache = load_data(cache_path)
        if not df_cache.empty:
            return df_cache

    raise RuntimeError(f"获取股票 {symbol} 数据失败（xtdata/缓存均失败）: {last_err}")


def fetch_intraday_data(symbol: str, period: str = "1m", count: int = 240) -> pd.DataFrame:
    """
    获取分钟级行情并规范成 trainer_app 需要的结构。

    返回列：dt, price, volume, avg
    """
    xtdata = _load_xtdata()
    xt_symbol = _to_xt_symbol(symbol)

    xtdata.download_history_data(xt_symbol, period=period, incrementally=True)
    try:
        xtdata.subscribe_quote(xt_symbol, period=period, count=-1)
    except Exception:
        # 订阅失败不阻断，继续尝试读取本地/已同步数据
        pass

    data = xtdata.get_market_data_ex(
        ["time", "open", "high", "low", "close", "volume", "amount"],
        [xt_symbol],
        period=period,
        count=count,
        dividend_type="none",
    )

    df = _normalize_xt_kline_df(data.get(xt_symbol))
    out = df.reset_index().rename(columns={"date": "dt", "close": "price"})

    if "volume" not in out.columns:
        out["volume"] = 0.0

    out["dt"] = pd.to_datetime(out["dt"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["dt", "price"]).sort_values("dt").reset_index(drop=True)

    pv = (out["price"] * out["volume"]).cumsum()
    vv = out["volume"].cumsum().replace(0, pd.NA)
    out["avg"] = (pv / vv).fillna(out["price"])

    return out[["dt", "price", "volume", "avg"]]


def list_a_share_symbols(limit: Optional[int] = None) -> List[str]:
    """获取 A 股代码列表（6位代码）。"""
    xtdata = _load_xtdata()
    sectors = set(xtdata.get_sector_list() or [])

    preferred = ["沪深A股", "沪深京A股"]
    fallback = ["上证A股", "深证A股", "京市A股", "科创板", "创业板"]

    raw_codes: List[str] = []

    picked = [s for s in preferred if s in sectors]
    if not picked:
        picked = [s for s in fallback if s in sectors]

    for sec in picked:
        try:
            raw_codes.extend(xtdata.get_stock_list_in_sector(sec) or [])
        except Exception:
            continue

    # 去重 + 规整
    seen = set()
    out = []
    for c in raw_codes:
        p = _to_plain_symbol(c)
        if p and p not in seen:
            seen.add(p)
            out.append(p)

    if limit is not None:
        out = out[: max(0, int(limit))]
    return out


def get_realtime_snapshots(symbols: Iterable[str], chunk_size: int = 800) -> Dict[str, Dict]:
    """批量获取实时快照（key 为 6 位代码）。"""
    xtdata = _load_xtdata()
    xt_symbols = [_to_xt_symbol(s) for s in symbols]

    result: Dict[str, Dict] = {}
    if not xt_symbols:
        return result

    n = len(xt_symbols)
    step = max(1, int(chunk_size))

    for i in range(0, n, step):
        chunk = xt_symbols[i : i + step]
        part = xtdata.get_full_tick(chunk) or {}
        for xt_code, snapshot in part.items():
            result[_to_plain_symbol(xt_code)] = snapshot or {}

    return result


def get_symbol_name(symbol: str) -> str:
    """获取证券简称（失败时返回空串）。"""
    xtdata = _load_xtdata()
    xt_symbol = _to_xt_symbol(symbol)
    try:
        info = xtdata.get_instrument_detail(xt_symbol, iscomplete=True) or {}
        return str(info.get("InstrumentName") or info.get("ExtendName") or "").strip()
    except Exception:
        return ""


def _pick_col_like(df: pd.DataFrame, candidates: List[str]) -> str:
    if df is None or df.empty:
        return ""

    cols = [str(c) for c in df.columns]
    lower_map = {str(c).lower(): str(c) for c in cols}

    for c in candidates:
        if c in cols:
            return c
        lc = c.lower()
        if lc in lower_map:
            return lower_map[lc]

    for c in cols:
        lc = c.lower()
        if any(k in lc for k in ["name", "sector", "block", "板块", "名称"]):
            return c
    return ""


def _is_industry_like_name(name: str) -> bool:
    s = (name or "").strip()
    if len(s) < 2:
        return False

    skip_tokens = [
        "A股", "B股", "ETF", "债券", "基金", "指数", "期权", "期货", "转债",
        "交易所", "市场", "沪深", "上证", "深证", "京市", "连续合约", "能源中心",
        "港股", "中金所", "上期所", "大商所", "郑商所", "科创板CDR",
    ]
    return not any(t in s for t in skip_tokens)


def get_sector_sync_status() -> Dict[str, Any]:
    """
    检测 MiniQMT 终端侧行业板块数据可用性。

    说明：
    - base_sector_count 来自 get_sector_list（通常可用）
    - dynamic_sector_available 依赖 get_sector_info（需要 SectorData）
    """
    xtdata = _load_xtdata()

    status: Dict[str, Any] = {
        "base_sector_count": 0,
        "dynamic_sector_available": False,
        "dynamic_sector_rows": 0,
        "dynamic_industry_count": 0,
        "error": "",
    }

    try:
        base = xtdata.get_sector_list() or []
        status["base_sector_count"] = len(base)
    except Exception as e:
        status["error"] = f"get_sector_list 失败: {e}"
        return status

    try:
        info = xtdata.get_sector_info("")
        if isinstance(info, pd.DataFrame) and not info.empty:
            status["dynamic_sector_available"] = True
            status["dynamic_sector_rows"] = int(len(info))

            name_col = _pick_col_like(
                info,
                ["板块名称", "行业名称", "name", "Name", "SectorName", "BlockName"],
            )
            if name_col:
                names = [str(x).strip() for x in info[name_col].dropna().tolist()]
                industry_names = sorted({n for n in names if _is_industry_like_name(n)})
                status["dynamic_industry_count"] = len(industry_names)
        else:
            status["error"] = "get_sector_info 返回空"
    except Exception as e:
        status["error"] = str(e)

    return status


def list_dynamic_industry_sectors(limit: Optional[int] = None) -> List[str]:
    """
    从 MiniQMT 终端板块库读取行业列表。
    若板块库未就绪（如 SectorData 缺失），返回空列表。
    """
    xtdata = _load_xtdata()
    try:
        info = xtdata.get_sector_info("")
    except Exception:
        return []

    if not isinstance(info, pd.DataFrame) or info.empty:
        return []

    name_col = _pick_col_like(
        info,
        ["板块名称", "行业名称", "name", "Name", "SectorName", "BlockName"],
    )
    if not name_col:
        return []

    names = [str(x).strip() for x in info[name_col].dropna().tolist()]
    out = sorted({n for n in names if _is_industry_like_name(n)})
    if limit is not None:
        out = out[: max(0, int(limit))]
    return out


def list_symbols_in_dynamic_sector(sector_name: str, limit: Optional[int] = None) -> List[str]:
    """按 MiniQMT 动态板块名获取成分股（返回 6 位代码）。"""
    xtdata = _load_xtdata()
    try:
        raw = xtdata.get_stock_list_in_sector(str(sector_name).strip()) or []
    except Exception:
        return []

    seen = set()
    out = []
    for c in raw:
        p = _to_plain_symbol(c)
        if not p or p in seen:
            continue
        # 只保留A/BJ常见6位证券代码
        if not (p.isdigit() and len(p) == 6):
            continue
        seen.add(p)
        out.append(p)

    if limit is not None:
        out = out[: max(0, int(limit))]
    return out


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
