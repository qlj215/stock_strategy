from datetime import datetime, timedelta
import os
import sys
import random
import uuid
import subprocess
import threading
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_strategy.data.fetcher import (
    fetch_stock_data,
    fetch_intraday_data,
    list_a_share_symbols,
    get_realtime_snapshots,
    get_symbol_name,
    get_sector_sync_status,
    list_dynamic_industry_sectors,
    list_symbols_in_dynamic_sector,
)
from stock_strategy.probability_backend import (
    predict_probability,
    predict_probability_batch,
    get_backend_runtime_status,
)

app = Flask(__name__, static_folder="web", static_url_path="")

SYMBOL_POOL = ["000858", "600519", "000001", "600036", "300750", "002594", "600276", "603986"]
# 行业扫描兜底本地池（当 MiniQMT 动态板块不可用时回退）
LOCAL_SECTOR_SYMBOLS = {
    "银行": ["000001", "600036"],
    "白酒": ["600519", "000858"],
    "新能源": ["002594", "300750"],
    "半导体": ["603986", "688981"],
    "医药": ["600276", "300760"],
}
# 动态板块列表缓存，避免频繁触发终端侧 I/O
SECTOR_CACHE = {
    "ts": 0.0,
    "source": "local_fallback",
    "sectors": sorted(LOCAL_SECTOR_SYMBOLS.keys()),
    "status": {},
}
SECTOR_CACHE_TTL_SEC = 300
HISTORY_PANEL_CACHE = {
    "path": "",
    "mtime": 0.0,
    "panel": None,
    "symbols": set(),
    "industries": [],
}
CHALLENGES = {}
REVIEWS = {}
REPLAY_JOBS = {}
SCAN_SORT_LABELS = {
    "today_up": "当日上涨概率",
    "next_5d_up": "5日上涨概率",
    "long_up": "长期上涨概率",
    "pct": "当日涨跌幅",
    "turnover": "成交额",
}
BOARD_FILTER_LABELS = {
    "all": "全部",
    "gem_only": "仅创业板",
    "exclude_gem": "排除创业板",
    "main_only": "仅主板",
    "star_only": "仅科创板",
    "exclude_star": "排除科创板",
    "bj_only": "仅北交所",
    "exclude_bj": "排除北交所",
}
BACKTEST_UNIVERSE_LABELS = {
    "manual": "手动股票池",
    "main_board": "全市场主板",
    "industry": "单一行业",
}
BACKTEST_STRATEGY_LABELS = {
    "rule": "直接规则策略",
    "model": "神经网络 / 机器学习策略",
}
BACKTEST_RULE_PRESET_LABELS = {
    "limit_up_follow": "涨停次日跟随",
    "breakout_20d_high": "20日新高突破",
    "ma5_cross_ma20": "5日线上穿20日线",
    "volume_breakout_20d_high": "放量突破20日新高",
    "oversold_rebound": "超跌反弹",
}
REALTIME_SIGNAL_PRESET_LABELS = dict(BACKTEST_RULE_PRESET_LABELS)
BACKTEST_MODEL_PRESET_LABELS = {
    "topk_prob_1d": "1日上涨概率 Top-K 等权",
}
BACKTEST_MODEL_BACKEND_LABELS = {
    "rule": "rule / 内置规则",
    "dl": "dl / 外部深度学习",
    "auto": "auto / 优先 DL，失败回退 rule",
}
BACKTEST_DATA_SOURCE_LABELS = {
    "panel": "本地历史面板",
    "miniqmt": "MiniQMT",
}
BACKTEST_COST_DEFAULTS = {
    "preset": "huatai_a_share_default",
    "preset_label": "华泰证券A股默认",
    "initial_capital": 1000000.0,
    "buy_commission_rate": 0.0003,
    "sell_commission_rate": 0.0003,
    "min_commission": 5.0,
    "sell_stamp_tax_rate": 0.0005,
    "transfer_fee_rate": 0.00001,
    "slippage_rate": 0.0,
}


def _today_str():
    return datetime.now().strftime("%Y%m%d")


def _normalize_plain_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]
    elif s.startswith(("SH", "SZ", "BJ")) and s[2:].isdigit():
        s = s[2:]

    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.zfill(6) if digits else ""


def _classify_symbol_board(symbol: str) -> str:
    code = _normalize_plain_symbol(symbol)
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("688", "689")):
        return "科创板"
    if code.startswith(("8", "4")):
        return "北交所"
    if code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "主板"
    return "其他"


def _normalize_board_filter(board_filter: str) -> str:
    key = str(board_filter or "").strip()
    return key if key in BOARD_FILTER_LABELS else "all"


def _board_filter_matches(symbol: str, board_filter: str) -> bool:
    board = _classify_symbol_board(symbol)
    if board_filter == "gem_only":
        return board == "创业板"
    if board_filter == "exclude_gem":
        return board != "创业板"
    if board_filter == "main_only":
        return board == "主板"
    if board_filter == "star_only":
        return board == "科创板"
    if board_filter == "exclude_star":
        return board != "科创板"
    if board_filter == "bj_only":
        return board == "北交所"
    if board_filter == "exclude_bj":
        return board != "北交所"
    return True


def _apply_board_filter(symbols: list[str], board_filter: str) -> list[str]:
    if board_filter == "all":
        return list(symbols)
    return [s for s in symbols if _board_filter_matches(s, board_filter)]


def _history_panel_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data", "processed", "daily_panel.parquet")


def _load_history_panel(force_refresh: bool = False) -> pd.DataFrame:
    """
    历史回测统一使用本地日线面板，避免对实时接口做大批量回看。
    """
    path = _history_panel_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"历史面板不存在：{path}")

    mtime = os.path.getmtime(path)
    cached = HISTORY_PANEL_CACHE.get("panel")
    if (not force_refresh) and cached is not None and float(HISTORY_PANEL_CACHE.get("mtime", 0.0)) == float(mtime):
        return cached

    cols = ["date", "stock_code", "open", "high", "low", "close", "volume", "amount", "is_trading", "industry"]
    panel = pd.read_parquet(path, columns=cols)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["stock_code"] = panel["stock_code"].astype(str).map(_normalize_plain_symbol)

    for col in ["open", "high", "low", "close", "volume", "amount", "is_trading"]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    if "industry" not in panel.columns:
        panel["industry"] = ""
    panel["industry"] = panel["industry"].fillna("").astype(str).str.strip()
    panel = panel.dropna(subset=["date"]).sort_values(["stock_code", "date"]).reset_index(drop=True)

    symbols = sorted({s for s in panel["stock_code"].tolist() if s})
    industries = sorted({s for s in panel["industry"].tolist() if s})
    HISTORY_PANEL_CACHE.update({
        "path": path,
        "mtime": mtime,
        "panel": panel,
        "symbols": set(symbols),
        "industries": industries,
    })
    return panel


def _panel_symbols() -> set[str]:
    _load_history_panel(force_refresh=False)
    return set(HISTORY_PANEL_CACHE.get("symbols", set()))


def _panel_industries() -> list[str]:
    _load_history_panel(force_refresh=False)
    return list(HISTORY_PANEL_CACHE.get("industries", []))


def _get_sector_registry(force_refresh: bool = False) -> Dict:
    """
    获取行业列表注册表：优先 MiniQMT 动态板块，失败时回退本地池。
    """
    now_ts = datetime.now().timestamp()
    if (not force_refresh) and SECTOR_CACHE.get("sectors") and (now_ts - float(SECTOR_CACHE.get("ts", 0))) < SECTOR_CACHE_TTL_SEC:
        return {
            "source": SECTOR_CACHE.get("source", "local_fallback"),
            "sectors": list(SECTOR_CACHE.get("sectors", [])),
            "status": dict(SECTOR_CACHE.get("status", {})),
        }

    status = get_sector_sync_status()
    sectors = []
    source = "local_fallback"

    if status.get("dynamic_sector_available"):
        sectors = list_dynamic_industry_sectors(limit=1000)
        if sectors:
            source = "miniqmt_dynamic"

    if not sectors:
        try:
            sectors = _panel_industries()
            if sectors:
                source = "history_panel"
        except Exception:
            sectors = []

    if not sectors:
        sectors = sorted(LOCAL_SECTOR_SYMBOLS.keys())

    SECTOR_CACHE.update({
        "ts": now_ts,
        "source": source,
        "sectors": sectors,
        "status": status,
    })

    return {
        "source": source,
        "sectors": list(sectors),
        "status": status,
    }


def _trend_label(df: pd.DataFrame, i: int) -> str:
    ma20 = df["close"].rolling(20).mean().iloc[i]
    ma60 = df["close"].rolling(60).mean().iloc[i]
    if pd.isna(ma20) or pd.isna(ma60):
        return "震荡"
    gap = (ma20 - ma60) / ma60
    if gap > 0.02:
        return "上涨"
    if gap < -0.02:
        return "下跌"
    return "震荡"


def _difficulty_window(level: str):
    # 返回历史窗口长度，和预测天数
    if level == "easy":
        return 80, 5
    if level == "hard":
        return 40, 10
    return 60, 5  # normal


def _daily_from_fetcher(symbol: str, start: str, end: str) -> pd.DataFrame:
    """统一把 fetch_stock_data 返回的 index 结构转成 trainer 需要的列结构。"""
    idx_df = fetch_stock_data(symbol, start, end, adjust="qfq", retries=2)
    if idx_df is None or idx_df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "turnover", "pct"])

    df = idx_df.copy().reset_index()
    if "date" not in df.columns:
        # 兼容极少数 index 名称不是 date 的情况
        df = df.rename(columns={df.columns[0]: "date"})

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if "turnover" not in df.columns:
        # fetch_stock_data 默认返回不含成交额，先置 0
        df["turnover"] = 0.0
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce").fillna(0.0)

    df["pct"] = df["close"].pct_change() * 100
    return df[["date", "open", "high", "low", "close", "volume", "turnover", "pct"]]


def _intraday_from_fetcher(symbol: str, count: int = 240) -> pd.DataFrame:
    try:
        out = fetch_intraday_data(symbol, period="1m", count=count)
        if out is None or out.empty:
            return pd.DataFrame(columns=["dt", "price", "volume", "avg"])
        return out[["dt", "price", "volume", "avg"]].copy()
    except Exception:
        return pd.DataFrame(columns=["dt", "price", "volume", "avg"])


def _intraday_completion_ratio(intraday_df: pd.DataFrame) -> float:
    if intraday_df is None or intraday_df.empty or "dt" not in intraday_df.columns:
        return 1.0

    times = pd.to_datetime(intraday_df["dt"], errors="coerce").dropna()
    if times.empty:
        return 1.0

    latest = pd.Timestamp(times.max())
    minute_of_day = latest.hour * 60 + latest.minute

    if minute_of_day <= 9 * 60 + 30:
        traded_minutes = 1
    elif minute_of_day < 11 * 60 + 30:
        traded_minutes = minute_of_day - (9 * 60 + 30) + 1
    elif minute_of_day < 13 * 60:
        traded_minutes = 120
    elif minute_of_day <= 15 * 60:
        traded_minutes = 120 + (minute_of_day - 13 * 60) + 1
    else:
        traded_minutes = 240

    traded_minutes = max(1, min(int(traded_minutes), 240))
    return float(traded_minutes / 240.0)


def _merge_today_estimated_bar(daily_df: pd.DataFrame, intraday_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    daily = daily_df.copy()
    meta = {
        "applied": False,
        "date": "",
        "completion_ratio": 1.0,
        "estimated_volume": None,
        "estimated_turnover": None,
        "raw_volume": None,
        "raw_turnover": None,
        "latest_price": None,
    }

    if intraday_df is None or intraday_df.empty or daily.empty:
        return daily, meta

    intra = intraday_df.copy()
    intra["dt"] = pd.to_datetime(intra["dt"], errors="coerce")
    intra = intra.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    if intra.empty:
        return daily, meta

    trade_date = pd.Timestamp(intra["dt"].iloc[-1]).normalize()
    today_key = trade_date.strftime("%Y-%m-%d")
    if str(daily.iloc[-1]["date"].strftime("%Y-%m-%d")) != today_key:
        return daily, meta

    ratio = _intraday_completion_ratio(intra)
    if ratio <= 0:
        ratio = 1.0

    raw_volume = float(pd.to_numeric(intra.get("volume"), errors="coerce").fillna(0.0).sum())
    prices = pd.to_numeric(intra.get("price"), errors="coerce")
    vols = pd.to_numeric(intra.get("volume"), errors="coerce").fillna(0.0)
    raw_turnover = float((prices.fillna(method="ffill").fillna(0.0) * vols).sum())
    latest_price = float(prices.dropna().iloc[-1]) if prices.dropna().shape[0] else None
    day_high = float(prices.max()) if prices.notna().any() else None
    day_low = float(prices.min()) if prices.notna().any() else None
    day_open = float(prices.dropna().iloc[0]) if prices.dropna().shape[0] else None

    est_volume = raw_volume / ratio if ratio > 0 else raw_volume
    est_turnover = raw_turnover / ratio if ratio > 0 else raw_turnover

    idx = daily.index[-1]
    if day_open and day_open > 0:
        daily.at[idx, "open"] = day_open
    if day_high and day_high > 0:
        daily.at[idx, "high"] = day_high
    if day_low and day_low > 0:
        daily.at[idx, "low"] = day_low
    if latest_price and latest_price > 0:
        daily.at[idx, "close"] = latest_price
    daily.at[idx, "volume"] = est_volume
    if "turnover" in daily.columns:
        daily.at[idx, "turnover"] = est_turnover
    daily["pct"] = daily["close"].pct_change() * 100

    meta.update({
        "applied": True,
        "date": today_key,
        "completion_ratio": float(ratio),
        "estimated_volume": float(est_volume),
        "estimated_turnover": float(est_turnover),
        "raw_volume": float(raw_volume),
        "raw_turnover": float(raw_turnover),
        "latest_price": latest_price,
    })
    return daily, meta


def _fetch_event_context(symbol: str, anchor_date: str, max_items: int = 5, lookback_days: int = 45):
    """
    MiniQMT/xtdata 当前不直接提供 AKShare 风格新闻流接口。
    为避免引入公网抓取，本函数在迁移版中返回空事件列表。
    """
    return []


def _build_codex_prompt(item: dict) -> str:
    # 控制上下文长度，降低大模型超时概率
    candles = item["candles"][-20:]
    ohlc_lines = "\n".join([
        f"{c['date']}, O:{c['open']:.2f}, H:{c['high']:.2f}, L:{c['low']:.2f}, C:{c['close']:.2f}"
        for c in candles
    ])
    event_lines = "\n".join([
        f"- {e['time']} | {e['title']}（{e['source']}）" for e in item.get("events", [])
    ]) or "- 无可用事件数据"

    return f"""你是一名严谨的A股技术分析教练。请基于以下历史K线片段分析该样本在截面日后的走势成因。

样本信息：
- 股票: {item['symbol']}
- 截面日期: {item['anchor_date']}
- 用户判断方向: {item['pred_direction']}
- 用户判断趋势: {item['pred_trend']}
- 实际5日涨跌: {item['ret5_pct']}%
- 实际方向: {item['truth_direction']}
- 实际趋势: {item['truth_trend']}

事件上下文（用于解释可能的基本面/政策扰动）：
{event_lines}

K线数据（最近20根）：
{ohlc_lines}

请输出（严格控制简洁）：
1) 方向判断关键证据（最多3条）
2) 趋势判断关键证据（最多3条）
3) 用户偏差与盲点（最多2条）
4) 下次可执行改进（3条）
5) 30字内总结

要求：中文、结构化、简洁、可执行。
"""


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/challenge")
def challenge():
    symbol = request.args.get("symbol") or random.choice(SYMBOL_POOL)
    level = request.args.get("level", "normal")
    hist_len, pred_days = _difficulty_window(level)

    end = _today_str()
    df = fetch_stock_data(symbol, "20190101", end, retries=2)

    if len(df) < hist_len + pred_days + 20:
        return jsonify({"error": "样本不足"}), 400

    require_events = request.args.get("require_events", "1") == "1"

    selected = None
    max_try = 8
    low_idx = hist_len + 20
    high_idx = len(df) - pred_days - 1

    # MiniQMT 迁移后不依赖公网新闻接口，直接走K线随机抽题

    # 回退：常规随机抽题
    for _ in range(max_try):
        if selected is not None:
            break
        i_try = random.randint(low_idx, high_idx)
        anchor_try = str(df.index[i_try].date())
        events_try = _fetch_event_context(symbol, anchor_date=anchor_try, max_items=5)
        if (not require_events) or events_try:
            selected = (i_try, anchor_try, events_try)
            break
        selected = (i_try, anchor_try, events_try)

    i, anchor_date, events = selected
    hist = df.iloc[i - hist_len:i + 1].copy()

    current_close = float(df["close"].iloc[i])
    future_close = float(df["close"].iloc[i + pred_days])
    ret = (future_close / current_close) - 1

    truth_direction = "上涨" if ret >= 0 else "下跌"
    truth_trend = _trend_label(df, i)

    cid = str(uuid.uuid4())
    CHALLENGES[cid] = {
        "symbol": symbol,
        "anchor_date": anchor_date,
        "ret": ret,
        "truth_direction": truth_direction,
        "truth_trend": truth_trend,
        "pred_days": pred_days,
        "candles": [
            {
                "date": str(idx.date()),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
            }
            for idx, r in hist.iterrows()
        ],
        "events": events,
    }

    return jsonify({
        "id": cid,
        "symbol": symbol,
        "anchor_date": anchor_date,
        "pred_days": pred_days,
        "level": level,
        "candles": CHALLENGES[cid]["candles"],
        "events": CHALLENGES[cid]["events"],
        "events_mode": "period_matched" if CHALLENGES[cid]["events"] else "not_available",
        "prompt": f"请根据K线判断：未来{pred_days}个交易日更可能上涨还是下跌？当前走势属于上涨/下跌/震荡？",
    })


@app.route("/api/answer", methods=["POST"])
def answer():
    data = request.get_json(force=True)
    cid = data.get("id")
    pred_direction = data.get("pred_direction")
    pred_trend = data.get("pred_trend")

    if cid not in CHALLENGES:
        return jsonify({"error": "题目不存在或已失效"}), 400

    item = CHALLENGES.pop(cid)

    score = 0
    details = []

    if pred_direction == item["truth_direction"]:
        score += 60
        details.append("方向判断正确 +60")
    else:
        details.append("方向判断错误 +0")

    if pred_trend == item["truth_trend"]:
        score += 40
        details.append("趋势判断正确 +40")
    else:
        details.append("趋势判断错误 +0")

    level = "优秀" if score >= 80 else "合格" if score >= 60 else "继续训练"

    rid = str(uuid.uuid4())
    REVIEWS[rid] = {
        "symbol": item["symbol"],
        "anchor_date": item["anchor_date"],
        "ret5_pct": round(item["ret"] * 100, 2),
        "truth_direction": item["truth_direction"],
        "truth_trend": item["truth_trend"],
        "pred_direction": pred_direction,
        "pred_trend": pred_trend,
        "candles": item["candles"],
        "events": item.get("events", []),
    }

    return jsonify({
        "score": score,
        "level": level,
        "review_id": rid,
        "truth": {
            "direction": item["truth_direction"],
            "trend": item["truth_trend"],
            "ret5_pct": round(item["ret"] * 100, 2),
            "anchor_date": item["anchor_date"],
            "symbol": item["symbol"],
        },
        "details": details,
        "coach": "重点看：关键均线位置、放量突破/跌破、连续K线结构，而不是只看单日涨跌。",
    })


def _run_replay_job(job_id: str, review_id: str):
    item = REVIEWS[review_id]
    prompt = _build_codex_prompt(item)
    REPLAY_JOBS[job_id]["status"] = "running"
    REPLAY_JOBS[job_id]["started_at"] = datetime.now().isoformat()

    try:
        t0 = datetime.now()
        proc = subprocess.run(
            ["codex", "exec", prompt],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        elapsed = (datetime.now() - t0).total_seconds()
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()

        if proc.returncode != 0:
            REPLAY_JOBS[job_id].update({
                "status": "error",
                "error": "Codex 调用失败",
                "detail": err[:500] if err else "unknown",
                "elapsed_sec": round(elapsed, 1),
            })
            return

        REPLAY_JOBS[job_id].update({
            "status": "done",
            "analysis": out or "Codex未返回内容",
            "elapsed_sec": round(elapsed, 1),
        })
    except FileNotFoundError:
        REPLAY_JOBS[job_id].update({"status": "error", "error": "未检测到 codex 命令，请先安装并配置。"})
    except subprocess.TimeoutExpired:
        REPLAY_JOBS[job_id].update({"status": "error", "error": "Codex 分析超时，请重试。"})


@app.route("/api/replay/codex", methods=["POST"])
def replay_codex():
    data = request.get_json(force=True)
    rid = data.get("review_id")
    if rid not in REVIEWS:
        return jsonify({"error": "复盘记录不存在"}), 400

    job_id = str(uuid.uuid4())
    REPLAY_JOBS[job_id] = {
        "status": "queued",
        "review_id": rid,
        "created_at": datetime.now().isoformat(),
    }

    t = threading.Thread(target=_run_replay_job, args=(job_id, rid), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/replay/codex/<job_id>", methods=["GET"])
def replay_codex_status(job_id):
    if job_id not in REPLAY_JOBS:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(REPLAY_JOBS[job_id])


def _probability_model(daily_df: pd.DataFrame, backend: str | None = None) -> Dict:
    """统一概率预测入口：支持 rule / dl / auto 后端切换。"""
    return predict_probability(daily_df, backend=backend, allow_fallback=True)


def _build_market_codex_prompt(symbol: str, daily_df: pd.DataFrame, intraday_df: pd.DataFrame, model_result: Dict) -> str:
    d = daily_df.tail(30)
    i = intraday_df.tail(30)

    daily_lines = "\n".join([
        f"{r['date'].strftime('%Y-%m-%d')} C:{r['close']:.2f} V:{r['volume']:.0f} Pct:{r['pct']:.2f}%"
        for _, r in d.iterrows()
    ])
    intra_lines = "\n".join([
        f"{r['dt'].strftime('%H:%M')} P:{r['price']:.2f} V:{r['volume']:.0f}"
        for _, r in i.iterrows()
    ]) or "无分时数据"

    return f"""你是A股交易研究员。请结合以下量价数据，对 {symbol} 给出当日、5日、长期（3-6月）上涨概率的分析理由。

模型先验概率：
- 当日上涨概率: {model_result['p_up_today']:.2%}
- 5日上涨概率: {model_result['p_up_5d']:.2%}
- 长期上涨概率: {model_result['p_up_long']:.2%}

特征：{model_result.get('features', {})}

最近日线（30个交易日）：
{daily_lines}

最近分时（30个时点）：
{intra_lines}

请输出：
1) 三个周期上涨概率是否需要上调/下调（每项一句）
2) 关键依据（技术面、量能、可能的政策/行业事件、主力行为）
3) 风险点（2-3条）
4) 一段100字内结论

要求：中文、简洁、可执行，不要编造具体未给出的新闻标题。"""


def _run_market_codex_reason(symbol: str, daily_df: pd.DataFrame, intraday_df: pd.DataFrame, model_result: Dict) -> Tuple[str, str]:
    prompt = _build_market_codex_prompt(symbol, daily_df, intraday_df, model_result)
    try:
        proc = subprocess.run(
            ["codex", "exec", prompt],
            capture_output=True,
            text=True,
            timeout=360,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if proc.returncode != 0:
            return "", (proc.stderr or "Codex调用失败")[:600]
        return (proc.stdout or "").strip(), ""
    except FileNotFoundError:
        return "", "未检测到 codex 命令，请先安装。"
    except subprocess.TimeoutExpired:
        return "", "Codex 分析超时。"


@app.route("/market")
def market_page():
    return send_from_directory("web", "market.html")


@app.route("/api/market/model_backend_status")
def market_model_backend_status():
    backend = (request.args.get("backend") or "").strip() or None
    return jsonify(get_backend_runtime_status(backend))


@app.route("/api/market/overview")
def market_overview():
    symbol = (request.args.get("symbol") or "000001").strip()
    days = int(request.args.get("days", "60"))
    days = max(20, min(days, 250))
    backend = (request.args.get("backend") or "").strip() or None

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days * 2 + 20)).strftime("%Y%m%d")

    try:
        daily = _daily_from_fetcher(symbol, start, end).tail(days)
        if daily.empty:
            return jsonify({"error": "未获取到日线数据"}), 400
        daily["symbol"] = symbol

        intraday = _intraday_from_fetcher(symbol, count=240)
        daily, intraday_estimation = _merge_today_estimated_bar(daily, intraday)

        model_result = _probability_model(daily, backend=backend)
        latest = daily.iloc[-1]

        return jsonify({
            "symbol": symbol,
            "latest": {
                "date": latest["date"].strftime("%Y-%m-%d"),
                "close": round(float(latest["close"]), 3),
                "pct": round(float(latest["pct"]), 3),
                "volume": float(latest["volume"]),
                "turnover": float(latest["turnover"]),
            },
            "daily": [
                {
                    "date": r["date"].strftime("%Y-%m-%d"),
                    "open": round(float(r["open"]), 3) if pd.notna(r["open"]) else None,
                    "high": round(float(r["high"]), 3) if pd.notna(r["high"]) else None,
                    "low": round(float(r["low"]), 3) if pd.notna(r["low"]) else None,
                    "close": round(float(r["close"]), 3),
                    "volume": float(r["volume"]),
                    "pct": round(float(r["pct"]), 3) if pd.notna(r["pct"]) else None,
                }
                for _, r in daily.iterrows()
            ],
            "intraday": [
                {
                    "dt": r["dt"].strftime("%Y-%m-%d %H:%M"),
                    "price": round(float(r["price"]), 3),
                    "volume": float(r["volume"]),
                    "avg": round(float(r["avg"]), 3),
                }
                for _, r in intraday.tail(240).iterrows()
            ],
            "prediction": {
                "today_up": round(model_result["p_up_today"], 4),
                "next_5d_up": round(model_result["p_up_5d"], 4),
                "long_up": round(model_result["p_up_long"], 4),
                "reasons": model_result["reasons"],
                "features": model_result.get("features", {}),
                "backend": model_result.get("backend", "rule"),
                "backend_requested": model_result.get("backend_requested", backend or "rule"),
                "backend_fallback": bool(model_result.get("backend_fallback", False)),
                "backend_error": model_result.get("backend_error", ""),
            },
            "intraday_estimation": intraday_estimation,
            "signal_presets": [
                {"value": key, "label": label}
                for key, label in REALTIME_SIGNAL_PRESET_LABELS.items()
            ],
        })
    except Exception as e:
        return jsonify({"error": f"数据获取失败: {str(e)}"}), 500


@app.route("/api/market/signals")
def market_signals():
    symbol = (request.args.get("symbol") or "000001").strip()
    days = int(request.args.get("days", "60"))
    days = max(20, min(days, 250))
    signal_preset = (request.args.get("signal_preset") or "limit_up_follow").strip()

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days * 2 + 40)).strftime("%Y%m%d")

    daily = _daily_from_fetcher(symbol, start, end).tail(days)
    if daily.empty:
        return jsonify({"error": "未获取到日线数据"}), 400
    daily["symbol"] = symbol

    intraday = _intraday_from_fetcher(symbol, count=240)
    daily, intraday_estimation = _merge_today_estimated_bar(daily, intraday)

    try:
        signal_result = _generate_realtime_signal_marks(daily, signal_preset)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "symbol": symbol,
        "days": days,
        "signal": signal_result,
        "intraday_estimation": intraday_estimation,
        "available_presets": [
            {"value": key, "label": label}
            for key, label in REALTIME_SIGNAL_PRESET_LABELS.items()
        ],
    })


@app.route("/api/market/codex_reason", methods=["POST"])
def market_codex_reason():
    data = request.get_json(force=True)
    symbol = (data.get("symbol") or "000001").strip()
    days = int(data.get("days", 60))
    days = max(20, min(days, 250))
    backend = (data.get("backend") or "").strip() or None

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days * 2 + 20)).strftime("%Y%m%d")

    daily = _daily_from_fetcher(symbol, start, end).tail(days)
    if daily.empty:
        return jsonify({"error": "未获取到日线数据"}), 400
    daily["symbol"] = symbol

    intraday = _intraday_from_fetcher(symbol, count=240)
    daily, _ = _merge_today_estimated_bar(daily, intraday)
    model_result = _probability_model(daily, backend=backend)

    analysis, err = _run_market_codex_reason(symbol, daily, intraday, model_result)
    if err:
        return jsonify({"error": err}), 500

    return jsonify({
        "symbol": symbol,
        "analysis": analysis,
        "backend": model_result.get("backend", "rule"),
        "backend_requested": model_result.get("backend_requested", backend or "rule"),
        "backend_fallback": bool(model_result.get("backend_fallback", False)),
    })


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _calc_symbol_snapshot(symbol: str, days: int = 60, backend: str | None = None) -> Dict:
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days * 2 + 20)).strftime("%Y%m%d")
    daily = _daily_from_fetcher(symbol, start, end).tail(days)
    if daily.empty:
        return {}

    pred = _probability_model(daily, backend=backend)
    latest = daily.iloc[-1]

    return {
        "symbol": symbol,
        "close": round(_to_float(latest.get("close")), 3),
        "pct": round(_to_float(latest.get("pct")), 3),
        "volume": _to_float(latest.get("volume")),
        "turnover": _to_float(latest.get("turnover")),
        "today_up": round(pred["p_up_today"], 4),
        "next_5d_up": round(pred["p_up_5d"], 4),
        "long_up": round(pred["p_up_long"], 4),
        "model_backend": pred.get("backend", "rule"),
        "model_backend_fallback": bool(pred.get("backend_fallback", False)),
    }


@app.route("/api/market/sector_sync_status")
def market_sector_sync_status():
    force = request.args.get("force", "0") == "1"
    reg = _get_sector_registry(force_refresh=force)
    return jsonify({
        "source": reg["source"],
        "sector_count": len(reg["sectors"]),
        "status": reg["status"],
    })


@app.route("/api/market/sectors")
def market_sectors():
    force = request.args.get("force", "0") == "1"
    reg = _get_sector_registry(force_refresh=force)
    return jsonify({
        "sectors": reg["sectors"],
        "source": reg["source"],
        "sync_status": reg["status"],
    })


@app.route("/api/market/scan")
def market_scan():
    mode = (request.args.get("mode") or "industry").strip()  # industry | all
    industry = (request.args.get("industry") or "").strip()
    sort_by = (request.args.get("sort_by") or "next_5d_up").strip()
    board_filter = _normalize_board_filter(request.args.get("board_filter"))
    days = max(20, min(int(request.args.get("days", "60")), 250))
    limit = max(1, min(int(request.args.get("limit", "30")), 400))
    backend = (request.args.get("backend") or "").strip() or None

    allowed_sort = {"today_up", "next_5d_up", "long_up", "pct", "turnover"}
    if sort_by not in allowed_sort:
        sort_by = "next_5d_up"

    try:
        symbols = []
        names_map = {}
        snapshots = {}

        sector_source = "all_mode"
        filtered_universe = 0

        if mode == "industry":
            if not industry:
                return jsonify({"error": "industry 模式需要 industry 参数"}), 400

            reg = _get_sector_registry(force_refresh=False)
            available_sectors = set(reg.get("sectors", []))

            if reg.get("source") == "miniqmt_dynamic" and industry in available_sectors:
                symbols = list_symbols_in_dynamic_sector(industry, limit=None)
                sector_source = "miniqmt_dynamic"
            else:
                symbols = [str(x).zfill(6) for x in LOCAL_SECTOR_SYMBOLS.get(industry, [])]
                sector_source = "local_fallback"

            if not symbols:
                # 动态源拿不到时，再兜底一次本地池
                if industry in LOCAL_SECTOR_SYMBOLS:
                    symbols = [str(x).zfill(6) for x in LOCAL_SECTOR_SYMBOLS[industry]]
                    sector_source = "local_fallback"
                else:
                    return jsonify({
                        "error": f"不支持的行业：{industry}",
                        "available": sorted(list(available_sectors)) if available_sectors else sorted(LOCAL_SECTOR_SYMBOLS.keys()),
                    }), 400

            symbols = _apply_board_filter(symbols, board_filter)
            filtered_universe = len(symbols)
            symbols = symbols[:limit]
            names_map = {s: get_symbol_name(s) for s in symbols}

        elif mode == "all":
            sector_source = "miniqmt_realtime"
            all_symbols = list_a_share_symbols()
            if not all_symbols:
                return jsonify({"error": "无法获取全A股列表，请确认 MiniQMT 已连接且行情可用。"}), 500

            all_symbols = _apply_board_filter(all_symbols, board_filter)
            filtered_universe = len(all_symbols)
            if not all_symbols:
                return jsonify({
                    "mode": mode,
                    "mode_label": "全A股遍历（按成交额优先）",
                    "industry": "全A股",
                    "sector_source": sector_source,
                    "board_filter": board_filter,
                    "board_filter_label": BOARD_FILTER_LABELS[board_filter],
                    "sort_by": sort_by,
                    "sort_by_label": SCAN_SORT_LABELS.get(sort_by, sort_by),
                    "days": days,
                    "requested": 0,
                    "success": 0,
                    "failed": 0,
                    "filtered_universe": filtered_universe,
                    "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_backend": get_backend_runtime_status(backend),
                    "top": [],
                })

            snapshots = get_realtime_snapshots(all_symbols, chunk_size=800)
            if not snapshots:
                return jsonify({"error": "无法获取全市场实时快照，请稍后重试。"}), 500

            ranked = sorted(
                snapshots.items(),
                key=lambda kv: _to_float((kv[1] or {}).get("amount"), 0.0),
                reverse=True,
            )
            symbols = [code for code, _ in ranked[:limit]]
            names_map = {s: get_symbol_name(s) for s in symbols}
        else:
            return jsonify({"error": "mode 仅支持 industry 或 all"}), 400

        rows = []
        failed = []
        for s in symbols:
            try:
                item = _calc_symbol_snapshot(s, days=days, backend=backend)
                if not item:
                    failed.append(s)
                    continue

                # all 模式下优先使用实时快照的最新价/成交额
                if mode == "all":
                    snap = snapshots.get(s, {})
                    if snap:
                        last_price = _to_float(snap.get("lastPrice"), 0.0)
                        amount = _to_float(snap.get("amount"), 0.0)
                        if last_price > 0:
                            item["close"] = round(last_price, 3)
                        if amount > 0:
                            item["turnover"] = amount

                if names_map.get(s):
                    item["name"] = names_map[s]
                item["board"] = _classify_symbol_board(s)
                rows.append(item)
            except Exception:
                failed.append(s)

        rows = sorted(rows, key=lambda x: x.get(sort_by, 0), reverse=True)

        return jsonify({
            "mode": mode,
            "mode_label": "行业内对比" if mode == "industry" else "全A股遍历（按成交额优先）",
            "industry": industry if mode == "industry" else "全A股",
            "sector_source": sector_source,
            "model_backend": get_backend_runtime_status(backend),
            "days": days,
            "sort_by": sort_by,
            "sort_by_label": SCAN_SORT_LABELS.get(sort_by, sort_by),
            "board_filter": board_filter,
            "board_filter_label": BOARD_FILTER_LABELS[board_filter],
            "filtered_universe": filtered_universe or len(symbols),
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "requested": len(symbols),
            "success": len(rows),
            "failed": len(failed),
            "top": rows,
        })
    except Exception as e:
        return jsonify({"error": f"批量扫描失败: {str(e)}"}), 500


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, len(y_true))

    p = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y_true) ** 2))
    logloss = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    return {
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "brier": round(brier, 6),
        "logloss": round(logloss, 6),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    items = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < bins - 1 else y_prob <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        items.append({
            "bin": f"[{lo:.1f},{hi:.1f}{')' if i < bins - 1 else ']'}",
            "count": cnt,
            "avg_prob": round(float(y_prob[mask].mean()), 4),
            "real_up_rate": round(float(y_true[mask].mean()), 4),
        })
    return items


def _strategy_metrics(daily: pd.DataFrame, prob_series: pd.Series, horizon: int, threshold: float = 0.5) -> Dict:
    """
    非重叠交易回测（避免把 horizon 期收益按“每天”重复复利导致夸大）。
    规则：当 p>=threshold 时，以当日收盘买入并持有 horizon 个交易日，到期卖出。
    持仓期间不重复开仓（非重叠）。
    """
    if daily is None or daily.empty or "close" not in daily.columns:
        return {"strategy_total_return": 0.0, "buyhold_total_return": 0.0, "strategy_sharpe": 0.0, "max_drawdown": 0.0, "trade_count": 0, "hit_rate": 0.0, "points": []}

    close = daily["close"].astype(float)
    p = prob_series.reindex(daily.index).fillna(0.0).astype(float)

    n = len(daily)
    if n <= horizon + 1:
        return {"strategy_total_return": 0.0, "buyhold_total_return": 0.0, "strategy_sharpe": 0.0, "max_drawdown": 0.0, "trade_count": 0, "hit_rate": 0.0, "points": []}

    equity = 1.0
    trade_returns = []
    points = []

    i = 0
    while i + horizon < n:
        if p.iloc[i] >= threshold:
            entry = float(close.iloc[i])
            exit_ = float(close.iloc[i + horizon])
            r = (exit_ / entry) - 1.0
            trade_returns.append(r)
            equity *= (1.0 + r)

            dt = daily.index[i + horizon]
            bh = float(close.iloc[i + horizon] / close.iloc[0])
            points.append({
                "date": dt.strftime("%Y-%m-%d"),
                "strategy": round(float(equity), 6),
                "buyhold": round(float(bh), 6),
            })
            i += horizon
        else:
            i += 1

    if not trade_returns:
        return {
            "strategy_total_return": 0.0,
            "buyhold_total_return": round(float(close.iloc[-1] / close.iloc[0] - 1), 4),
            "strategy_sharpe": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "hit_rate": 0.0,
            "points": [],
        }

    trade_returns = np.array(trade_returns, dtype=float)
    strat_curve = np.cumprod(1 + trade_returns)
    running_max = np.maximum.accumulate(strat_curve)
    mdd = float(np.min(strat_curve / running_max - 1)) if len(strat_curve) else 0.0

    mean_r = float(np.mean(trade_returns))
    std_r = float(np.std(trade_returns))
    sharpe = (mean_r / std_r * np.sqrt(252 / max(horizon, 1))) if std_r > 1e-12 else 0.0

    first_idx = daily.index[0]
    last_idx = daily.index[min(n - 1, i)] if i < n else daily.index[-1]
    bh_return = float(close.loc[last_idx] / close.loc[first_idx] - 1)

    return {
        "strategy_total_return": round(float(equity - 1), 4),
        "buyhold_total_return": round(bh_return, 4),
        "strategy_sharpe": round(float(sharpe), 4),
        "max_drawdown": round(mdd, 4),
        "trade_count": int(len(trade_returns)),
        "hit_rate": round(float((trade_returns > 0).mean()), 4),
        "points": points,
    }


# 历史回测统一层：股票池解析 -> 信号生成 -> 非重叠批量执行 -> JSON 序列化
def _format_bt_date(dt: Any) -> str:
    return pd.Timestamp(dt).strftime("%Y-%m-%d")


def _parse_bt_date(text: str, field_name: str) -> pd.Timestamp:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError(f"{field_name} 不能为空")

    dt = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(raw, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"{field_name} 格式错误，应为 YYYYMMDD 或 YYYY-MM-DD")
    return pd.Timestamp(dt).normalize()


def _parse_manual_symbols(text: str) -> Tuple[List[str], List[str]]:
    tokens = [tok for tok in re.split(r"[\s,，;；、]+", str(text or "").strip()) if tok]
    out: List[str] = []
    invalid: List[str] = []
    seen = set()
    for token in tokens:
        code = _normalize_plain_symbol(token)
        if not code:
            invalid.append(token)
            continue
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out, invalid


def _parse_backtest_cost_config(args) -> Dict[str, float]:
    def _clamp_arg(name: str, default: float, min_value: float, max_value: float) -> float:
        raw = args.get(name) if args is not None else None
        value = _to_float(raw if raw not in {None, ""} else default, default)
        if not np.isfinite(value):
            value = default
        return float(max(min_value, min(value, max_value)))

    defaults = dict(BACKTEST_COST_DEFAULTS)
    cfg = {
        "initial_capital": _clamp_arg("initial_capital", defaults["initial_capital"], 10000.0, 1e10),
        "buy_commission_rate": _clamp_arg("buy_commission_rate", defaults["buy_commission_rate"], 0.0, 0.05),
        "sell_commission_rate": _clamp_arg("sell_commission_rate", defaults["sell_commission_rate"], 0.0, 0.05),
        "min_commission": _clamp_arg("min_commission", defaults["min_commission"], 0.0, 1000.0),
        "sell_stamp_tax_rate": _clamp_arg("sell_stamp_tax_rate", defaults["sell_stamp_tax_rate"], 0.0, 0.05),
        "transfer_fee_rate": _clamp_arg("transfer_fee_rate", defaults["transfer_fee_rate"], 0.0, 0.01),
        "slippage_rate": _clamp_arg("slippage_rate", defaults["slippage_rate"], 0.0, 0.01),
    }
    is_default = all(abs(float(cfg[key]) - float(defaults[key])) < 1e-12 for key in cfg.keys())
    cfg["preset"] = str(defaults["preset"] if is_default else "custom")
    cfg["preset_label"] = str(defaults["preset_label"] if is_default else "自定义成本参数")
    return cfg


def _symbol_limit_up_rate(symbol: str) -> float:
    board = _classify_symbol_board(symbol)
    if board in {"创业板", "科创板"}:
        return 0.20
    if board == "北交所":
        return 0.30
    return 0.10


def _is_limit_up_close(symbol: str, prev_close: Any, close: Any) -> bool:
    prev_v = _to_float(prev_close, 0.0)
    close_v = _to_float(close, 0.0)
    if prev_v <= 0 or close_v <= 0:
        return False
    limit_rate = _symbol_limit_up_rate(symbol)
    return (close_v / prev_v - 1.0) >= (limit_rate - 0.0015)


def _row_is_tradeable(row: pd.Series) -> bool:
    return bool(int(_to_float(row.get("is_trading"), 0.0)) == 1 and pd.notna(row.get("close")) and _to_float(row.get("close"), 0.0) > 0)


def _resolve_backtest_universe(universe_mode: str, symbols_text: str, industry: str, data_source: str) -> Dict[str, Any]:
    panel_symbols = _panel_symbols()
    panel_industries = set(_panel_industries())
    warnings: List[str] = []
    use_panel = data_source == "panel"

    if universe_mode == "manual":
        manual_symbols, invalid_tokens = _parse_manual_symbols(symbols_text)
        if not manual_symbols:
            return {"error": "手动股票池不能为空，请输入至少 1 个股票代码。"}

        if invalid_tokens:
            warnings.append(f"已忽略无法识别的代码：{', '.join(invalid_tokens[:8])}")

        if use_panel:
            missing_symbols = [s for s in manual_symbols if s not in panel_symbols]
            symbols = [s for s in manual_symbols if s in panel_symbols]
            if missing_symbols:
                warnings.append(f"历史面板中无数据：{', '.join(missing_symbols[:12])}")
            if not symbols:
                return {"error": "手动股票池无有效历史数据，请检查输入代码是否正确，或切换到 MiniQMT 数据源。"}
        else:
            symbols = list(manual_symbols)
            if len(symbols) > 40:
                warnings.append("MiniQMT 模式会逐只拉取历史数据，手动股票池较大时回测会明显变慢。")

        return {
            "mode": universe_mode,
            "label": f"{BACKTEST_UNIVERSE_LABELS[universe_mode]}（{len(symbols)}只）",
            "source": "manual_input",
            "data_source": data_source,
            "data_source_label": BACKTEST_DATA_SOURCE_LABELS[data_source],
            "symbols": symbols,
            "symbols_preview": symbols[:20],
            "requested_symbol_count": len(manual_symbols),
            "resolved_symbol_count": len(symbols),
            "warnings": warnings,
        }

    if universe_mode == "main_board":
        source = "miniqmt_a_share_list"
        try:
            listed_symbols = list_a_share_symbols(limit=None)
        except Exception:
            listed_symbols = []
        listed_symbols = [s for s in listed_symbols if _classify_symbol_board(s) == "主板"]

        if not listed_symbols:
            listed_symbols = sorted([s for s in panel_symbols if _classify_symbol_board(s) == "主板"])
            source = "history_panel"
            warnings.append("实时 A 股列表不可用，已回退历史面板主板股票池。")

        if use_panel:
            symbols = [s for s in listed_symbols if s in panel_symbols]
            if len(symbols) < len(listed_symbols):
                warnings.append(f"已过滤 {len(listed_symbols) - len(symbols)} 只历史面板缺失的主板股票。")
        else:
            symbols = list(listed_symbols)
            if len(symbols) > 80:
                warnings.append("MiniQMT 模式会逐只拉取历史数据，主板全市场回测可能较慢。")

        if not symbols:
            return {"error": "主板股票池为空，请确认所选数据源可用。"}

        return {
            "mode": universe_mode,
            "label": f"{BACKTEST_UNIVERSE_LABELS[universe_mode]}（{len(symbols)}只）",
            "source": source,
            "data_source": data_source,
            "data_source_label": BACKTEST_DATA_SOURCE_LABELS[data_source],
            "symbols": symbols,
            "symbols_preview": symbols[:20],
            "requested_symbol_count": len(listed_symbols),
            "resolved_symbol_count": len(symbols),
            "warnings": warnings,
        }

    if universe_mode == "industry":
        picked_industry = str(industry or "").strip()
        if not picked_industry:
            return {"error": "行业模式需要选择 industry。"}

        reg = _get_sector_registry(force_refresh=False)
        available = set(reg.get("sectors", [])) | panel_industries | set(LOCAL_SECTOR_SYMBOLS.keys())
        source = "history_panel"
        symbols: List[str] = []

        if reg.get("source") == "miniqmt_dynamic" and picked_industry in set(reg.get("sectors", [])):
            source = "miniqmt_dynamic"
            symbols = list_symbols_in_dynamic_sector(picked_industry, limit=None)

        if not symbols and picked_industry in panel_industries:
            panel = _load_history_panel(force_refresh=False)
            symbols = sorted(panel.loc[panel["industry"] == picked_industry, "stock_code"].dropna().astype(str).unique().tolist())
            source = "history_panel"

        if not symbols and picked_industry in LOCAL_SECTOR_SYMBOLS:
            symbols = [str(x).zfill(6) for x in LOCAL_SECTOR_SYMBOLS[picked_industry]]
            source = "local_fallback"

        if use_panel:
            filtered_symbols = [s for s in symbols if s in panel_symbols]
            if len(filtered_symbols) < len(symbols):
                warnings.append(f"已过滤 {len(symbols) - len(filtered_symbols)} 只历史面板缺失的行业股票。")
            symbols = filtered_symbols
        elif len(symbols) > 80:
            warnings.append("MiniQMT 模式会逐只拉取历史数据，行业股票较多时回测可能较慢。")

        if not symbols:
            return {
                "error": f"所选行业在当前数据源下无可用历史数据：{picked_industry}",
                "available": sorted(available),
            }

        return {
            "mode": universe_mode,
            "label": f"{BACKTEST_UNIVERSE_LABELS[universe_mode]}：{picked_industry}（{len(symbols)}只）",
            "source": source,
            "data_source": data_source,
            "data_source_label": BACKTEST_DATA_SOURCE_LABELS[data_source],
            "industry": picked_industry,
            "symbols": symbols,
            "symbols_preview": symbols[:20],
            "requested_symbol_count": len(symbols),
            "resolved_symbol_count": len(symbols),
            "warnings": warnings,
        }

    return {"error": f"不支持的 universe_mode: {universe_mode}"}


def _finalize_backtest_dataset(
    subset: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    warnings: List[str] | None = None,
) -> Dict[str, Any]:
    warnings = list(warnings or [])
    if subset is None or subset.empty:
        return {
            "histories": {},
            "calendar_dates": [],
            "benchmark_points": [],
            "benchmark_dates": pd.DatetimeIndex([]),
            "benchmark_values": np.array([], dtype=float),
            "warnings": warnings,
        }

    subset = subset.copy()
    subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
    subset["stock_code"] = subset["stock_code"].astype(str).map(_normalize_plain_symbol)

    for col in ["open", "high", "low", "close", "volume", "amount", "is_trading"]:
        if col not in subset.columns:
            subset[col] = np.nan
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    if "industry" not in subset.columns:
        subset["industry"] = ""
    subset["industry"] = subset["industry"].fillna("").astype(str).str.strip()
    subset = subset.dropna(subset=["date"]).sort_values(["stock_code", "date"]).reset_index(drop=True)
    subset = subset[subset["date"] <= end_dt].copy()

    histories: Dict[str, pd.DataFrame] = {}
    for symbol, one in subset.groupby("stock_code", sort=False):
        one = one.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        close_num = pd.to_numeric(one["close"], errors="coerce")
        prev_close = close_num.shift(1)
        one["tradeable"] = (one["is_trading"].fillna(0) == 1) & close_num.notna() & (close_num > 0)
        one["close_prev"] = prev_close
        one["limit_up"] = [
            _is_limit_up_close(symbol, prev_close.iloc[i], close_num.iloc[i]) if i > 0 else False
            for i in range(len(one))
        ]
        histories[str(symbol)] = one

    calendar_dates = sorted(pd.Timestamp(x) for x in subset.loc[(subset["date"] >= start_dt) & (subset["date"] <= end_dt), "date"].drop_duplicates().tolist())

    bench_df = subset.copy()
    bench_df["ret_1d"] = bench_df.groupby("stock_code")["close"].pct_change()
    bench_df = bench_df[(bench_df["date"] >= start_dt) & (bench_df["date"] <= end_dt)]
    bench_df = bench_df[(bench_df["is_trading"].fillna(0) == 1) & bench_df["close"].notna() & bench_df["ret_1d"].notna()]
    daily_ret = bench_df.groupby("date")["ret_1d"].mean().sort_index() if not bench_df.empty else pd.Series(dtype=float)
    bench_nav = (1.0 + daily_ret.fillna(0.0)).cumprod()

    benchmark_points = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "date_ts": pd.Timestamp(idx),
            "equity": round(float(value), 6),
        }
        for idx, value in bench_nav.items()
    ]

    return {
        "histories": histories,
        "calendar_dates": calendar_dates,
        "benchmark_points": benchmark_points,
        "benchmark_dates": pd.DatetimeIndex([item["date_ts"] for item in benchmark_points]),
        "benchmark_values": np.array([item["equity"] for item in benchmark_points], dtype=float),
        "warnings": warnings,
    }


def _prepare_backtest_dataset_from_panel(symbols: List[str], start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> Dict[str, Any]:
    panel = _load_history_panel(force_refresh=False)
    subset = panel[(panel["stock_code"].isin(symbols)) & (panel["date"] <= end_dt)].copy()
    return _finalize_backtest_dataset(subset, start_dt=start_dt, end_dt=end_dt, warnings=[])


def _prepare_backtest_dataset_from_miniqmt(
    symbols: List[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    warmup_days: int = 60,
    industry_hint: str = "",
) -> Dict[str, Any]:
    warnings: List[str] = []
    fetch_start_dt = start_dt - timedelta(days=max(45, int(warmup_days) * 2))
    fetch_start = fetch_start_dt.strftime("%Y%m%d")
    fetch_end = end_dt.strftime("%Y%m%d")

    industry_map: Dict[str, str] = {}
    try:
        panel = _load_history_panel(force_refresh=False)
        industry_map = (
            panel[["stock_code", "industry"]]
            .dropna(subset=["stock_code"])
            .drop_duplicates(subset=["stock_code"], keep="last")
            .set_index("stock_code")["industry"]
            .astype(str)
            .to_dict()
        )
    except Exception:
        industry_map = {}

    frames: List[pd.DataFrame] = []
    failed_symbols: List[str] = []
    empty_symbols: List[str] = []

    for symbol in symbols:
        try:
            one = _daily_from_fetcher(symbol, fetch_start, fetch_end)
        except Exception:
            failed_symbols.append(symbol)
            continue

        if one is None or one.empty:
            empty_symbols.append(symbol)
            continue

        one = one.copy()
        one["date"] = pd.to_datetime(one["date"], errors="coerce")
        one = one.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if one.empty:
            empty_symbols.append(symbol)
            continue

        one["stock_code"] = symbol
        if "turnover" in one.columns:
            one["amount"] = pd.to_numeric(one["turnover"], errors="coerce")
        else:
            one["amount"] = pd.to_numeric(one.get("close"), errors="coerce") * pd.to_numeric(one.get("volume"), errors="coerce").fillna(0.0)
        one["is_trading"] = 1
        one["industry"] = str(industry_hint or industry_map.get(symbol, "")).strip()
        frames.append(one[["date", "stock_code", "open", "high", "low", "close", "volume", "amount", "is_trading", "industry"]])

    if failed_symbols:
        preview = ', '.join(failed_symbols[:8])
        suffix = ' 等' if len(failed_symbols) > 8 else ''
        warnings.append(f"MiniQMT 历史获取失败：{preview}{suffix}")
    if empty_symbols:
        preview = ', '.join(empty_symbols[:8])
        suffix = ' 等' if len(empty_symbols) > 8 else ''
        warnings.append(f"MiniQMT 未返回有效历史：{preview}{suffix}")

    subset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _finalize_backtest_dataset(subset, start_dt=start_dt, end_dt=end_dt, warnings=warnings)


def _prepare_backtest_dataset(
    symbols: List[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    data_source: str = "panel",
    warmup_days: int = 60,
    industry_hint: str = "",
) -> Dict[str, Any]:
    if data_source == "miniqmt":
        return _prepare_backtest_dataset_from_miniqmt(
            symbols,
            start_dt=start_dt,
            end_dt=end_dt,
            warmup_days=warmup_days,
            industry_hint=industry_hint,
        )
    return _prepare_backtest_dataset_from_panel(symbols, start_dt=start_dt, end_dt=end_dt)


def _history_index_on_or_before(df: pd.DataFrame, target_dt: pd.Timestamp) -> int:
    if df is None or df.empty:
        return -1
    pos = df["date"].searchsorted(pd.Timestamp(target_dt), side="right") - 1
    return int(pos)


def _find_first_tradable_index(df: pd.DataFrame, start_idx: int, avoid_limit_up: bool = False) -> int:
    idx = max(0, int(start_idx))
    while idx < len(df):
        row = df.iloc[idx]
        if bool(row.get("tradeable", False)):
            if avoid_limit_up and bool(row.get("limit_up", False)):
                idx += 1
                continue
            return idx
        idx += 1
    return -1


def _build_position_outcome(symbol: str, df: pd.DataFrame, entry_idx: int, hold_days: int) -> Dict[str, Any] | None:
    entry_row = df.iloc[entry_idx]
    if not _row_is_tradeable(entry_row):
        return None

    exit_idx = _find_first_tradable_index(df, entry_idx + max(1, int(hold_days)), avoid_limit_up=False)
    if exit_idx < 0:
        return None

    exit_row = df.iloc[exit_idx]
    entry_price = _to_float(entry_row.get("close"), 0.0)
    exit_price = _to_float(exit_row.get("close"), 0.0)
    if entry_price <= 0 or exit_price <= 0:
        return None

    gross_return = float(exit_price / entry_price - 1.0)
    return {
        "symbol": symbol,
        "industry": str(entry_row.get("industry") or "").strip(),
        "entry_idx": int(entry_idx),
        "exit_idx": int(exit_idx),
        "entry_date_ts": pd.Timestamp(entry_row["date"]),
        "exit_date_ts": pd.Timestamp(exit_row["date"]),
        "entry_date": _format_bt_date(entry_row["date"]),
        "exit_date": _format_bt_date(exit_row["date"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross_return,
        "return": gross_return,
        "hold_trade_days": int(exit_idx - entry_idx),
    }


def _apply_position_transaction_costs(position: Dict[str, Any], allocated_capital: float, cost_config: Dict[str, Any]) -> Dict[str, Any]:
    capital = max(_to_float(allocated_capital, 0.0), 1.0)
    gross_return = _to_float(position.get("gross_return", position.get("return", 0.0)), 0.0)
    exit_notional = max(capital * (1.0 + gross_return), 0.0)

    buy_commission_rate = _to_float(cost_config.get("buy_commission_rate"), 0.0)
    sell_commission_rate = _to_float(cost_config.get("sell_commission_rate"), 0.0)
    min_commission = _to_float(cost_config.get("min_commission"), 0.0)
    sell_stamp_tax_rate = _to_float(cost_config.get("sell_stamp_tax_rate"), 0.0)
    transfer_fee_rate = _to_float(cost_config.get("transfer_fee_rate"), 0.0)
    slippage_rate = _to_float(cost_config.get("slippage_rate"), 0.0)

    buy_commission = max(capital * buy_commission_rate, min_commission) if buy_commission_rate > 0 else 0.0
    sell_commission = max(exit_notional * sell_commission_rate, min_commission) if sell_commission_rate > 0 else 0.0
    buy_transfer_fee = capital * transfer_fee_rate
    sell_transfer_fee = exit_notional * transfer_fee_rate
    buy_slippage = capital * slippage_rate
    sell_slippage = exit_notional * slippage_rate
    sell_stamp_tax = exit_notional * sell_stamp_tax_rate

    total_cost = buy_commission + sell_commission + buy_transfer_fee + sell_transfer_fee + buy_slippage + sell_slippage + sell_stamp_tax
    net_profit = exit_notional - sell_commission - sell_transfer_fee - sell_slippage - sell_stamp_tax - capital - buy_commission - buy_transfer_fee - buy_slippage
    net_return = float(net_profit / capital)

    out = dict(position)
    out.update({
        "allocated_capital": float(capital),
        "gross_return": float(gross_return),
        "return": net_return,
        "cost_return": float(max(gross_return - net_return, 0.0)),
        "cost_amount": float(total_cost),
        "buy_commission": float(buy_commission),
        "sell_commission": float(sell_commission),
        "buy_transfer_fee": float(buy_transfer_fee),
        "sell_transfer_fee": float(sell_transfer_fee),
        "buy_slippage": float(buy_slippage),
        "sell_slippage": float(sell_slippage),
        "sell_stamp_tax": float(sell_stamp_tax),
    })
    return out


def _history_slice_for_model(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    cols = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    return df.iloc[: idx + 1][cols].copy().reset_index(drop=True)


def _prepare_rule_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    one = df.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(one.get("close"), errors="coerce")
    high = pd.to_numeric(one.get("high"), errors="coerce")
    open_ = pd.to_numeric(one.get("open"), errors="coerce")
    volume = pd.to_numeric(one.get("volume"), errors="coerce")

    one["ret_1d"] = close.pct_change()
    one["prev_ret_1d"] = one["ret_1d"].shift(1)
    one["ma5"] = close.rolling(5).mean()
    one["ma10"] = close.rolling(10).mean()
    one["ma20"] = close.rolling(20).mean()
    one["ma5_prev"] = one["ma5"].shift(1)
    one["ma20_prev"] = one["ma20"].shift(1)
    one["high20_prev"] = high.shift(1).rolling(20).max()
    one["volume_ma20_prev"] = volume.shift(1).rolling(20).mean()
    one["body_ret"] = np.where(open_ > 0, close / open_ - 1.0, np.nan)
    return one


def _rule_signal_limit_up_follow(row: pd.Series) -> bool:
    return bool(row.get("tradeable", False)) and bool(row.get("limit_up", False))


def _rule_signal_breakout_20d_high(row: pd.Series) -> bool:
    if not bool(row.get("tradeable", False)):
        return False
    high20_prev = row.get("high20_prev")
    ma20 = row.get("ma20")
    close = _to_float(row.get("close"), 0.0)
    return pd.notna(high20_prev) and pd.notna(ma20) and close > float(high20_prev) and close > float(ma20)


def _rule_signal_ma5_cross_ma20(row: pd.Series) -> bool:
    if not bool(row.get("tradeable", False)):
        return False
    needed = [row.get("ma5_prev"), row.get("ma20_prev"), row.get("ma5"), row.get("ma20")]
    if any(pd.isna(x) for x in needed):
        return False
    close = _to_float(row.get("close"), 0.0)
    return float(row["ma5_prev"]) <= float(row["ma20_prev"]) and float(row["ma5"]) > float(row["ma20"]) and close > float(row["ma20"])


def _rule_signal_volume_breakout_20d_high(row: pd.Series) -> bool:
    if not _rule_signal_breakout_20d_high(row):
        return False
    volume_ma20_prev = row.get("volume_ma20_prev")
    volume = _to_float(row.get("volume"), 0.0)
    return pd.notna(volume_ma20_prev) and volume > float(volume_ma20_prev) * 1.5


def _rule_signal_oversold_rebound(row: pd.Series) -> bool:
    if not bool(row.get("tradeable", False)):
        return False
    prev_ret = row.get("prev_ret_1d")
    ret_1d = row.get("ret_1d")
    body_ret = row.get("body_ret")
    if pd.isna(prev_ret) or pd.isna(ret_1d) or pd.isna(body_ret):
        return False
    return float(prev_ret) <= -0.05 and float(ret_1d) >= 0.02 and float(body_ret) > 0.0


def _build_rule_signal_note(rule_preset: str, delay_days: int) -> str:
    base_map = {
        "limit_up_follow": "涨停信号后次日跟随",
        "breakout_20d_high": "收盘价突破前20日高点",
        "ma5_cross_ma20": "5日均线上穿20日均线",
        "volume_breakout_20d_high": "放量突破前20日高点",
        "oversold_rebound": "前一日超跌后当日转强反弹",
    }
    base = base_map.get(rule_preset, BACKTEST_RULE_PRESET_LABELS.get(rule_preset, rule_preset))
    if delay_days <= 0:
        return f"{base}，T+1 执行"
    return f"{base}，延后 {delay_days} 个交易日执行"


def _realtime_signal_note(rule_preset: str) -> str:
    mapping = {
        "limit_up_follow": "当日涨停，可关注次日跟随机会",
        "breakout_20d_high": "当日收盘突破前20日高点",
        "ma5_cross_ma20": "5日均线当日上穿20日均线",
        "volume_breakout_20d_high": "当日放量突破前20日高点",
        "oversold_rebound": "前一日超跌后当日转强反弹",
    }
    return mapping.get(rule_preset, REALTIME_SIGNAL_PRESET_LABELS.get(rule_preset, rule_preset))


def _generate_realtime_signal_marks(daily_df: pd.DataFrame, rule_preset: str) -> Dict[str, Any]:
    signal_map = {
        "limit_up_follow": _rule_signal_limit_up_follow,
        "breakout_20d_high": _rule_signal_breakout_20d_high,
        "ma5_cross_ma20": _rule_signal_ma5_cross_ma20,
        "volume_breakout_20d_high": _rule_signal_volume_breakout_20d_high,
        "oversold_rebound": _rule_signal_oversold_rebound,
    }
    signal_fn = signal_map.get(rule_preset)
    if signal_fn is None:
        raise ValueError(f"不支持的 signal_preset: {rule_preset}")

    prepared = _prepare_rule_indicator_frame(daily_df)
    prepared["tradeable"] = prepared["close"].notna() & (pd.to_numeric(prepared["close"], errors="coerce") > 0)
    symbol_code = _normalize_plain_symbol(str(daily_df.iloc[0].get("symbol") or daily_df.iloc[0].get("stock_code") or "000001")) or "000001"
    prev_close = pd.to_numeric(prepared["close"], errors="coerce").shift(1)
    prepared["limit_up"] = [
        _is_limit_up_close(symbol_code, prev_close.iloc[i], prepared["close"].iloc[i]) if i > 0 else False
        for i in range(len(prepared))
    ]

    marks = []
    for i in range(len(prepared)):
        row = prepared.iloc[i]
        if not signal_fn(row):
            continue
        price = _to_float(row.get("high") or row.get("close"), 0.0)
        marks.append({
            "date": _format_bt_date(row["date"]),
            "price": round(price, 4),
            "label": REALTIME_SIGNAL_PRESET_LABELS.get(rule_preset, rule_preset),
            "note": _realtime_signal_note(rule_preset),
            "is_latest": bool(i == len(prepared) - 1),
        })

    return {
        "preset": rule_preset,
        "preset_label": REALTIME_SIGNAL_PRESET_LABELS.get(rule_preset, rule_preset),
        "marks": marks,
        "latest_signal": marks[-1] if marks else None,
        "signal_count": len(marks),
    }


def _generate_rule_batches(
    histories: Dict[str, pd.DataFrame],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    hold_days: int,
    rule_preset: str,
) -> List[Dict[str, Any]]:
    signal_map = {
        "limit_up_follow": _rule_signal_limit_up_follow,
        "breakout_20d_high": _rule_signal_breakout_20d_high,
        "ma5_cross_ma20": _rule_signal_ma5_cross_ma20,
        "volume_breakout_20d_high": _rule_signal_volume_breakout_20d_high,
        "oversold_rebound": _rule_signal_oversold_rebound,
    }
    min_history_map = {
        "limit_up_follow": 2,
        "breakout_20d_high": 22,
        "ma5_cross_ma20": 22,
        "volume_breakout_20d_high": 22,
        "oversold_rebound": 3,
    }
    signal_fn = signal_map.get(rule_preset)
    if signal_fn is None:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    required_rows = max(min_history_map.get(rule_preset, 2), hold_days + 2)

    for symbol, raw_df in histories.items():
        if len(raw_df) <= required_rows:
            continue

        df = _prepare_rule_indicator_frame(raw_df)
        i = 1
        while i < len(df) - hold_days:
            row = df.iloc[i]
            signal_date = pd.Timestamp(row["date"])
            if signal_date > end_dt:
                break
            if signal_date < start_dt or (not signal_fn(row)):
                i += 1
                continue

            entry_idx = _find_first_tradable_index(df, i + 1, avoid_limit_up=True)
            if entry_idx < 0:
                break

            position = _build_position_outcome(symbol, df, entry_idx, hold_days)
            if not position:
                i += 1
                continue
            if position["entry_date_ts"] > end_dt or position["exit_date_ts"] > end_dt:
                i += 1
                continue

            delay_days = max(0, entry_idx - (i + 1))
            position["signal_date_ts"] = signal_date
            position["signal_date"] = _format_bt_date(signal_date)
            position["score"] = None
            position["note"] = _build_rule_signal_note(rule_preset, delay_days)
            grouped[position["entry_date"]].append(position)
            i = position["exit_idx"] + 1

    batches: List[Dict[str, Any]] = []
    strategy_label = BACKTEST_RULE_PRESET_LABELS.get(rule_preset, rule_preset)
    for entry_date in sorted(grouped.keys()):
        positions = sorted(grouped[entry_date], key=lambda item: item["symbol"])
        batch_exit = max(item["exit_date_ts"] for item in positions)
        batches.append({
            "entry_date": entry_date,
            "entry_date_ts": pd.Timestamp(positions[0]["entry_date_ts"]),
            "exit_date": _format_bt_date(batch_exit),
            "exit_date_ts": batch_exit,
            "positions": positions,
            "candidate_count": len(positions),
            "fallback_count": 0,
            "note": f"{len(positions)} 只股票触发{strategy_label}信号",
        })
    return batches


def _generate_model_topk_batches(
    histories: Dict[str, pd.DataFrame],
    calendar_dates: List[pd.Timestamp],
    end_dt: pd.Timestamp,
    hold_days: int,
    top_k: int,
    backend: str | None,
    min_history: int,
) -> List[Dict[str, Any]]:
    if not calendar_dates:
        return []

    batches: List[Dict[str, Any]] = []
    required_history = max(30, int(min_history))
    rebalance_dates = calendar_dates[:: max(1, int(hold_days))]

    for rebalance_dt in rebalance_dates:
        history_windows: List[pd.DataFrame] = []
        candidates: List[Dict[str, Any]] = []

        for symbol, df in histories.items():
            idx = _history_index_on_or_before(df, rebalance_dt)
            if idx < required_history - 1:
                continue
            if pd.Timestamp(df.iloc[idx]["date"]) != pd.Timestamp(rebalance_dt):
                continue
            if not bool(df.iloc[idx].get("tradeable", False)):
                continue

            position = _build_position_outcome(symbol, df, idx, hold_days)
            if not position or position["exit_date_ts"] > end_dt:
                continue

            history_windows.append(_history_slice_for_model(df, idx))
            candidates.append(position)

        if not history_windows:
            continue

        preds = predict_probability_batch(history_windows, backend=backend, allow_fallback=True)
        scored: List[Dict[str, Any]] = []
        fallback_count = 0
        for position, pred in zip(candidates, preds):
            item = dict(position)
            item["score"] = float(pred.get("p_up_today", 0.5))
            item["backend"] = pred.get("backend", "rule")
            item["backend_requested"] = pred.get("backend_requested", backend or "rule")
            item["backend_fallback"] = bool(pred.get("backend_fallback", False))
            item["backend_error"] = pred.get("backend_error", "")
            item["note"] = f"1日上涨概率 {item['score']:.1%}"
            scored.append(item)
            if item["backend_fallback"]:
                fallback_count += 1

        scored = sorted(scored, key=lambda item: (item.get("score", 0.0), item.get("return", -999.0)), reverse=True)
        picked = scored[: min(int(top_k), len(scored))]
        if not picked:
            continue

        batch_exit = max(item["exit_date_ts"] for item in picked)
        batches.append({
            "entry_date": _format_bt_date(rebalance_dt),
            "entry_date_ts": pd.Timestamp(rebalance_dt),
            "exit_date": _format_bt_date(batch_exit),
            "exit_date_ts": batch_exit,
            "positions": picked,
            "candidate_count": len(scored),
            "fallback_count": fallback_count,
            "note": f"候选 {len(scored)} 只，买入 Top {len(picked)}",
        })
    return batches


def _sample_benchmark_equity(benchmark_dates: pd.DatetimeIndex, benchmark_values: np.ndarray, target_dt: pd.Timestamp) -> float:
    if benchmark_dates.empty or len(benchmark_values) == 0:
        return 1.0
    pos = int(benchmark_dates.searchsorted(pd.Timestamp(target_dt), side="right") - 1)
    if pos < 0:
        return 1.0
    return float(benchmark_values[pos])


def _build_equity_curve(
    executed_batches: List[Dict[str, Any]],
    benchmark_points: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not benchmark_points:
        return [
            {
                "date": item["exit_date"],
                "strategy": round(float(item["strategy_equity"]), 6),
                "benchmark": round(float(item["benchmark_equity"]), 6),
            }
            for item in executed_batches
        ]

    curve: List[Dict[str, Any]] = []
    batch_idx = 0
    strategy_equity = 1.0
    for point in benchmark_points:
        while batch_idx < len(executed_batches) and executed_batches[batch_idx]["exit_date_ts"] <= point["date_ts"]:
            strategy_equity = float(executed_batches[batch_idx]["strategy_equity"])
            batch_idx += 1
        curve.append({
            "date": point["date"],
            "strategy": round(strategy_equity, 6),
            "benchmark": round(float(point["equity"]), 6),
        })
    return curve


def _run_non_overlapping_batches(
    batches: List[Dict[str, Any]],
    benchmark_points: List[Dict[str, Any]],
    benchmark_dates: pd.DatetimeIndex,
    benchmark_values: np.ndarray,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    cost_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    sorted_batches = sorted(batches, key=lambda item: (item["entry_date_ts"], item["exit_date_ts"]))
    cost_cfg = dict(BACKTEST_COST_DEFAULTS)
    cost_cfg.update(cost_config or {})

    executed: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    batch_returns: List[float] = []
    skipped_overlap = 0
    strategy_equity = 1.0
    strategy_gross_equity = 1.0
    lock_until: pd.Timestamp | None = None
    total_cost_amount = 0.0

    for batch in sorted_batches:
        if lock_until is not None and batch["entry_date_ts"] <= lock_until:
            skipped_overlap += 1
            continue

        positions = list(batch.get("positions", []))
        if not positions:
            continue

        allocated_capital = strategy_equity * _to_float(cost_cfg.get("initial_capital"), 1000000.0) / max(len(positions), 1)
        costed_positions = [_apply_position_transaction_costs(position, allocated_capital, cost_cfg) for position in positions]

        gross_batch_return = float(np.mean([item.get("gross_return", item.get("return", 0.0)) for item in costed_positions]))
        batch_return = float(np.mean([item["return"] for item in costed_positions]))
        batch_cost_return = float(np.mean([item.get("cost_return", 0.0) for item in costed_positions]))
        batch_cost_amount = float(np.sum([item.get("cost_amount", 0.0) for item in costed_positions]))

        strategy_gross_equity *= (1.0 + gross_batch_return)
        strategy_equity *= (1.0 + batch_return)
        total_cost_amount += batch_cost_amount
        lock_until = pd.Timestamp(batch["exit_date_ts"])
        benchmark_equity = _sample_benchmark_equity(benchmark_dates, benchmark_values, batch["exit_date_ts"])
        symbol_preview = ", ".join([item["symbol"] for item in costed_positions[:6]])
        if len(costed_positions) > 6:
            symbol_preview += f" 等{len(costed_positions)}只"

        batch_log = {
            "entry_date": batch["entry_date"],
            "entry_date_ts": pd.Timestamp(batch["entry_date_ts"]),
            "exit_date": batch["exit_date"],
            "exit_date_ts": pd.Timestamp(batch["exit_date_ts"]),
            "trade_count": len(costed_positions),
            "candidate_count": int(batch.get("candidate_count", len(costed_positions))),
            "hold_days": int(max(item.get("hold_trade_days", 0) for item in costed_positions)),
            "gross_batch_return": round(gross_batch_return, 6),
            "batch_return": round(batch_return, 6),
            "cost_return": round(batch_cost_return, 6),
            "cost_amount": round(batch_cost_amount, 2),
            "strategy_equity": round(strategy_equity, 6),
            "gross_strategy_equity": round(strategy_gross_equity, 6),
            "benchmark_equity": round(benchmark_equity, 6),
            "fallback_count": int(batch.get("fallback_count", 0)),
            "symbols": symbol_preview,
            "note": str(batch.get("note") or ""),
        }
        executed.append(batch_log)
        batch_returns.append(batch_return)

        for position in costed_positions:
            trade_rows.append({
                "signal_date": position.get("signal_date", position["entry_date"]),
                "entry_date": position["entry_date"],
                "exit_date": position["exit_date"],
                "symbol": position["symbol"],
                "industry": position.get("industry", ""),
                "entry_price": round(float(position["entry_price"]), 4),
                "exit_price": round(float(position["exit_price"]), 4),
                "gross_return": round(float(position.get("gross_return", position["return"])), 6),
                "return": round(float(position["return"]), 6),
                "cost_return": round(float(position.get("cost_return", 0.0)), 6),
                "cost_amount": round(float(position.get("cost_amount", 0.0)), 2),
                "score": round(float(position["score"]), 4) if position.get("score") is not None else None,
                "hold_days": int(position.get("hold_trade_days", 0)),
                "note": str(position.get("note") or ""),
            })

    final_benchmark = _sample_benchmark_equity(benchmark_dates, benchmark_values, end_dt)
    if executed:
        final_strategy = float(executed[-1]["strategy_equity"])
        final_gross_strategy = float(executed[-1]["gross_strategy_equity"])
        curve = _build_equity_curve(executed, benchmark_points)
    else:
        final_strategy = 1.0
        final_gross_strategy = 1.0
        curve = [
            {
                "date": point["date"],
                "strategy": 1.0,
                "benchmark": round(float(point["equity"]), 6),
            }
            for point in benchmark_points
        ]

    strategy_curve = np.array([point["strategy"] for point in curve], dtype=float) if curve else np.array([1.0], dtype=float)
    running_max = np.maximum.accumulate(strategy_curve)
    max_drawdown = float(np.min(strategy_curve / running_max - 1.0)) if len(strategy_curve) else 0.0

    batch_returns_np = np.array(batch_returns, dtype=float)
    avg_hold_days = float(np.mean([row["hold_days"] for row in executed])) if executed else 0.0
    annual_factor = np.sqrt(252.0 / max(avg_hold_days, 1.0)) if avg_hold_days else 0.0
    sharpe = float(batch_returns_np.mean() / batch_returns_np.std() * annual_factor) if len(batch_returns_np) > 1 and batch_returns_np.std() > 1e-12 else 0.0
    span_days = max(int((end_dt - start_dt).days), 1)
    annual_return = float((final_strategy ** (365.0 / span_days)) - 1.0) if final_strategy > 0 else -1.0

    recent_trades = sorted(trade_rows, key=lambda item: (item["entry_date"], item["symbol"]), reverse=True)
    recent_batches = [
        {k: v for k, v in row.items() if not k.endswith("_ts")}
        for row in reversed(executed)
    ]

    return {
        "summary": {
            "start": _format_bt_date(start_dt),
            "end": _format_bt_date(end_dt),
            "calendar_days": span_days,
            "batch_count": len(executed),
            "trade_count": len(trade_rows),
            "skipped_overlap_batches": skipped_overlap,
            "avg_positions_per_batch": round(float(np.mean([row["trade_count"] for row in executed])), 2) if executed else 0.0,
            "avg_hold_days": round(avg_hold_days, 2),
            "strategy_total_return": round(final_strategy - 1.0, 6),
            "gross_strategy_total_return": round(final_gross_strategy - 1.0, 6),
            "benchmark_total_return": round(final_benchmark - 1.0, 6),
            "excess_return": round((final_strategy / final_benchmark - 1.0) if final_benchmark > 0 else (final_strategy - 1.0), 6),
            "annual_return": round(annual_return, 6),
            "sharpe": round(sharpe, 6),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round(float((batch_returns_np > 0).mean()), 6) if len(batch_returns_np) else 0.0,
            "last_strategy_equity": round(final_strategy, 6),
            "last_benchmark_equity": round(final_benchmark, 6),
            "total_cost_amount": round(float(total_cost_amount), 2),
            "avg_cost_per_trade": round(float(total_cost_amount / len(trade_rows)), 4) if trade_rows else 0.0,
            "initial_capital": round(float(cost_cfg.get("initial_capital", 1000000.0)), 2),
        },
        "equity_curve": curve,
        "recent_trades": recent_trades,
        "batch_log": recent_batches,
    }


def _is_legacy_backtest_request(args) -> bool:
    new_keys = {
        "universe_mode", "symbols", "industry", "strategy_category", "rule_preset", "model_preset", "model_backend",
        "top_k", "hold_days", "data_source", "initial_capital", "buy_commission_rate", "sell_commission_rate",
        "min_commission", "sell_stamp_tax_rate", "transfer_fee_rate", "slippage_rate"
    }
    if any(args.get(key) is not None for key in new_keys):
        return False
    return any(args.get(key) is not None for key in {"symbol", "threshold", "long_horizon", "backend"})


def _market_backtest_legacy():
    symbol = (request.args.get("symbol") or "000001").strip()
    start = (request.args.get("start") or "20180101").strip()
    end = (request.args.get("end") or datetime.now().strftime("%Y%m%d")).strip()
    threshold = float(request.args.get("threshold", "0.5"))
    threshold = max(0.05, min(threshold, 0.95))
    backend = (request.args.get("backend") or "").strip() or None

    min_history = max(30, min(int(request.args.get("min_history", "120")), 400))
    long_horizon = max(20, min(int(request.args.get("long_horizon", "60")), 120))

    daily = _daily_from_fetcher(symbol, start, end)
    if len(daily) < min_history + long_horizon + 10:
        return jsonify({"error": "历史样本不足，无法回测。请扩大时间区间。"}), 400

    daily = daily.set_index("date")

    eval_indices = list(range(min_history, len(daily) - long_horizon))
    histories = [daily.iloc[: i + 1].reset_index() for i in eval_indices]
    preds = predict_probability_batch(histories, backend=backend, allow_fallback=True)

    rows = []
    for i, pred in zip(eval_indices, preds):
        close_t = float(daily["close"].iloc[i])
        ret1 = float(daily["close"].iloc[i + 1] / close_t - 1)
        ret5 = float(daily["close"].iloc[i + 5] / close_t - 1)
        retl = float(daily["close"].iloc[i + long_horizon] / close_t - 1)

        rows.append({
            "date": daily.index[i],
            "p1": pred["p_up_today"],
            "p5": pred["p_up_5d"],
            "pl": pred["p_up_long"],
            "y1": 1 if ret1 > 0 else 0,
            "y5": 1 if ret5 > 0 else 0,
            "yl": 1 if retl > 0 else 0,
        })

    bt = pd.DataFrame(rows).set_index("date")
    if bt.empty:
        return jsonify({"error": "回测结果为空"}), 400

    y1, p1 = bt["y1"].values.astype(int), bt["p1"].values.astype(float)
    y5, p5 = bt["y5"].values.astype(int), bt["p5"].values.astype(float)
    yl, pl = bt["yl"].values.astype(int), bt["pl"].values.astype(float)

    m1 = _classification_metrics(y1, p1, threshold=threshold)
    m5 = _classification_metrics(y5, p5, threshold=threshold)
    ml = _classification_metrics(yl, pl, threshold=threshold)

    s1 = _strategy_metrics(daily.loc[bt.index], bt["p1"], horizon=1, threshold=threshold)
    s5 = _strategy_metrics(daily.loc[bt.index], bt["p5"], horizon=5, threshold=threshold)
    sl = _strategy_metrics(daily.loc[bt.index], bt["pl"], horizon=long_horizon, threshold=threshold)

    return jsonify({
        "symbol": symbol,
        "model_backend": get_backend_runtime_status(backend),
        "window": {"start": start, "end": end, "samples": int(len(bt)), "min_history": min_history, "long_horizon": long_horizon},
        "classification": {"d1": m1, "d5": m5, "long": ml},
        "strategy": {"d1": s1, "d5": s5, "long": sl},
        "calibration": {
            "d1": _calibration_bins(y1, p1, bins=10),
            "d5": _calibration_bins(y5, p5, bins=10),
            "long": _calibration_bins(yl, pl, bins=10),
        },
        "preview": [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "p1": round(float(r["p1"]), 4), "y1": int(r["y1"]),
                "p5": round(float(r["p5"]), 4), "y5": int(r["y5"]),
                "pl": round(float(r["pl"]), 4), "yl": int(r["yl"]),
            }
            for idx, r in bt.tail(80).iterrows()
        ],
        "legacy_mode": True,
    })


@app.route("/api/market/backtest")
def market_backtest():
    if _is_legacy_backtest_request(request.args):
        try:
            return _market_backtest_legacy()
        except Exception as e:
            return jsonify({"error": f"回测失败: {str(e)}"}), 500

    universe_mode = (request.args.get("universe_mode") or "manual").strip()
    symbols_text = request.args.get("symbols") or ""
    industry = request.args.get("industry") or ""
    strategy_category = (request.args.get("strategy_category") or "rule").strip()
    rule_preset = (request.args.get("rule_preset") or "limit_up_follow").strip()
    model_preset = (request.args.get("model_preset") or "topk_prob_1d").strip()
    model_backend = (request.args.get("model_backend") or "auto").strip()
    data_source = (request.args.get("data_source") or "panel").strip()
    start_text = request.args.get("start") or "20200101"
    end_text = request.args.get("end") or datetime.now().strftime("%Y%m%d")
    hold_days = max(1, min(int(request.args.get("hold_days", "1")), 60))
    top_k = max(1, min(int(request.args.get("top_k", "5")), 200))
    min_history = max(30, min(int(request.args.get("min_history", "60")), 240))
    cost_config = _parse_backtest_cost_config(request.args)

    if universe_mode not in BACKTEST_UNIVERSE_LABELS:
        return jsonify({"error": f"不支持的 universe_mode: {universe_mode}"}), 400
    if strategy_category not in BACKTEST_STRATEGY_LABELS:
        return jsonify({"error": f"不支持的 strategy_category: {strategy_category}"}), 400
    if rule_preset not in BACKTEST_RULE_PRESET_LABELS:
        return jsonify({"error": f"不支持的 rule_preset: {rule_preset}"}), 400
    if model_preset not in BACKTEST_MODEL_PRESET_LABELS:
        return jsonify({"error": f"不支持的 model_preset: {model_preset}"}), 400
    if model_backend not in BACKTEST_MODEL_BACKEND_LABELS:
        return jsonify({"error": f"不支持的 model_backend: {model_backend}"}), 400
    if data_source not in BACKTEST_DATA_SOURCE_LABELS:
        return jsonify({"error": f"不支持的 data_source: {data_source}"}), 400

    try:
        start_dt = _parse_bt_date(start_text, "start")
        end_dt = _parse_bt_date(end_text, "end")
        if start_dt >= end_dt:
            return jsonify({"error": "start 必须早于 end。"}), 400

        universe = _resolve_backtest_universe(universe_mode, symbols_text=symbols_text, industry=industry, data_source=data_source)
        if universe.get("error"):
            status = 400
            return jsonify(universe), status

        warmup_days = max(min_history if strategy_category == "model" else (hold_days + 5), 40)
        dataset = _prepare_backtest_dataset(
            universe["symbols"],
            start_dt=start_dt,
            end_dt=end_dt,
            data_source=data_source,
            warmup_days=warmup_days,
            industry_hint=industry if universe_mode == "industry" else "",
        )
        histories = dataset["histories"]
        if not histories:
            if data_source == "miniqmt":
                return jsonify({"error": "MiniQMT 未返回可用于回测的历史数据，请确认代码、日期区间和终端行情状态。", "warnings": dataset.get("warnings", [])}), 400
            return jsonify({"error": "所选股票池没有可用于回测的历史数据。", "warnings": dataset.get("warnings", [])}), 400

        if strategy_category == "rule":
            batches = _generate_rule_batches(
                histories,
                start_dt=start_dt,
                end_dt=end_dt,
                hold_days=hold_days,
                rule_preset=rule_preset,
            )
            strategy_info = {
                "category": strategy_category,
                "category_label": BACKTEST_STRATEGY_LABELS[strategy_category],
                "preset": rule_preset,
                "preset_label": BACKTEST_RULE_PRESET_LABELS[rule_preset],
                "hold_days": hold_days,
            }
        else:
            batches = _generate_model_topk_batches(
                histories,
                calendar_dates=dataset["calendar_dates"],
                end_dt=end_dt,
                hold_days=hold_days,
                top_k=top_k,
                backend=model_backend,
                min_history=min_history,
            )
            strategy_info = {
                "category": strategy_category,
                "category_label": BACKTEST_STRATEGY_LABELS[strategy_category],
                "preset": model_preset,
                "preset_label": BACKTEST_MODEL_PRESET_LABELS[model_preset],
                "hold_days": hold_days,
                "top_k": top_k,
                "backend_requested": model_backend,
                "backend_label": BACKTEST_MODEL_BACKEND_LABELS[model_backend],
                "backend_status": get_backend_runtime_status(model_backend),
                "min_history": min_history,
            }

        run_result = _run_non_overlapping_batches(
            batches=batches,
            benchmark_points=dataset["benchmark_points"],
            benchmark_dates=dataset["benchmark_dates"],
            benchmark_values=dataset["benchmark_values"],
            start_dt=start_dt,
            end_dt=end_dt,
            cost_config=cost_config,
        )

        warnings = list(universe.get("warnings", [])) + list(dataset.get("warnings", []))
        if not batches:
            warnings.append("当前参数下没有产生可执行批次，策略净值保持为 1。")

        return jsonify({
            "mode": "portfolio_v2",
            "request": {
                "start": _format_bt_date(start_dt),
                "end": _format_bt_date(end_dt),
                "hold_days": hold_days,
                "data_source": data_source,
                "data_source_label": BACKTEST_DATA_SOURCE_LABELS[data_source],
                "transaction_costs": cost_config,
            },
            "universe": {
                "mode": universe["mode"],
                "label": universe["label"],
                "source": universe["source"],
                "industry": universe.get("industry", ""),
                "requested_symbol_count": int(universe["requested_symbol_count"]),
                "resolved_symbol_count": int(universe["resolved_symbol_count"]),
                "symbols_preview": universe.get("symbols_preview", []),
            },
            "strategy": strategy_info,
            "summary": run_result["summary"],
            "equity_curve": run_result["equity_curve"],
            "recent_trades": run_result["recent_trades"],
            "batch_log": run_result["batch_log"],
            "warnings": warnings,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"回测失败: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8787, debug=True)
