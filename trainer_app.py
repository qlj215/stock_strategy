from datetime import datetime, timedelta
import os
import sys
import random
import uuid
import subprocess
import threading
from typing import Dict, Tuple

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

        intraday = _intraday_from_fetcher(symbol, count=240)

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
        })
    except Exception as e:
        return jsonify({"error": f"数据获取失败: {str(e)}"}), 500


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

    intraday = _intraday_from_fetcher(symbol, count=240)
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


@app.route("/api/market/backtest")
def market_backtest():
    symbol = (request.args.get("symbol") or "000001").strip()
    start = (request.args.get("start") or "20180101").strip()
    end = (request.args.get("end") or datetime.now().strftime("%Y%m%d")).strip()
    threshold = float(request.args.get("threshold", "0.5"))
    threshold = max(0.05, min(threshold, 0.95))
    backend = (request.args.get("backend") or "").strip() or None

    min_history = max(30, min(int(request.args.get("min_history", "120")), 400))
    long_horizon = max(20, min(int(request.args.get("long_horizon", "60")), 120))

    try:
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
        })
    except Exception as e:
        return jsonify({"error": f"回测失败: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8787, debug=True)
