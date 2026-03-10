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
import akshare as ak

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_strategy.data.fetcher import fetch_stock_data

app = Flask(__name__, static_folder="web", static_url_path="")

SYMBOL_POOL = ["000858", "600519", "000001", "600036", "300750", "002594", "600276", "603986"]
CHALLENGES = {}
REVIEWS = {}
REPLAY_JOBS = {}


def _today_str():
    return datetime.now().strftime("%Y%m%d")


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


def _fetch_event_context(symbol: str, anchor_date: str, max_items: int = 5, lookback_days: int = 45):
    """抓取题目截面日期附近的新闻事件（不是物理当前时间）。"""
    try:
        news = ak.stock_news_em(symbol=symbol)
        if news is None or news.empty:
            return []

        anchor_dt = pd.to_datetime(anchor_date)
        start_dt = anchor_dt - timedelta(days=lookback_days)

        news = news.copy()
        news["发布时间_dt"] = pd.to_datetime(news["发布时间"], errors="coerce")
        news = news.dropna(subset=["发布时间_dt"]) 

        # 只保留“题目截面日及之前”的事件，避免穿越
        filt = news[(news["发布时间_dt"] <= anchor_dt) & (news["发布时间_dt"] >= start_dt)]

        # 若窗口内没有，退化为“截面日前最近事件”
        if filt.empty:
            filt = news[news["发布时间_dt"] <= anchor_dt]

        if filt.empty:
            return []

        filt = filt.sort_values("发布时间_dt", ascending=False).head(max_items)

        items = []
        for _, r in filt.iterrows():
            items.append({
                "time": str(r.get("发布时间", "")),
                "title": str(r.get("新闻标题", "")).strip(),
                "source": str(r.get("文章来源", "")).strip(),
            })
        return items
    except Exception:
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

    # 优先按新闻时间抽题，确保事件与题目时期一致
    if require_events:
        try:
            news = ak.stock_news_em(symbol=symbol)
            if news is not None and not news.empty:
                news = news.copy()
                news["发布时间_dt"] = pd.to_datetime(news["发布时间"], errors="coerce")
                news = news.dropna(subset=["发布时间_dt"]) 
                candidate_dates = news["发布时间_dt"].tolist()
                random.shuffle(candidate_dates)
                for dt in candidate_dates[:10]:
                    i_try = df.index.searchsorted(dt, side="right") - 1
                    if i_try < low_idx or i_try > high_idx:
                        continue
                    anchor_try = str(df.index[i_try].date())
                    events_try = _fetch_event_context(symbol, anchor_date=anchor_try, max_items=5)
                    if events_try:
                        selected = (i_try, anchor_try, events_try)
                        break
        except Exception:
            pass

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


def _pick_col(df: pd.DataFrame, names) -> str:
    for n in names:
        if n in df.columns:
            return n
    return ""


def _norm_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "turnover", "pct"])

    df = df.copy()
    date_col = _pick_col(df, ["日期", "date", "时间"])
    open_col = _pick_col(df, ["开盘", "open"])
    high_col = _pick_col(df, ["最高", "high"])
    low_col = _pick_col(df, ["最低", "low"])
    close_col = _pick_col(df, ["收盘", "close", "最新价"])
    volume_col = _pick_col(df, ["成交量", "volume"])
    turnover_col = _pick_col(df, ["成交额", "amount", "turnover"])
    pct_col = _pick_col(df, ["涨跌幅", "pct_chg", "change_percent"])

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT,
        "open": pd.to_numeric(df[open_col], errors="coerce") if open_col else None,
        "high": pd.to_numeric(df[high_col], errors="coerce") if high_col else None,
        "low": pd.to_numeric(df[low_col], errors="coerce") if low_col else None,
        "close": pd.to_numeric(df[close_col], errors="coerce") if close_col else None,
        "volume": pd.to_numeric(df[volume_col], errors="coerce") if volume_col else 0.0,
        "turnover": pd.to_numeric(df[turnover_col], errors="coerce") if turnover_col else 0.0,
        "pct": pd.to_numeric(df[pct_col], errors="coerce") if pct_col else None,
    }).dropna(subset=["date", "close"])

    if out["pct"].isna().all():
        out["pct"] = out["close"].pct_change() * 100

    return out.sort_values("date").reset_index(drop=True)


def _norm_intraday_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["dt", "price", "volume", "avg"])

    df = df.copy()
    dt_col = _pick_col(df, ["时间", "date", "datetime"])
    price_col = _pick_col(df, ["收盘", "close", "最新价", "价格"])
    volume_col = _pick_col(df, ["成交量", "volume"])

    out = pd.DataFrame({
        "dt": pd.to_datetime(df[dt_col], errors="coerce") if dt_col else pd.NaT,
        "price": pd.to_numeric(df[price_col], errors="coerce") if price_col else None,
        "volume": pd.to_numeric(df[volume_col], errors="coerce") if volume_col else 0.0,
    }).dropna(subset=["dt", "price"])

    out = out.sort_values("dt").reset_index(drop=True)
    pv = (out["price"] * out["volume"]).cumsum()
    vv = out["volume"].cumsum().replace(0, pd.NA)
    out["avg"] = (pv / vv).fillna(out["price"])
    return out


def _probability_model(daily_df: pd.DataFrame) -> Dict:
    if len(daily_df) < 30:
        return {
            "p_up_today": 0.50,
            "p_up_5d": 0.50,
            "p_up_long": 0.50,
            "reasons": ["历史样本较少，采用中性先验概率。"],
        }

    d = daily_df.copy()
    d["ret1"] = d["close"].pct_change()
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["vol5"] = d["volume"].rolling(5).mean()
    d["vol20"] = d["volume"].rolling(20).mean()

    last = d.iloc[-1]
    momentum = (last["close"] / d["close"].iloc[-6]) - 1 if len(d) >= 6 else 0
    ma_bias = ((last["ma5"] - last["ma20"]) / last["ma20"]) if pd.notna(last["ma5"]) and pd.notna(last["ma20"]) and last["ma20"] else 0
    vol_ratio = (last["vol5"] / last["vol20"]) if pd.notna(last["vol5"]) and pd.notna(last["vol20"]) and last["vol20"] else 1
    recent_win = (d["ret1"].tail(10) > 0).mean()

    score_today = 0.35 * momentum + 0.45 * ma_bias + 0.15 * (vol_ratio - 1) + 0.25 * (recent_win - 0.5)
    score_5d = 0.45 * momentum + 0.55 * ma_bias + 0.20 * (vol_ratio - 1) + 0.35 * (recent_win - 0.5)
    score_long = 0.25 * momentum + 0.70 * ma_bias + 0.10 * (vol_ratio - 1) + 0.20 * (recent_win - 0.5)

    sigmoid = lambda x: 1 / (1 + pow(2.718281828, -5 * x))

    p_today = float(max(0.05, min(0.95, sigmoid(score_today))))
    p_5d = float(max(0.05, min(0.95, sigmoid(score_5d))))
    p_long = float(max(0.05, min(0.95, sigmoid(score_long))))

    reasons = []
    reasons.append(f"短期动量（近5日）为 {momentum * 100:.2f}%")
    reasons.append(f"均线结构（MA5-MA20）偏离 {ma_bias * 100:.2f}%")
    reasons.append(f"量能比（VOL5/VOL20）为 {vol_ratio:.2f}")
    reasons.append(f"近10日上涨胜率 {recent_win * 100:.1f}%")

    return {
        "p_up_today": p_today,
        "p_up_5d": p_5d,
        "p_up_long": p_long,
        "reasons": reasons,
        "features": {
            "momentum_5d": round(momentum * 100, 2),
            "ma_bias_pct": round(ma_bias * 100, 2),
            "vol_ratio": round(vol_ratio, 2),
            "win_rate_10d": round(recent_win * 100, 1),
        },
    }


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


@app.route("/api/market/overview")
def market_overview():
    symbol = (request.args.get("symbol") or "000001").strip()
    days = int(request.args.get("days", "60"))
    days = max(20, min(days, 250))

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days * 2 + 20)).strftime("%Y%m%d")

    try:
        daily_raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        daily = _norm_daily_df(daily_raw).tail(days)
        if daily.empty:
            return jsonify({"error": "未获取到日线数据"}), 400

        try:
            intraday_raw = ak.stock_zh_a_hist_min_em(symbol=symbol, period="1", adjust="")
        except Exception:
            intraday_raw = pd.DataFrame()

        intraday = _norm_intraday_df(intraday_raw)

        model_result = _probability_model(daily)
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

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days * 2 + 20)).strftime("%Y%m%d")

    daily_raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
    daily = _norm_daily_df(daily_raw).tail(days)
    if daily.empty:
        return jsonify({"error": "未获取到日线数据"}), 400

    try:
        intraday_raw = ak.stock_zh_a_hist_min_em(symbol=symbol, period="1", adjust="")
    except Exception:
        intraday_raw = pd.DataFrame()

    intraday = _norm_intraday_df(intraday_raw)
    model_result = _probability_model(daily)

    analysis, err = _run_market_codex_reason(symbol, daily, intraday, model_result)
    if err:
        return jsonify({"error": err}), 500

    return jsonify({"symbol": symbol, "analysis": analysis})


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _calc_symbol_snapshot(symbol: str, days: int = 60) -> Dict:
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days * 2 + 20)).strftime("%Y%m%d")
    daily_raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
    daily = _norm_daily_df(daily_raw).tail(days)
    if daily.empty:
        return {}

    pred = _probability_model(daily)
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
    }


@app.route("/api/market/sectors")
def market_sectors():
    try:
        df = ak.stock_board_industry_name_em()
        col = _pick_col(df, ["板块名称", "板块", "name"])
        if not col:
            return jsonify({"error": "行业列表字段识别失败"}), 500
        names = sorted([str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()])
        return jsonify({"sectors": names})
    except Exception as e:
        return jsonify({"error": f"获取行业列表失败: {str(e)}"}), 500


@app.route("/api/market/scan")
def market_scan():
    mode = (request.args.get("mode") or "industry").strip()  # industry | all
    industry = (request.args.get("industry") or "").strip()
    sort_by = (request.args.get("sort_by") or "next_5d_up").strip()
    days = max(20, min(int(request.args.get("days", "60")), 250))
    limit = max(1, min(int(request.args.get("limit", "30")), 400))

    allowed_sort = {"today_up", "next_5d_up", "long_up", "pct", "turnover"}
    if sort_by not in allowed_sort:
        sort_by = "next_5d_up"

    try:
        symbols = []

        if mode == "industry":
            if not industry:
                return jsonify({"error": "industry 模式需要 industry 参数"}), 400
            cons = ak.stock_board_industry_cons_em(symbol=industry)
            code_col = _pick_col(cons, ["代码", "股票代码", "symbol"])
            if not code_col:
                return jsonify({"error": "行业成分股字段识别失败"}), 500
            symbols = [str(x).zfill(6) for x in cons[code_col].dropna().tolist()]

        elif mode == "all":
            spot = ak.stock_zh_a_spot_em()
            code_col = _pick_col(spot, ["代码", "symbol"])
            turnover_col = _pick_col(spot, ["成交额", "amount", "turnover"])
            name_col = _pick_col(spot, ["名称", "name"])
            if not code_col:
                return jsonify({"error": "全市场代码字段识别失败"}), 500

            if turnover_col:
                spot = spot.copy()
                spot["_turnover"] = pd.to_numeric(spot[turnover_col], errors="coerce").fillna(0)
                spot = spot.sort_values("_turnover", ascending=False)

            symbols = [str(x).zfill(6) for x in spot[code_col].dropna().tolist()][:limit]
            names_map = {str(r[code_col]).zfill(6): str(r[name_col]) for _, r in spot.iterrows()} if name_col else {}
        else:
            return jsonify({"error": "mode 仅支持 industry 或 all"}), 400

        if mode == "industry":
            symbols = symbols[:limit]
            names_map = {}

        rows = []
        failed = []
        for s in symbols:
            try:
                item = _calc_symbol_snapshot(s, days=days)
                if not item:
                    failed.append(s)
                    continue
                if s in names_map:
                    item["name"] = names_map[s]
                rows.append(item)
            except Exception:
                failed.append(s)

        rows = sorted(rows, key=lambda x: x.get(sort_by, 0), reverse=True)

        return jsonify({
            "mode": mode,
            "industry": industry if mode == "industry" else "全A股",
            "days": days,
            "sort_by": sort_by,
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

    min_history = max(30, min(int(request.args.get("min_history", "120")), 400))
    long_horizon = max(20, min(int(request.args.get("long_horizon", "60")), 120))

    try:
        daily_raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        daily = _norm_daily_df(daily_raw)
        if len(daily) < min_history + long_horizon + 10:
            return jsonify({"error": "历史样本不足，无法回测。请扩大时间区间。"}), 400

        daily = daily.set_index("date")

        rows = []
        for i in range(min_history, len(daily) - long_horizon):
            hist = daily.iloc[: i + 1].reset_index()
            pred = _probability_model(hist)

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
