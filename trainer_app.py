from datetime import datetime
import os
import sys
import random
import uuid
import subprocess
import threading

from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
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


def _fetch_event_context(symbol: str, max_items: int = 5):
    """抓取个股近期新闻作为事件上下文。失败时返回空列表。"""
    try:
        news = ak.stock_news_em(symbol=symbol)
        if news is None or news.empty:
            return []
        news = news.head(max_items)
        items = []
        for _, r in news.iterrows():
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

    i = random.randint(hist_len + 20, len(df) - pred_days - 1)
    hist = df.iloc[i - hist_len:i + 1].copy()

    current_close = float(df["close"].iloc[i])
    future_close = float(df["close"].iloc[i + pred_days])
    ret = (future_close / current_close) - 1

    truth_direction = "上涨" if ret >= 0 else "下跌"
    truth_trend = _trend_label(df, i)

    events = _fetch_event_context(symbol, max_items=5)

    cid = str(uuid.uuid4())
    CHALLENGES[cid] = {
        "symbol": symbol,
        "anchor_date": str(df.index[i].date()),
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
        "anchor_date": str(df.index[i].date()),
        "pred_days": pred_days,
        "level": level,
        "candles": CHALLENGES[cid]["candles"],
        "events": CHALLENGES[cid]["events"],
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8787, debug=True)
