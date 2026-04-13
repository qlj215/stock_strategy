# -*- coding: utf-8 -*-
"""
假想股票组合收益试算网页工具。

运行方式：
    python portfolio_whatif_web.py

启动后会自动打开浏览器，在本地网页中输入：
1. 假想股票组合
2. 买入和卖出时间
3. 仓位模式（按金额 / 按手数）
4. 时间模式（按天 / 精确到秒）

然后程序会自动：
- 从 MiniQMT + xtdata 读取价格
- 计算买入成本、卖出回款
- 计算组合收益金额与收益率

当前口径说明：
1. 按天模式：使用日线收盘价；若输入日期不是交易日，会自动匹配最近可用交易日。
2. 精确到秒模式：前端允许输入到秒，但底层历史行情基于 1 分钟数据，
   会匹配“请求时刻附近的可用分钟价”（优先不晚于请求时刻的最近分钟）。
3. 手续费口径：
   - 佣金：万三（0.03%），单笔最低 5 元
   - 过户费：十万分之一（0.001%）
   - 卖出额外计收印花税：千分之 0.5（0.05%）
4. 金额模式下，输入金额按“总预算”理解，程序会自动向下取整到可买的整手数量。
"""

from __future__ import annotations

import argparse
import math
import threading
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from data import fetcher as fetcher_mod
from data.fetcher import fetch_stock_data, get_symbol_name


COMMISSION_RATE = 0.0003
TRANSFER_FEE_RATE = 0.00001
STAMP_DUTY_RATE = 0.0005
MIN_COMMISSION = 5.0
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

app = Flask(__name__)

HTML_PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>假想股票组合收益试算</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f7f8fa; margin:0; color:#222; }
    .wrap { max-width: 1180px; margin: 24px auto; padding: 0 16px 40px; }
    .card { background:#fff; border-radius:16px; padding:20px; box-shadow:0 6px 24px rgba(0,0,0,.06); margin-bottom:18px; }
    h1,h2,h3 { margin: 0 0 12px; }
    p { line-height: 1.7; }
    .muted { color:#666; font-size:14px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
    label { display:block; font-weight:600; margin-bottom:6px; }
    input, select, button, textarea { width:100%; box-sizing:border-box; padding:10px 12px; border:1px solid #d8dce5; border-radius:10px; font-size:14px; }
    input[type="radio"] { width:auto; }
    button { background:#2563eb; color:#fff; border:none; cursor:pointer; font-weight:600; }
    button.secondary { background:#edf2ff; color:#2949c7; }
    button.danger { background:#fef2f2; color:#c53030; }
    .inline { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .inline label { margin:0; font-weight:500; }
    table { width:100%; border-collapse: collapse; margin-top:12px; font-size:14px; }
    th, td { border-bottom:1px solid #edf0f5; padding:10px 8px; text-align:left; vertical-align:top; }
    th { background:#fafbfd; }
    .stock-row { display:grid; grid-template-columns: 1.2fr 1fr auto; gap:10px; margin-bottom:10px; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#edf2ff; color:#2949c7; font-size:12px; margin-right:8px; }
    .ok { color:#067647; }
    .bad { color:#b42318; }
    .summary-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; }
    .stat { background:#fafbfd; border:1px solid #edf0f5; border-radius:12px; padding:12px; }
    .stat .k { color:#667085; font-size:13px; }
    .stat .v { font-size:20px; font-weight:700; margin-top:6px; }
    #loading { display:none; color:#2949c7; font-weight:600; }
    #error { display:none; color:#b42318; white-space:pre-wrap; }
    .hint { background:#f8fafc; border-left:4px solid #2563eb; padding:12px; border-radius:8px; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>假想股票组合收益试算</h1>
    <p class="muted">这个小网页是用来做“如果我当时这样买、这样卖，最后赚了多少钱”的快速估算。支持按金额建仓，也支持按手数建仓。</p>
    <div class="hint">
      <div><span class="pill">费用口径</span>佣金按万三，单笔最低 5 元；过户费按十万分之一；卖出额外收印花税 0.05%。</div>
      <div style="margin-top:8px;"><span class="pill">时间口径</span>按天模式使用日线收盘价。精确到秒模式底层使用 1 分钟数据，所以会匹配最近可用分钟价，不是真正逐秒成交价。</div>
    </div>
  </div>

  <div class="card">
    <h2>1. 输入组合</h2>

    <div class="grid" style="margin-bottom:14px;">
      <div>
        <label>仓位模式</label>
        <div class="inline">
          <label><input type="radio" name="position_mode" value="amount" checked> 按金额</label>
          <label><input type="radio" name="position_mode" value="lots"> 按手数</label>
        </div>
      </div>
      <div>
        <label>时间模式</label>
        <div class="inline">
          <label><input type="radio" name="time_mode" value="day" checked> 按天</label>
          <label><input type="radio" name="time_mode" value="second"> 精确到秒</label>
        </div>
      </div>
    </div>

    <div class="grid" style="margin-bottom:16px;">
      <div id="buy_day_wrap">
        <label>买入日期</label>
        <input id="buy_day" type="date">
      </div>
      <div id="sell_day_wrap">
        <label>卖出日期</label>
        <input id="sell_day" type="date">
      </div>
      <div id="buy_second_wrap" style="display:none;">
        <label>买入时间（到秒）</label>
        <input id="buy_second" type="datetime-local" step="1">
      </div>
      <div id="sell_second_wrap" style="display:none;">
        <label>卖出时间（到秒）</label>
        <input id="sell_second" type="datetime-local" step="1">
      </div>
    </div>

    <h3>组合持仓</h3>
    <p class="muted">按金额模式下，这里的数值表示每只股票的总预算。按手数模式下，表示买入多少手（1 手 = 100 股）。</p>
    <div id="stock_rows"></div>
    <div class="inline" style="margin-top:12px;">
      <button type="button" class="secondary" onclick="addRow()" style="width:auto;">+ 增加一只股票</button>
      <button type="button" onclick="calculatePortfolio()" style="width:auto;">开始计算</button>
      <span id="loading">正在计算，请稍等...</span>
    </div>
    <div id="error" style="margin-top:12px;"></div>
  </div>

  <div class="card" id="result_card" style="display:none;">
    <h2>2. 计算结果</h2>
    <div id="summary"></div>
    <div id="detail"></div>
  </div>
</div>

<script>
  function todayStr() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  }

  function nowLocalStr(offsetMinutes) {
    const d = new Date(Date.now() + offsetMinutes * 60000);
    const pad = (n) => String(n).padStart(2, '0');
    const yyyy = d.getFullYear();
    const mm = pad(d.getMonth() + 1);
    const dd = pad(d.getDate());
    const hh = pad(d.getHours());
    const mi = pad(d.getMinutes());
    const ss = pad(d.getSeconds());
    return `${yyyy}-${mm}-${dd}T${hh}:${mi}:${ss}`;
  }

  function setDefaultTimes() {
    document.getElementById('buy_day').value = todayStr();
    document.getElementById('sell_day').value = todayStr();
    document.getElementById('buy_second').value = nowLocalStr(-60);
    document.getElementById('sell_second').value = nowLocalStr(0);
  }

  function addRow(symbol = '', value = '') {
    const rows = document.getElementById('stock_rows');
    const div = document.createElement('div');
    div.className = 'stock-row';
    div.innerHTML = `
      <input class="symbol" placeholder="股票代码，例如 600519" value="${symbol}">
      <input class="position-value" type="number" min="0" step="0.01" placeholder="金额或手数" value="${value}">
      <button type="button" class="danger" onclick="this.parentElement.remove()" style="width:auto;">删除</button>
    `;
    rows.appendChild(div);
  }

  function getRadio(name) {
    return document.querySelector(`input[name="${name}"]:checked`).value;
  }

  function refreshTimeMode() {
    const timeMode = getRadio('time_mode');
    const showDay = timeMode === 'day';
    document.getElementById('buy_day_wrap').style.display = showDay ? 'block' : 'none';
    document.getElementById('sell_day_wrap').style.display = showDay ? 'block' : 'none';
    document.getElementById('buy_second_wrap').style.display = showDay ? 'none' : 'block';
    document.getElementById('sell_second_wrap').style.display = showDay ? 'none' : 'block';
  }

  function fmtMoney(x) {
    const n = Number(x || 0);
    return n.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtPct(x) {
    const n = Number(x || 0) * 100;
    return `${n.toFixed(2)}%`;
  }

  async function calculatePortfolio() {
    const errorBox = document.getElementById('error');
    const loading = document.getElementById('loading');
    const resultCard = document.getElementById('result_card');
    errorBox.style.display = 'none';
    resultCard.style.display = 'none';
    loading.style.display = 'inline';

    const positionMode = getRadio('position_mode');
    const timeMode = getRadio('time_mode');
    const positions = Array.from(document.querySelectorAll('.stock-row')).map(row => ({
      symbol: row.querySelector('.symbol').value.trim(),
      value: row.querySelector('.position-value').value.trim(),
    })).filter(item => item.symbol && item.value);

    const payload = {
      position_mode: positionMode,
      time_mode: timeMode,
      buy_time: timeMode === 'day' ? document.getElementById('buy_day').value : document.getElementById('buy_second').value,
      sell_time: timeMode === 'day' ? document.getElementById('sell_day').value : document.getElementById('sell_second').value,
      positions: positions,
    };

    try {
      const resp = await fetch('/api/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok || !data.success) {
        throw new Error(data.error || '计算失败');
      }
      renderResult(data.result);
      resultCard.style.display = 'block';
    } catch (err) {
      errorBox.textContent = err.message || String(err);
      errorBox.style.display = 'block';
    } finally {
      loading.style.display = 'none';
    }
  }

  function renderResult(result) {
    const summary = result.summary;
    document.getElementById('summary').innerHTML = `
      <div class="summary-grid">
        <div class="stat"><div class="k">组合买入总成本</div><div class="v">¥ ${fmtMoney(summary.total_buy_cost)}</div></div>
        <div class="stat"><div class="k">组合卖出净回款</div><div class="v">¥ ${fmtMoney(summary.total_sell_net_amount)}</div></div>
        <div class="stat"><div class="k">总收益金额</div><div class="v ${summary.total_profit >= 0 ? 'ok' : 'bad'}">¥ ${fmtMoney(summary.total_profit)}</div></div>
        <div class="stat"><div class="k">总收益率</div><div class="v ${summary.total_return_rate >= 0 ? 'ok' : 'bad'}">${fmtPct(summary.total_return_rate)}</div></div>
        <div class="stat"><div class="k">买入总费用</div><div class="v">¥ ${fmtMoney(summary.total_buy_fees)}</div></div>
        <div class="stat"><div class="k">卖出总费用</div><div class="v">¥ ${fmtMoney(summary.total_sell_fees)}</div></div>
        <div class="stat"><div class="k">金额模式剩余现金</div><div class="v">¥ ${fmtMoney(summary.total_leftover_cash)}</div></div>
      </div>
      <p class="muted" style="margin-top:14px;">说明：金额模式下，输入的是预算，程序会自动向下取整到可买的整手数量。按天模式使用日线收盘价，按秒模式匹配的是最近可用分钟价。</p>
    `;

    const rows = result.details.map(item => `
      <tr>
        <td>${item.symbol}<br><span class="muted">${item.name || ''}</span></td>
        <td>${item.input_mode_label}<br><span class="muted">${item.input_value_label}${item.leftover_cash > 0 ? `，剩余 ¥ ${fmtMoney(item.leftover_cash)}` : ''}</span></td>
        <td>${item.shares} 股<br><span class="muted">${item.lots} 手</span></td>
        <td>¥ ${fmtMoney(item.buy_price)}<br><span class="muted">${item.matched_buy_time}</span></td>
        <td>¥ ${fmtMoney(item.sell_price)}<br><span class="muted">${item.matched_sell_time}</span></td>
        <td>¥ ${fmtMoney(item.buy_total_cost)}<br><span class="muted">买入费 ¥ ${fmtMoney(item.buy_total_fees)}</span></td>
        <td>¥ ${fmtMoney(item.sell_net_amount)}<br><span class="muted">卖出费 ¥ ${fmtMoney(item.sell_total_fees)}</span></td>
        <td class="${item.profit >= 0 ? 'ok' : 'bad'}">¥ ${fmtMoney(item.profit)}</td>
        <td class="${item.return_rate >= 0 ? 'ok' : 'bad'}">${fmtPct(item.return_rate)}</td>
      </tr>
    `).join('');

    document.getElementById('detail').innerHTML = `
      <h3 style="margin-top:20px;">逐只股票明细</h3>
      <table>
        <thead>
          <tr>
            <th>股票</th>
            <th>输入方式</th>
            <th>实际买入数量</th>
            <th>买入价格</th>
            <th>卖出价格</th>
            <th>买入总成本</th>
            <th>卖出净回款</th>
            <th>收益金额</th>
            <th>收益率</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  document.querySelectorAll('input[name="time_mode"]').forEach(el => el.addEventListener('change', refreshTimeMode));
  setDefaultTimes();
  refreshTimeMode();
  addRow('600519', '10000');
  addRow('000001', '8000');
</script>
</body>
</html>
"""


@dataclass
class TradeResult:
    symbol: str
    name: str
    input_mode_label: str
    input_value_label: str
    lots: int
    shares: int
    buy_price: float
    sell_price: float
    matched_buy_time: str
    matched_sell_time: str
    buy_trade_amount: float
    buy_total_fees: float
    buy_total_cost: float
    sell_trade_amount: float
    sell_total_fees: float
    sell_net_amount: float
    leftover_cash: float
    profit: float
    return_rate: float


def _normalize_symbol(symbol: str) -> str:
    helper = getattr(fetcher_mod, "_to_plain_symbol", None)
    if callable(helper):
        return helper(symbol)
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    if not digits:
        raise ValueError(f"无法识别股票代码：{symbol}")
    return digits.zfill(6)


def _parse_input_time(raw: str, mode: str) -> datetime:
    if not raw:
        raise ValueError("买入时间和卖出时间不能为空")
    if mode == "day":
        return datetime.strptime(raw.strip(), "%Y-%m-%d")
    cleaned = raw.strip().replace("T", " ")
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(cleaned, "%Y-%m-%d %H:%M")


def _calc_commission(trade_amount: float) -> float:
    if trade_amount <= 0:
        return 0.0
    return max(trade_amount * COMMISSION_RATE, MIN_COMMISSION)


def _calc_buy_fees(trade_amount: float) -> Dict[str, float]:
    commission = _calc_commission(trade_amount)
    transfer_fee = trade_amount * TRANSFER_FEE_RATE
    total = commission + transfer_fee
    return {
        "commission": commission,
        "transfer_fee": transfer_fee,
        "stamp_duty": 0.0,
        "total": total,
    }


def _calc_sell_fees(trade_amount: float) -> Dict[str, float]:
    commission = _calc_commission(trade_amount)
    transfer_fee = trade_amount * TRANSFER_FEE_RATE
    stamp_duty = trade_amount * STAMP_DUTY_RATE
    total = commission + transfer_fee + stamp_duty
    return {
        "commission": commission,
        "transfer_fee": transfer_fee,
        "stamp_duty": stamp_duty,
        "total": total,
    }


def _max_lots_by_budget(budget: float, buy_price: float) -> int:
    if budget <= 0:
        raise ValueError("预算金额必须大于 0")
    if buy_price <= 0:
        raise ValueError("买入价格必须大于 0")

    lo, hi = 0, int(budget // (buy_price * 100)) + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        trade_amount = mid * 100 * buy_price
        total_cost = trade_amount + _calc_buy_fees(trade_amount)["total"]
        if total_cost <= budget + 1e-9:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _pick_daily_price(df: pd.DataFrame, requested: datetime, side: str) -> Tuple[float, str]:
    if df is None or df.empty:
        raise ValueError("未获取到日线数据")

    series = df.copy().sort_index()
    series.index = pd.to_datetime(series.index)
    if side == "buy":
        picked = series[series.index.date >= requested.date()]
        if picked.empty:
            raise ValueError(f"{requested.date()} 之后没有可用买入日线")
        row = picked.iloc[0]
        dt = picked.index[0]
    else:
        picked = series[series.index.date <= requested.date()]
        if picked.empty:
            raise ValueError(f"{requested.date()} 之前没有可用卖出日线")
        row = picked.iloc[-1]
        dt = picked.index[-1]

    return float(row["close"]), pd.Timestamp(dt).strftime("%Y-%m-%d")


def _fetch_intraday_range(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    xtdata = fetcher_mod._load_xtdata()
    xt_symbol = fetcher_mod._to_xt_symbol(symbol)

    start_time = start_dt.strftime("%Y%m%d%H%M%S")
    end_time = end_dt.strftime("%Y%m%d%H%M%S")

    xtdata.download_history_data(
        xt_symbol,
        period="1m",
        start_time=start_time,
        end_time=end_time,
        incrementally=True,
    )
    data = xtdata.get_market_data_ex(
        ["time", "open", "high", "low", "close", "volume", "amount"],
        [xt_symbol],
        period="1m",
        start_time=start_time,
        end_time=end_time,
        dividend_type="none",
    )

    raw_df = data.get(xt_symbol)
    df = fetcher_mod._normalize_xt_kline_df(raw_df)
    out = df.reset_index().rename(columns={"date": "dt", "close": "price"})
    out["dt"] = pd.to_datetime(out["dt"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["dt", "price"]).sort_values("dt").reset_index(drop=True)
    return out[["dt", "price"]]


def _pick_intraday_price(df: pd.DataFrame, requested: datetime, side: str) -> Tuple[float, str]:
    if df is None or df.empty:
        raise ValueError("未获取到分钟级数据")

    ts = pd.to_datetime(requested)
    df = df.copy().sort_values("dt").reset_index(drop=True)
    before_or_equal = df[df["dt"] <= ts]

    if not before_or_equal.empty:
        row = before_or_equal.iloc[-1]
    else:
        after = df[df["dt"] > ts]
        if after.empty:
            raise ValueError("请求时间附近没有可用分钟数据")
        row = after.iloc[0]

    matched_dt = pd.Timestamp(row["dt"])
    return float(row["price"]), matched_dt.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_prices(symbol: str, buy_time: datetime, sell_time: datetime, time_mode: str) -> Dict[str, object]:
    if time_mode == "day":
        start_date = buy_time.strftime("%Y%m%d")
        end_date = sell_time.strftime("%Y%m%d")
        df = fetch_stock_data(symbol, start_date=start_date, end_date=end_date)
        buy_price, matched_buy_time = _pick_daily_price(df, buy_time, side="buy")
        sell_price, matched_sell_time = _pick_daily_price(df, sell_time, side="sell")
        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "matched_buy_time": matched_buy_time,
            "matched_sell_time": matched_sell_time,
        }

    start_dt = buy_time - timedelta(minutes=2)
    end_dt = sell_time + timedelta(minutes=2)
    df = _fetch_intraday_range(symbol, start_dt=start_dt, end_dt=end_dt)
    buy_price, matched_buy_time = _pick_intraday_price(df, buy_time, side="buy")
    sell_price, matched_sell_time = _pick_intraday_price(df, sell_time, side="sell")
    return {
        "buy_price": buy_price,
        "sell_price": sell_price,
        "matched_buy_time": matched_buy_time,
        "matched_sell_time": matched_sell_time,
    }


def _build_trade_result(position: Dict[str, str], position_mode: str, time_mode: str, buy_time: datetime, sell_time: datetime) -> TradeResult:
    symbol = _normalize_symbol(position.get("symbol", ""))
    raw_value = position.get("value", "")
    try:
        input_value = float(raw_value)
    except Exception as e:
        raise ValueError(f"股票 {symbol} 的数量/金额无效：{raw_value}") from e

    if input_value <= 0:
        raise ValueError(f"股票 {symbol} 的数量/金额必须大于 0")

    prices = _resolve_prices(symbol, buy_time=buy_time, sell_time=sell_time, time_mode=time_mode)
    buy_price = float(prices["buy_price"])
    sell_price = float(prices["sell_price"])

    if position_mode == "amount":
        lots = _max_lots_by_budget(input_value, buy_price)
        if lots <= 0:
            raise ValueError(
                f"股票 {symbol} 在买入价 {buy_price:.2f} 下，预算 {input_value:.2f} 元不足以买入 1 手（100 股）"
            )
        input_mode_label = "按金额"
        input_value_label = f"预算 ¥ {input_value:,.2f}"
    else:
        lots = int(input_value)
        if not math.isclose(float(lots), input_value, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"股票 {symbol} 的手数必须是整数")
        if lots <= 0:
            raise ValueError(f"股票 {symbol} 的手数必须大于 0")
        input_mode_label = "按手数"
        input_value_label = f"{lots} 手"

    shares = lots * 100
    buy_trade_amount = shares * buy_price
    buy_fee_info = _calc_buy_fees(buy_trade_amount)
    buy_total_cost = buy_trade_amount + buy_fee_info["total"]

    if position_mode == "amount":
        leftover_cash = max(input_value - buy_total_cost, 0.0)
    else:
        leftover_cash = 0.0

    sell_trade_amount = shares * sell_price
    sell_fee_info = _calc_sell_fees(sell_trade_amount)
    sell_net_amount = sell_trade_amount - sell_fee_info["total"]

    profit = sell_net_amount - buy_total_cost
    return_rate = profit / buy_total_cost if buy_total_cost > 0 else 0.0

    return TradeResult(
        symbol=symbol,
        name=get_symbol_name(symbol),
        input_mode_label=input_mode_label,
        input_value_label=input_value_label,
        lots=lots,
        shares=shares,
        buy_price=buy_price,
        sell_price=sell_price,
        matched_buy_time=str(prices["matched_buy_time"]),
        matched_sell_time=str(prices["matched_sell_time"]),
        buy_trade_amount=buy_trade_amount,
        buy_total_fees=buy_fee_info["total"],
        buy_total_cost=buy_total_cost,
        sell_trade_amount=sell_trade_amount,
        sell_total_fees=sell_fee_info["total"],
        sell_net_amount=sell_net_amount,
        leftover_cash=leftover_cash,
        profit=profit,
        return_rate=return_rate,
    )


def calculate_portfolio(payload: Dict[str, object]) -> Dict[str, object]:
    position_mode = str(payload.get("position_mode") or "amount").strip().lower()
    time_mode = str(payload.get("time_mode") or "day").strip().lower()
    if position_mode not in {"amount", "lots"}:
        raise ValueError("position_mode 只能是 amount 或 lots")
    if time_mode not in {"day", "second"}:
        raise ValueError("time_mode 只能是 day 或 second")

    positions = payload.get("positions") or []
    if not isinstance(positions, list) or not positions:
        raise ValueError("请至少输入一只股票")

    buy_time = _parse_input_time(str(payload.get("buy_time") or ""), mode=time_mode)
    sell_time = _parse_input_time(str(payload.get("sell_time") or ""), mode=time_mode)
    if sell_time < buy_time:
        raise ValueError("卖出时间不能早于买入时间")

    results: List[TradeResult] = []
    for position in positions:
        if not isinstance(position, dict):
            raise ValueError("positions 中存在非法项")
        results.append(
            _build_trade_result(
                position=position,
                position_mode=position_mode,
                time_mode=time_mode,
                buy_time=buy_time,
                sell_time=sell_time,
            )
        )

    total_buy_cost = sum(x.buy_total_cost for x in results)
    total_sell_net_amount = sum(x.sell_net_amount for x in results)
    total_profit = total_sell_net_amount - total_buy_cost
    total_return_rate = total_profit / total_buy_cost if total_buy_cost > 0 else 0.0

    return {
        "summary": {
            "position_mode": position_mode,
            "time_mode": time_mode,
            "buy_time": buy_time.strftime("%Y-%m-%d %H:%M:%S") if time_mode == "second" else buy_time.strftime("%Y-%m-%d"),
            "sell_time": sell_time.strftime("%Y-%m-%d %H:%M:%S") if time_mode == "second" else sell_time.strftime("%Y-%m-%d"),
            "stock_count": len(results),
            "total_buy_cost": total_buy_cost,
            "total_sell_net_amount": total_sell_net_amount,
            "total_profit": total_profit,
            "total_return_rate": total_return_rate,
            "total_buy_fees": sum(x.buy_total_fees for x in results),
            "total_sell_fees": sum(x.sell_total_fees for x in results),
            "total_leftover_cash": sum(x.leftover_cash for x in results),
        },
        "details": [asdict(x) for x in results],
    }


@app.get("/")
def index():
    return render_template_string(HTML_PAGE)


@app.post("/api/calculate")
def api_calculate():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        result = calculate_portfolio(payload)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="假想股票组合收益试算网页工具")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址，默认 {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口，默认 {DEFAULT_PORT}")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"组合收益试算页面已启动：{url}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
