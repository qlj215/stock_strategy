# -*- coding: utf-8 -*-
"""
阶段6.0 组合回测系统（独立链路打通版）

用途：
1) 将阶段4/5/5.2等模型输出的横截面分数，统一转换为可比较的组合回测结果；
2) 先打通“模型分数 -> 组合构建 -> 成本扣减 -> 净值/风险指标 -> 分层检验”链路；
3) 尽量与主线页面/旧回测模块解耦，作为阶段6首版独立研究脚本。

重要说明：
- 本版优先解决“链路打通”，不是最终实盘级回测引擎；
- 默认使用阶段3特征表中的 `label_fwd_ret_5d` 作为未来5日收益口径；
- 推荐把 `--rebalance-every` 设为 5，与 5 日标签周期保持一致，避免重叠收益造成解释混乱；
- 后续阶段6.1/6.2 可继续升级为更贴近实盘的 T+1 开盘撮合、涨跌停/停牌约束、行业中性化等版本。

快速使用示例：
python portfolio_backtest.py \
  --signal-config data/dl/dl_v52_result.csv::pred_v52::stage5_2_selector \
  --signal-config data/baseline/baseline_v41_result.csv::pred_tree::stage4_1_tree \
  --signal-config data/baseline/baseline_v41_result.csv::pred_blend::stage4_1_blend \
  --features-file data/features/features_v1.parquet \
  --split test \
  --rebalance-every 5 \
  --top-quantile 0.2 \
  --out-dir data/backtest/stage6_0
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = "data/backtest/stage6_0"


@dataclass
class SignalConfig:
    path: str
    score_col: str
    alias: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段6.0 组合回测系统（独立链路打通版）")
    parser.add_argument(
        "--signal-config",
        action="append",
        required=True,
        help="信号配置，格式：<csv_path>::<score_col>::<alias>，可重复传入多次",
    )
    parser.add_argument("--features-file", default="data/features/features_v1.parquet", help="阶段3特征表")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="回测样本区间")
    parser.add_argument("--date-start", default="", help="可选：起始日期 YYYY-MM-DD")
    parser.add_argument("--date-end", default="", help="可选：结束日期 YYYY-MM-DD")
    parser.add_argument("--label-col", default="label_fwd_ret_5d", help="组合收益口径列")
    parser.add_argument("--bench-col", default="label_bench_ret_5d", help="基准收益列")
    parser.add_argument("--rebalance-every", type=int, default=5, help="调仓频率（交易日）")
    parser.add_argument("--top-n", type=int, default=0, help="每期选前 N 只，>0 时优先生效")
    parser.add_argument("--top-quantile", type=float, default=0.2, help="每期选前 q 分位（默认 0.2）")
    parser.add_argument("--group-count", type=int, default=5, help="分层组数")
    parser.add_argument("--init-nav", type=float, default=1.0, help="初始净值")
    parser.add_argument("--buy-cost-rate", type=float, default=0.0008, help="买入侧成本率（佣金+滑点等）")
    parser.add_argument("--sell-cost-rate", type=float, default=0.0018, help="卖出侧成本率（佣金+滑点+印花税等）")
    parser.add_argument("--min-candidates", type=int, default=5, help="单期最少候选股票数")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    return parser.parse_args()


def _normalize_stock_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)", expand=False).fillna(series.astype(str)).str.zfill(6)


def _parse_signal_config(spec: str) -> SignalConfig:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(f"signal-config 格式错误：{spec}，应为 <path>::<score_col>::<alias>")
    path, score_col, alias = [x.strip() for x in parts]
    if not path or not score_col or not alias:
        raise ValueError(f"signal-config 缺少必要字段：{spec}")
    return SignalConfig(path=path, score_col=score_col, alias=alias)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_features(path: str, label_col: str, bench_col: str) -> pd.DataFrame:
    need_cols = ["date", "stock_code", "industry", "is_trading", label_col, bench_col, "label_excess_ret_5d"]
    df = pd.read_parquet(path, columns=need_cols)
    df["date"] = pd.to_datetime(df["date"])
    df["stock_code"] = _normalize_stock_code(df["stock_code"])
    return df


def _load_signal_frame(cfg: SignalConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.path)
    if cfg.score_col not in df.columns:
        raise KeyError(f"{cfg.path} 中不存在列 {cfg.score_col}")
    for col in ["date", "stock_code"]:
        if col not in df.columns:
            raise KeyError(f"{cfg.path} 缺少必要列 {col}")

    out = df[["date", "stock_code", cfg.score_col] + (["split"] if "split" in df.columns else [])].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["stock_code"] = _normalize_stock_code(out["stock_code"])
    out = out.rename(columns={cfg.score_col: "score"})
    out["signal_alias"] = cfg.alias
    return out


def _filter_sample(df: pd.DataFrame, split: str, date_start: str, date_end: str) -> pd.DataFrame:
    out = df.copy()
    if split != "all" and "split" in out.columns:
        out = out[out["split"] == split]
    if date_start:
        out = out[out["date"] >= pd.to_datetime(date_start)]
    if date_end:
        out = out[out["date"] <= pd.to_datetime(date_end)]
    return out


def _pick_top_bucket(day_df: pd.DataFrame, top_n: int, top_quantile: float) -> pd.DataFrame:
    day_df = day_df.sort_values("score", ascending=False).reset_index(drop=True)
    if top_n and top_n > 0:
        n_pick = min(top_n, len(day_df))
    else:
        q = min(max(float(top_quantile), 0.01), 1.0)
        n_pick = max(1, int(math.ceil(len(day_df) * q)))
    return day_df.head(n_pick).copy()


def _calc_trade_turnover(prev_w: Dict[str, float], curr_w: Dict[str, float]) -> Tuple[float, float]:
    keys = set(prev_w) | set(curr_w)
    buy_turnover = 0.0
    sell_turnover = 0.0
    for key in keys:
        delta = curr_w.get(key, 0.0) - prev_w.get(key, 0.0)
        if delta > 0:
            buy_turnover += delta
        elif delta < 0:
            sell_turnover += -delta
    return float(buy_turnover), float(sell_turnover)


def _calc_drawdown(nav: pd.Series) -> pd.Series:
    running_max = nav.cummax()
    return nav / running_max - 1.0


def _annual_factor(rebalance_every: int) -> float:
    return 252.0 / max(int(rebalance_every), 1)


def _calc_metrics(nav_df: pd.DataFrame, rebalance_every: int) -> Dict[str, float]:
    period_factor = _annual_factor(rebalance_every)
    net_ret = nav_df["net_return"].astype(float)
    gross_ret = nav_df["gross_return"].astype(float)
    bench_ret = nav_df["benchmark_return"].astype(float)

    net_nav = nav_df["net_nav"].astype(float)
    gross_nav = nav_df["gross_nav"].astype(float)
    bench_nav = nav_df["benchmark_nav"].astype(float)

    ann_ret = (net_nav.iloc[-1] / net_nav.iloc[0]) ** (period_factor / max(len(net_nav) - 1, 1)) - 1 if len(net_nav) > 1 else 0.0
    ann_vol = net_ret.std(ddof=0) * math.sqrt(period_factor) if len(net_ret) > 1 else 0.0
    sharpe = (net_ret.mean() / net_ret.std(ddof=0) * math.sqrt(period_factor)) if net_ret.std(ddof=0) > 1e-12 else 0.0
    mdd = _calc_drawdown(net_nav).min() if len(net_nav) else 0.0
    bench_ann_ret = (bench_nav.iloc[-1] / bench_nav.iloc[0]) ** (period_factor / max(len(bench_nav) - 1, 1)) - 1 if len(bench_nav) > 1 else 0.0

    return {
        "periods": int(len(nav_df)),
        "gross_total_return": round(float(gross_nav.iloc[-1] / gross_nav.iloc[0] - 1), 6),
        "net_total_return": round(float(net_nav.iloc[-1] / net_nav.iloc[0] - 1), 6),
        "benchmark_total_return": round(float(bench_nav.iloc[-1] / bench_nav.iloc[0] - 1), 6),
        "annual_return": round(float(ann_ret), 6),
        "benchmark_annual_return": round(float(bench_ann_ret), 6),
        "annual_volatility": round(float(ann_vol), 6),
        "sharpe": round(float(sharpe), 6),
        "max_drawdown": round(float(mdd), 6),
        "win_rate": round(float((net_ret > 0).mean()), 6),
        "avg_turnover_buy": round(float(nav_df["buy_turnover"].mean()), 6),
        "avg_turnover_sell": round(float(nav_df["sell_turnover"].mean()), 6),
        "avg_selected": round(float(nav_df["selected_count"].mean()), 3),
        "mean_gross_return": round(float(gross_ret.mean()), 6),
        "mean_net_return": round(float(net_ret.mean()), 6),
        "mean_benchmark_return": round(float(bench_ret.mean()), 6),
        "excess_total_return": round(float(net_nav.iloc[-1] / bench_nav.iloc[-1] - 1), 6),
    }


def _build_group_curve(df: pd.DataFrame, group_count: int, label_col: str, init_nav: float) -> pd.DataFrame:
    group_rows: List[Dict[str, object]] = []
    for dt, day_df in df.groupby("date"):
        day_df = day_df.sort_values("score", ascending=False).copy()
        if len(day_df) < max(2, group_count):
            continue
        try:
            ranks = day_df["score"].rank(method="first", ascending=False)
            day_df["group_id"] = pd.qcut(ranks, q=group_count, labels=False, duplicates="drop")
        except Exception:
            continue
        for gid, gdf in day_df.groupby("group_id"):
            group_rows.append({
                "date": dt,
                "group_id": int(gid) + 1,
                "group_return": float(gdf[label_col].mean()),
                "count": int(len(gdf)),
            })
    group_df = pd.DataFrame(group_rows)
    if group_df.empty:
        return group_df

    pivot = group_df.pivot(index="date", columns="group_id", values="group_return").sort_index()
    pivot.columns = [f"group_{int(c)}" for c in pivot.columns]
    nav = (1.0 + pivot.fillna(0.0)).cumprod() * float(init_nav)
    nav = nav.reset_index()
    return nav


def run_stage6_backtest(
    merged_df: pd.DataFrame,
    alias: str,
    label_col: str,
    bench_col: str,
    rebalance_every: int,
    top_n: int,
    top_quantile: float,
    group_count: int,
    init_nav: float,
    buy_cost_rate: float,
    sell_cost_rate: float,
    min_candidates: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    df = merged_df.copy().sort_values(["date", "score"], ascending=[True, False])
    dates = sorted(df["date"].dropna().unique())
    rebalance_dates = dates[:: max(1, int(rebalance_every))]

    nav_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    prev_weights: Dict[str, float] = {}
    gross_nav = float(init_nav)
    net_nav = float(init_nav)
    benchmark_nav = float(init_nav)

    for dt in rebalance_dates:
        day_df = df[df["date"] == dt].copy()
        day_df = day_df.dropna(subset=["score", label_col, bench_col])
        if "is_trading" in day_df.columns:
            day_df = day_df[day_df["is_trading"] == 1]
        if len(day_df) < max(1, int(min_candidates)):
            continue

        picked = _pick_top_bucket(day_df, top_n=top_n, top_quantile=top_quantile)
        curr_weights = {code: 1.0 / len(picked) for code in picked["stock_code"].tolist()} if len(picked) else {}
        buy_turnover, sell_turnover = _calc_trade_turnover(prev_weights, curr_weights)

        gross_return = float(picked[label_col].mean())
        benchmark_return = float(day_df[bench_col].mean())
        cost_drag = buy_turnover * float(buy_cost_rate) + sell_turnover * float(sell_cost_rate)
        net_return = gross_return - cost_drag

        gross_nav *= (1.0 + gross_return)
        net_nav *= (1.0 + net_return)
        benchmark_nav *= (1.0 + benchmark_return)

        nav_rows.append({
            "date": pd.Timestamp(dt),
            "signal_alias": alias,
            "selected_count": int(len(picked)),
            "gross_return": gross_return,
            "benchmark_return": benchmark_return,
            "cost_drag": cost_drag,
            "net_return": net_return,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "gross_nav": gross_nav,
            "net_nav": net_nav,
            "benchmark_nav": benchmark_nav,
        })

        for _, row in picked.iterrows():
            detail_rows.append({
                "date": pd.Timestamp(dt),
                "signal_alias": alias,
                "stock_code": row["stock_code"],
                "industry": row.get("industry", ""),
                "score": float(row["score"]),
                "future_return": float(row[label_col]),
                "benchmark_return": float(row[bench_col]),
                "excess_return": float(row.get("label_excess_ret_5d", np.nan)),
                "weight": curr_weights.get(row["stock_code"], 0.0),
            })

        prev_weights = curr_weights

    nav_df = pd.DataFrame(nav_rows)
    detail_df = pd.DataFrame(detail_rows)
    group_curve_df = _build_group_curve(df[df["date"].isin(rebalance_dates)], group_count=group_count, label_col=label_col, init_nav=init_nav)
    metrics = _calc_metrics(nav_df, rebalance_every=rebalance_every) if not nav_df.empty else {}
    return nav_df, detail_df, group_curve_df, metrics


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def _render_summary_md(args: argparse.Namespace, metrics_df: pd.DataFrame) -> str:
    lines = [
        "# 阶段6.0 组合回测结果摘要",
        "",
        "## 运行配置",
        "",
        f"- split: `{args.split}`",
        f"- features_file: `{args.features_file}`",
        f"- label_col: `{args.label_col}`",
        f"- bench_col: `{args.bench_col}`",
        f"- rebalance_every: `{args.rebalance_every}`",
        f"- top_n: `{args.top_n}`",
        f"- top_quantile: `{args.top_quantile}`",
        f"- buy_cost_rate: `{args.buy_cost_rate}`",
        f"- sell_cost_rate: `{args.sell_cost_rate}`",
        "",
        "## 指标汇总",
        "",
    ]
    if metrics_df.empty:
        lines.append("无可用结果。")
    else:
        lines.append(_df_to_markdown_table(metrics_df))
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 本版优先打通链路，默认使用 `label_fwd_ret_5d` 作为未来收益口径。")
    lines.append("- 成本按换手近似扣减，属于研究级简化版，不是最终实盘撮合结果。")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    _ensure_dir(args.out_dir)

    features = _load_features(args.features_file, label_col=args.label_col, bench_col=args.bench_col)
    all_metrics: List[Dict[str, object]] = []

    signal_cfgs = [_parse_signal_config(spec) for spec in args.signal_config]
    for cfg in signal_cfgs:
        sig_df = _load_signal_frame(cfg)
        merged = sig_df.merge(features, on=["date", "stock_code"], how="left")
        merged = _filter_sample(merged, split=args.split, date_start=args.date_start, date_end=args.date_end)
        merged = merged.dropna(subset=[args.label_col, args.bench_col, "score"])
        nav_df, detail_df, group_curve_df, metrics = run_stage6_backtest(
            merged_df=merged,
            alias=cfg.alias,
            label_col=args.label_col,
            bench_col=args.bench_col,
            rebalance_every=args.rebalance_every,
            top_n=args.top_n,
            top_quantile=args.top_quantile,
            group_count=args.group_count,
            init_nav=args.init_nav,
            buy_cost_rate=args.buy_cost_rate,
            sell_cost_rate=args.sell_cost_rate,
            min_candidates=args.min_candidates,
        )

        if nav_df.empty:
            print(f"[WARN] {cfg.alias} 未产生有效回测结果")
            continue

        nav_path = os.path.join(args.out_dir, f"{cfg.alias}_portfolio_nav.csv")
        detail_path = os.path.join(args.out_dir, f"{cfg.alias}_rebalance_detail.csv")
        group_path = os.path.join(args.out_dir, f"{cfg.alias}_group_curve.csv")
        nav_df.to_csv(nav_path, index=False, encoding="utf-8-sig")
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        if not group_curve_df.empty:
            group_curve_df.to_csv(group_path, index=False, encoding="utf-8-sig")

        all_metrics.append({
            "signal_alias": cfg.alias,
            "signal_path": cfg.path,
            "score_col": cfg.score_col,
            **metrics,
        })
        print(f"[OK] {cfg.alias}: nav -> {nav_path}")

    metrics_df = pd.DataFrame(all_metrics)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values("net_total_return", ascending=False).reset_index(drop=True)

    metrics_path = os.path.join(args.out_dir, "comparison_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    summary_md = _render_summary_md(args, metrics_df)
    with open(os.path.join(args.out_dir, "backtest_report.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"[DONE] comparison metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
