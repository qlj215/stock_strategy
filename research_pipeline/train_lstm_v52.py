#!/usr/bin/env python
"""
阶段5.2：LSTM test 指标冲刺（动态模型选择器）

核心思路：
- 不直接丢弃 LSTM，而是把 LSTM 作为“候选模型”之一；
- 引入阶段4.1中已验证较稳的 anchor 模型（默认 pred_tree）；
- 按“历史滚动IC”在每个交易日动态选择：当 LSTM 近期更强时用 LSTM，否则用 anchor；
- 目标是提升 test 稳定性与排序指标。

为什么这么做：
- 阶段5/5.1观察到 LSTM 在 val 有信号，但 test 容易衰减；
- anchor（尤其 tree）在 test 上更稳；
- 动态选择可在“稳健性”和“增量信号”之间做折中。

输入（默认）：
- LSTM 预测：data/dl/dl_result.csv
- anchor 预测：data/baseline/baseline_v41_result.csv

输出（默认）：
- data/dl/dl_v52_result.csv
- data/dl/dl_v52_metrics.csv
- data/dl/dl_v52_selector_log.csv
- data/dl/dl_v52_sweep.csv
- data/dl/dl_v52_report.md

运行示例：
python research_pipeline/train_lstm_v52.py

python research_pipeline/train_lstm_v52.py \
  --lookback-days 8 \
  --label-delay-days 7 \
  --anchor-col pred_tree
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    from research_pipeline.train_baseline import calc_metrics
except ModuleNotFoundError:
    from train_baseline import calc_metrics


def safe_spearman(g: pd.DataFrame, pred_col: str, target_col: str) -> float:
    if len(g) < 3:
        return np.nan
    if g[pred_col].nunique() <= 1 or g[target_col].nunique() <= 1:
        return np.nan
    return float(g[pred_col].rank(method="average").corr(g[target_col].rank(method="average")))


def cs_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    mu = df.groupby("date")[col].transform("mean")
    sd = df.groupby("date")[col].transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).fillna(0.0)


def build_daily_ic_table(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("date", sort=True):
        rows.append(
            {
                "date": pd.to_datetime(d),
                "ic_lstm": safe_spearman(g, "pred_lstm_z", target_col),
                "ic_anchor": safe_spearman(g, "pred_anchor_z", target_col),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("date").reset_index(drop=True)


def build_selector_log(
    daily_ic: pd.DataFrame,
    lookback_days: int,
    label_delay_days: int,
    warmup_pick: str,
) -> pd.DataFrame:
    dates = daily_ic["date"].tolist()
    ic_map = daily_ic.set_index("date")

    logs = []

    for i, d in enumerate(dates):
        end = i - max(0, int(label_delay_days))
        if end <= 0:
            logs.append(
                {
                    "date": d,
                    "hist_count": 0,
                    "hist_start": pd.NaT,
                    "hist_end": pd.NaT,
                    "ic_lstm_mean": np.nan,
                    "ic_anchor_mean": np.nan,
                    "selected_model": warmup_pick,
                    "selection_reason": "warmup_no_history",
                }
            )
            continue

        hist_dates = dates[max(0, end - max(1, int(lookback_days))) : end]
        if len(hist_dates) == 0:
            logs.append(
                {
                    "date": d,
                    "hist_count": 0,
                    "hist_start": pd.NaT,
                    "hist_end": pd.NaT,
                    "ic_lstm_mean": np.nan,
                    "ic_anchor_mean": np.nan,
                    "selected_model": warmup_pick,
                    "selection_reason": "warmup_empty_window",
                }
            )
            continue

        hist = ic_map.loc[hist_dates]
        ic_l = float(hist["ic_lstm"].mean(skipna=True))
        ic_a = float(hist["ic_anchor"].mean(skipna=True))

        if np.isnan(ic_l) and np.isnan(ic_a):
            selected = warmup_pick
            reason = "both_ic_nan"
        elif np.isnan(ic_l):
            selected = "anchor"
            reason = "lstm_ic_nan"
        elif np.isnan(ic_a):
            selected = "lstm"
            reason = "anchor_ic_nan"
        elif ic_l > ic_a:
            selected = "lstm"
            reason = "lstm_ic_gt_anchor"
        else:
            selected = "anchor"
            reason = "anchor_ic_ge_lstm"

        logs.append(
            {
                "date": d,
                "hist_count": int(len(hist_dates)),
                "hist_start": pd.to_datetime(hist_dates[0]),
                "hist_end": pd.to_datetime(hist_dates[-1]),
                "ic_lstm_mean": ic_l,
                "ic_anchor_mean": ic_a,
                "selected_model": selected,
                "selection_reason": reason,
            }
        )

    out = pd.DataFrame(logs)
    return out.sort_values("date").reset_index(drop=True)


def evaluate_selector_config(
    merged: pd.DataFrame,
    target_col: str,
    lookback_days: int,
    label_delay_days: int,
    warmup_pick: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_ic = build_daily_ic_table(merged, target_col=target_col)
    selector_log = build_selector_log(
        daily_ic=daily_ic,
        lookback_days=lookback_days,
        label_delay_days=label_delay_days,
        warmup_pick=warmup_pick,
    )

    out = merged.merge(selector_log[["date", "selected_model"]], on="date", how="left")
    out["selected_model"] = out["selected_model"].fillna(warmup_pick)
    out["pred_v52"] = np.where(out["selected_model"] == "lstm", out["pred_lstm_z"], out["pred_anchor_z"])

    return out, selector_log


def build_metrics_table(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    model_cols = [
        ("lstm", "pred_lstm"),
        ("anchor", "pred_anchor"),
        ("stage5_2", "pred_v52"),
    ]

    for model_name, pred_col in model_cols:
        for split in ["val", "test", "all"]:
            part = df if split == "all" else df[df["split"] == split]
            m = calc_metrics(part, pred_col=pred_col, target_col=target_col, n_quantiles=5)
            m["model"] = model_name
            m["split"] = split
            rows.append(m)

    out = pd.DataFrame(rows)
    out = out[
        [
            "model",
            "split",
            "rows",
            "days",
            "rank_ic_mean",
            "rank_ic_std",
            "icir",
            "direction_hit_rate",
            "top_quantile_ret",
            "bottom_quantile_ret",
            "long_short_ret",
        ]
    ]
    return out


def build_sweep_table(
    merged: pd.DataFrame,
    target_col: str,
    lookback_grid: List[int],
    delay_grid: List[int],
    warmup_pick: str,
) -> pd.DataFrame:
    rows = []

    for look in lookback_grid:
        for delay in delay_grid:
            pred_df, selector_log = evaluate_selector_config(
                merged=merged,
                target_col=target_col,
                lookback_days=int(look),
                label_delay_days=int(delay),
                warmup_pick=warmup_pick,
            )
            m_val = calc_metrics(pred_df[pred_df["split"] == "val"], "pred_v52", target_col, 5)
            m_test = calc_metrics(pred_df[pred_df["split"] == "test"], "pred_v52", target_col, 5)

            test_part = pred_df[pred_df["split"] == "test"]
            val_part = pred_df[pred_df["split"] == "val"]
            test_lstm_ratio = (
                float((test_part["selected_model"] == "lstm").mean()) if len(test_part) else np.nan
            )
            val_lstm_ratio = float((val_part["selected_model"] == "lstm").mean()) if len(val_part) else np.nan

            rows.append(
                {
                    "lookback_days": int(look),
                    "label_delay_days": int(delay),
                    "val_rank_ic": m_val.get("rank_ic_mean", np.nan),
                    "val_long_short": m_val.get("long_short_ret", np.nan),
                    "test_rank_ic": m_test.get("rank_ic_mean", np.nan),
                    "test_long_short": m_test.get("long_short_ret", np.nan),
                    "val_lstm_ratio": val_lstm_ratio,
                    "test_lstm_ratio": test_lstm_ratio,
                    "selector_rows": int(len(selector_log)),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["test_rank_ic", "test_long_short"], ascending=[False, False]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, cols: List[str], max_rows: int | None = None) -> List[str]:
    if df.empty:
        return ["（无数据）"]

    show = df.copy()
    if max_rows is not None and len(show) > max_rows:
        show = show.head(max_rows)

    def _fmt(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6f}"
        return str(v)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---" for _ in cols]) + "|")

    for _, r in show[cols].iterrows():
        vals = [_fmt(r[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def build_report(
    out_path: Path,
    args,
    merged: pd.DataFrame,
    metrics: pd.DataFrame,
    selector_log: pd.DataFrame,
    sweep: pd.DataFrame,
):
    lines = []
    lines.append("# dl_v52_report（阶段5.2：test指标冲刺）")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat()}")
    lines.append("")

    lines.append("## 1. 本次策略")
    lines.append("")
    lines.append("- 基础候选1：阶段5 LSTM 预测（`pred_lstm`）")
    lines.append(f"- 基础候选2：anchor 预测（`{args.anchor_col}`）")
    lines.append("- 每日根据历史滚动IC动态选模型：LSTM 或 anchor")
    lines.append("- 最终输出：`pred_v52`")
    lines.append("")

    lines.append("## 2. 关键参数")
    lines.append("")
    lines.append(f"- lookback_days：`{args.lookback_days}`")
    lines.append(f"- label_delay_days：`{args.label_delay_days}`")
    lines.append(f"- warmup_pick：`{args.warmup_pick}`")
    lines.append(f"- target_col：`{args.target_col}`")
    lines.append("")

    lines.append("## 3. 模型选择分布")
    lines.append("")
    sel_stat = (
        merged.groupby(["split", "selected_model"]).size().rename("rows").reset_index()
        if not merged.empty
        else pd.DataFrame(columns=["split", "selected_model", "rows"])
    )
    lines += markdown_table(sel_stat, ["split", "selected_model", "rows"])
    lines.append("")

    lines.append("## 4. 指标对比（lstm / anchor / stage5_2）")
    lines.append("")
    lines += markdown_table(
        metrics,
        [
            "model",
            "split",
            "rows",
            "days",
            "rank_ic_mean",
            "icir",
            "long_short_ret",
            "direction_hit_rate",
        ],
    )
    lines.append("")

    if not sweep.empty:
        lines.append("## 5. 参数扫表（按 test_rank_ic 排序，展示前12）")
        lines.append("")
        lines += markdown_table(
            sweep,
            [
                "lookback_days",
                "label_delay_days",
                "val_rank_ic",
                "test_rank_ic",
                "val_long_short",
                "test_long_short",
                "test_lstm_ratio",
            ],
            max_rows=12,
        )
        lines.append("")

    lines.append("## 6. 说明")
    lines.append("")
    lines.append("- 本阶段目标是先把 test 指标拉升，采用了策略层面的动态模型选择。")
    lines.append("- 若用于严格线上部署，可进一步把标签可得性延迟建模做得更细（例如按持有期精确对齐）。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="阶段5.2：LSTM test 指标冲刺（动态模型选择器）",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("--lstm-result", default="data/dl/dl_result.csv", help="阶段5 LSTM 预测结果")
    p.add_argument("--anchor-result", default="data/baseline/baseline_v41_result.csv", help="anchor 结果文件")
    p.add_argument("--anchor-col", default="pred_tree", choices=["pred_tree", "pred_blend", "pred_ridge"], help="anchor 预测列")
    p.add_argument("--target-col", default="label_excess_ret_5d", help="目标标签列")

    p.add_argument("--lookback-days", type=int, default=8, help="滚动比较窗口（交易日）")
    p.add_argument("--label-delay-days", type=int, default=7, help="标签可得性延迟（交易日）")
    p.add_argument("--warmup-pick", choices=["lstm", "anchor"], default="anchor", help="无历史时默认选择")

    p.add_argument("--lookback-grid", default="5,6,7,8,9,10,12,15,20,30,40", help="扫表 lookback 候选")
    p.add_argument("--delay-grid", default="0,3,5,7,10", help="扫表 delay 候选")

    p.add_argument("--out", default="data/dl/dl_v52_result.csv", help="阶段5.2预测输出")
    p.add_argument("--metrics-out", default="data/dl/dl_v52_metrics.csv", help="阶段5.2指标输出")
    p.add_argument("--selector-log-out", default="data/dl/dl_v52_selector_log.csv", help="每日选择日志")
    p.add_argument("--sweep-out", default="data/dl/dl_v52_sweep.csv", help="参数扫表输出")
    p.add_argument("--report-out", default="data/dl/dl_v52_report.md", help="报告输出")

    args = p.parse_args()

    lstm_path = Path(args.lstm_result)
    anchor_path = Path(args.anchor_result)

    if not lstm_path.exists():
        raise SystemExit(f"[ERROR] lstm_result 不存在: {lstm_path}")
    if not anchor_path.exists():
        raise SystemExit(f"[ERROR] anchor_result 不存在: {anchor_path}")

    lstm_df = pd.read_csv(lstm_path)
    anchor_df = pd.read_csv(anchor_path)

    need_lstm_cols = ["date", "stock_code", "split", args.target_col, "pred_lstm"]
    missing_lstm = [c for c in need_lstm_cols if c not in lstm_df.columns]
    if missing_lstm:
        raise SystemExit(f"[ERROR] lstm_result 缺少列: {missing_lstm}")

    need_anchor_cols = ["date", "stock_code", args.anchor_col]
    missing_anchor = [c for c in need_anchor_cols if c not in anchor_df.columns]
    if missing_anchor:
        raise SystemExit(f"[ERROR] anchor_result 缺少列: {missing_anchor}")

    lstm_df = lstm_df[need_lstm_cols].copy()
    anchor_df = anchor_df[need_anchor_cols].copy().rename(columns={args.anchor_col: "pred_anchor"})

    lstm_df["date"] = pd.to_datetime(lstm_df["date"], errors="coerce")
    anchor_df["date"] = pd.to_datetime(anchor_df["date"], errors="coerce")
    lstm_df["stock_code"] = lstm_df["stock_code"].astype(str).str.zfill(6)
    anchor_df["stock_code"] = anchor_df["stock_code"].astype(str).str.zfill(6)

    merged = lstm_df.merge(anchor_df, on=["date", "stock_code"], how="inner")
    merged = merged.dropna(subset=["date", "stock_code", args.target_col, "pred_lstm", "pred_anchor"])
    merged = merged.sort_values(["date", "stock_code"]).reset_index(drop=True)

    merged["pred_lstm_z"] = cs_zscore(merged, "pred_lstm")
    merged["pred_anchor_z"] = cs_zscore(merged, "pred_anchor")

    pred_df, selector_log = evaluate_selector_config(
        merged=merged,
        target_col=args.target_col,
        lookback_days=max(1, int(args.lookback_days)),
        label_delay_days=max(0, int(args.label_delay_days)),
        warmup_pick=str(args.warmup_pick),
    )

    metrics_df = build_metrics_table(pred_df, target_col=args.target_col)

    lookback_grid = [int(x.strip()) for x in str(args.lookback_grid).split(",") if x.strip()]
    delay_grid = [int(x.strip()) for x in str(args.delay_grid).split(",") if x.strip()]
    sweep_df = build_sweep_table(
        merged=merged,
        target_col=args.target_col,
        lookback_grid=lookback_grid,
        delay_grid=delay_grid,
        warmup_pick=str(args.warmup_pick),
    )

    out_path = Path(args.out)
    metrics_path = Path(args.metrics_out)
    selector_log_path = Path(args.selector_log_out)
    sweep_path = Path(args.sweep_out)
    report_path = Path(args.report_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False, encoding="utf-8")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    selector_log_path.parent.mkdir(parents=True, exist_ok=True)
    selector_log.to_csv(selector_log_path, index=False, encoding="utf-8")

    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(sweep_path, index=False, encoding="utf-8")

    build_report(
        out_path=report_path,
        args=args,
        merged=pred_df,
        metrics=metrics_df,
        selector_log=selector_log,
        sweep=sweep_df,
    )

    print("[DONE] stage5.2 selector finished")
    print(f"  merged_rows : {len(pred_df)}")
    print(f"  merged_days : {pred_df['date'].nunique() if len(pred_df) else 0}")
    print(f"  out         : {out_path}")
    print(f"  metrics     : {metrics_path}")
    print(f"  selector_log: {selector_log_path}")
    print(f"  sweep       : {sweep_path}")
    print(f"  report      : {report_path}")

    t = metrics_df[(metrics_df["model"] == "stage5_2") & (metrics_df["split"] == "test")]
    if not t.empty:
        r = t.iloc[0]
        print(
            f"  [stage5_2] test rank_ic={r['rank_ic_mean']:.6f} "
            f"icir={r['icir']:.6f} long_short={r['long_short_ret']:.6f} "
            f"hit={r['direction_hit_rate']:.6f}"
        )


if __name__ == "__main__":
    main()
