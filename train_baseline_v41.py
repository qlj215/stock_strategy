#!/usr/bin/env python
"""
阶段4.1：稳健版基线模型实验脚本（train_baseline_v41.py）

目标：
- 在保留阶段4.0（train_baseline.py）不变的前提下，提供一个“更稳健”的基线实验版本。
- 核心增强：
  1) Ridge 与 Tree 支持不同滚动窗口/重训频率（模型分工）
  2) 支持动态融合（按近期表现自动调节权重）
  3) 提供时间分段诊断（按季度/月份看稳定性）
  4) 可选行业感知增强（行业 one-hot / 行业中性化预测）

输入（默认）：
- data/features/features_v1.parquet
- data/processed/split_manifest.json
- data/features/feature_manifest.json

输出（默认）：
- data/baseline/baseline_v41_result.csv
- data/baseline/baseline_v41_metrics.csv
- data/baseline/baseline_v41_period_metrics.csv
- data/baseline/baseline_v41_report.md

调参速查（优先顺序）：
1) 模型分工：
   - Ridge（稳态）：--ridge-train-window 0（扩展窗口）
   - Tree（应对变化）：--tree-train-window 750（近期窗口）
2) 重训频率：
   - --ridge-retrain-every 20
   - --tree-retrain-every 20（可改 10 提高灵敏度）
3) 融合策略：
   - --blend-mode dynamic（默认）
   - --blend-lookback-days 60（近期表现窗口）
4) 行业感知（可选）：
   - --use-industry-onehot（把行业编码加入特征）
   - --industry-neutralize-pred（预测后按行业做日内中性化）

运行示例：
- 默认稳健版：
  python train_baseline_v41.py
- 更快重训 + 行业感知：
  python train_baseline_v41.py --tree-retrain-every 10 --use-industry-onehot --industry-neutralize-pred
- 固定融合（回退简单模式）：
  python train_baseline_v41.py --blend-mode static --blend-static-w-ridge 0.6
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from train_baseline import (
    RidgeClosedForm,
    SimpleTreeRegressor,
    add_split_col,
    calc_metrics,
    load_split_boundaries,
    resolve_feature_cols,
    walk_forward_predict,
)


def _safe_spearman(g: pd.DataFrame, pred_col: str, target_col: str) -> float:
    if len(g) < 3:
        return np.nan
    if g[pred_col].nunique() <= 1 or g[target_col].nunique() <= 1:
        return np.nan
    return float(g[pred_col].rank(method="average").corr(g[target_col].rank(method="average")))


def _add_industry_onehot(df: pd.DataFrame, feature_cols: List[str]) -> tuple[pd.DataFrame, List[str]]:
    if "industry" not in df.columns:
        return df, feature_cols

    work = df.copy()
    ind = work["industry"].fillna("UNKNOWN").astype(str)
    dummies = pd.get_dummies(ind, prefix="ind", dtype=float)

    # 避免列名冲突
    new_cols = [c for c in dummies.columns if c not in work.columns]
    dummies = dummies[new_cols]

    work = pd.concat([work, dummies], axis=1)
    feature_cols = feature_cols + new_cols
    return work, feature_cols


def _neutralize_by_industry(df: pd.DataFrame, pred_cols: List[str]) -> pd.DataFrame:
    if "industry" not in df.columns:
        return df

    out = df.copy()
    for c in pred_cols:
        if c not in out.columns:
            continue
        group_mean = out.groupby(["date", "industry"])[c].transform("mean")
        out[c] = out[c] - group_mean.fillna(0.0)
    return out


def _cs_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    mean = df.groupby("date")[col].transform("mean")
    std = df.groupby("date")[col].transform("std").replace(0, np.nan)
    return (df[col] - mean) / std


def _compute_dynamic_weights(
    merged: pd.DataFrame,
    target_col: str,
    lookback_days: int,
    default_w_ridge: float,
) -> pd.DataFrame:
    out = merged.copy()
    pred_dates = sorted(out["date"].dropna().unique())
    if not pred_dates:
        out["w_ridge"] = default_w_ridge
        out["w_tree"] = 1.0 - default_w_ridge
        return out

    date_to_idx = {d: i for i, d in enumerate(pred_dates)}
    daily_ic = out.groupby("date").apply(
        lambda g: pd.Series(
            {
                "ic_ridge": _safe_spearman(g, "pred_ridge_z", target_col),
                "ic_tree": _safe_spearman(g, "pred_tree_z", target_col),
            }
        )
    )

    w_ridge_map: Dict[pd.Timestamp, float] = {}

    for d in pred_dates:
        idx = date_to_idx[d]
        start = max(0, idx - max(1, lookback_days))
        hist_dates = pred_dates[start:idx]

        if len(hist_dates) == 0:
            w_ridge = float(default_w_ridge)
        else:
            hist = daily_ic.loc[hist_dates]
            ic_r = float(hist["ic_ridge"].mean()) if "ic_ridge" in hist.columns else np.nan
            ic_t = float(hist["ic_tree"].mean()) if "ic_tree" in hist.columns else np.nan

            s_r = max(0.0, 0.0 if pd.isna(ic_r) else ic_r)
            s_t = max(0.0, 0.0 if pd.isna(ic_t) else ic_t)

            if s_r + s_t <= 1e-12:
                w_ridge = float(default_w_ridge)
            else:
                w_ridge = float(s_r / (s_r + s_t))

        w_ridge = min(0.9, max(0.1, w_ridge))
        w_ridge_map[d] = w_ridge

    out["w_ridge"] = out["date"].map(w_ridge_map).fillna(float(default_w_ridge))
    out["w_tree"] = 1.0 - out["w_ridge"]
    return out


def _build_period_metrics(
    df: pd.DataFrame,
    target_col: str,
    pred_cols: List[str],
    period_freq: str,
) -> pd.DataFrame:
    rows = []

    work = df.copy()
    work["period"] = work["date"].dt.to_period(period_freq).astype(str)

    for model, pred_col in pred_cols:
        for split in ["val", "test", "all"]:
            part = work if split == "all" else work[work["split"] == split]
            if part.empty:
                continue
            for period, gp in part.groupby("period"):
                m = calc_metrics(gp, pred_col=pred_col, target_col=target_col, n_quantiles=5)
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "period": period,
                        "rows": m.get("rows", 0),
                        "days": m.get("days", 0),
                        "rank_ic_mean": m.get("rank_ic_mean", np.nan),
                        "icir": m.get("icir", np.nan),
                        "long_short_ret": m.get("long_short_ret", np.nan),
                        "direction_hit_rate": m.get("direction_hit_rate", np.nan),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["model", "split", "period"]).reset_index(drop=True)


def _markdown_table(df: pd.DataFrame, cols: List[str]) -> List[str]:
    if df.empty:
        return ["（无数据）"]

    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---" for _ in cols]) + "|"
    lines = [head, sep]

    for _, r in df[cols].iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                vals.append("NA")
            elif isinstance(v, (int, np.integer)):
                vals.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def build_report(
    out_path: Path,
    args,
    merged: pd.DataFrame,
    metrics: pd.DataFrame,
    period_metrics: pd.DataFrame,
    feature_cols: List[str],
):
    lines: List[str] = []
    lines.append("# baseline_v41_report（阶段4.1稳健版）")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat()}")
    lines.append("")

    lines.append("## 1. 运行配置")
    lines.append("")
    lines.append(f"- 输入：`{args.in_path}`")
    lines.append(f"- split_manifest：`{args.split_manifest}`")
    lines.append(f"- target：`{args.target_col}`")
    lines.append(f"- 特征数：`{len(feature_cols)}`")
    lines.append(f"- use_industry_onehot：`{args.use_industry_onehot}`")
    lines.append(f"- industry_neutralize_pred：`{args.industry_neutralize_pred}`")
    lines.append("")
    lines.append("### 模型参数")
    lines.append(f"- Ridge: retrain_every={args.ridge_retrain_every}, train_window={args.ridge_train_window}, alpha={args.ridge_alpha}")
    lines.append(
        "- Tree: retrain_every={re}, train_window={tw}, depth={d}, min_leaf={ml}, thresholds={nt}".format(
            re=args.tree_retrain_every,
            tw=args.tree_train_window,
            d=args.tree_max_depth,
            ml=args.tree_min_leaf,
            nt=args.tree_n_thresholds,
        )
    )
    lines.append("")
    lines.append("### 融合参数")
    lines.append(f"- blend_mode={args.blend_mode}")
    lines.append(f"- blend_lookback_days={args.blend_lookback_days}")
    lines.append(f"- blend_default_w_ridge={args.blend_default_w_ridge}")
    lines.append(f"- blend_static_w_ridge={args.blend_static_w_ridge}")
    lines.append("")

    if "w_ridge" in merged.columns and len(merged):
        wd = merged.groupby("split")["w_ridge"].agg(["mean", "min", "max"]).reset_index()
        lines.append("### 动态权重统计（w_ridge）")
        lines += _markdown_table(wd, ["split", "mean", "min", "max"])
        lines.append("")

    lines.append("## 2. 综合指标（val/test/all）")
    lines.append("")
    lines += _markdown_table(
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

    lines.append("## 3. 时间分段指标")
    lines.append("")
    lines.append(f"- 分段频率：`{args.period_freq}`")
    lines.append("")

    show = period_metrics.copy()
    if len(show) > 60:
        show = show.tail(60)
        lines.append("（仅展示最近60行，完整数据见 baseline_v41_period_metrics.csv）")
        lines.append("")
    lines += _markdown_table(
        show,
        ["model", "split", "period", "days", "rank_ic_mean", "long_short_ret", "direction_hit_rate"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="阶段4.1：稳健版基线实验（模型分工 + 动态融合 + 时间分段诊断）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例：\n"
            "  python train_baseline_v41.py\n"
            "  python train_baseline_v41.py --tree-retrain-every 10 --tree-train-window 750\n"
            "  python train_baseline_v41.py --blend-mode static --blend-static-w-ridge 0.6\n"
            "  python train_baseline_v41.py --use-industry-onehot --industry-neutralize-pred\n"
        ),
    )

    p.add_argument("--in", dest="in_path", default="data/features/features_v1.parquet", help="阶段3特征标签输入")
    p.add_argument("--split-manifest", default="data/processed/split_manifest.json", help="阶段2切分清单")
    p.add_argument("--feature-manifest", default="data/features/feature_manifest.json", help="阶段3特征清单")
    p.add_argument("--feature-cols", default="", help="显式特征列（逗号分隔）")

    p.add_argument("--target-col", default="label_excess_ret_5d", help="回归目标列")
    p.add_argument("--fillna", type=float, default=0.0, help="特征缺失填充值")
    p.add_argument("--min-train-days", type=int, default=240, help="最小训练交易日数")

    p.add_argument("--ridge-retrain-every", type=int, default=20, help="Ridge 重训间隔")
    p.add_argument("--ridge-train-window", type=int, default=0, help="Ridge 训练窗口(0=扩展)")
    p.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge L2 强度")

    p.add_argument("--tree-retrain-every", type=int, default=20, help="Tree 重训间隔")
    p.add_argument("--tree-train-window", type=int, default=750, help="Tree 训练窗口(建议近期窗口)")
    p.add_argument("--tree-max-depth", type=int, default=3, help="Tree 最大深度")
    p.add_argument("--tree-min-split", type=int, default=200, help="Tree 分裂最小样本")
    p.add_argument("--tree-min-leaf", type=int, default=80, help="Tree 叶子最小样本")
    p.add_argument("--tree-n-thresholds", type=int, default=8, help="Tree 候选阈值数")
    p.add_argument("--tree-max-features", type=int, default=10, help="Tree 每节点特征采样数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    p.add_argument("--blend-mode", choices=["dynamic", "static"], default="dynamic", help="融合模式")
    p.add_argument("--blend-lookback-days", type=int, default=60, help="动态融合回看天数")
    p.add_argument("--blend-default-w-ridge", type=float, default=0.6, help="动态融合缺省 Ridge 权重")
    p.add_argument("--blend-static-w-ridge", type=float, default=0.6, help="静态融合 Ridge 权重")

    p.add_argument("--use-industry-onehot", action="store_true", help="行业 one-hot 加入特征")
    p.add_argument("--industry-neutralize-pred", action="store_true", help="预测后按行业做日内中性化")

    p.add_argument("--period-freq", choices=["Q", "M"], default="Q", help="分段诊断频率")
    p.add_argument("--n-quantiles", type=int, default=5, help="分层收益分位数")

    p.add_argument("--out", default="data/baseline/baseline_v41_result.csv", help="预测结果输出")
    p.add_argument("--metrics-out", default="data/baseline/baseline_v41_metrics.csv", help="指标输出")
    p.add_argument(
        "--period-metrics-out",
        default="data/baseline/baseline_v41_period_metrics.csv",
        help="时间分段指标输出",
    )
    p.add_argument("--report-out", default="data/baseline/baseline_v41_report.md", help="报告输出")

    args = p.parse_args()

    if not (0.0 <= args.blend_default_w_ridge <= 1.0):
        raise SystemExit("[ERROR] --blend-default-w-ridge 需在 [0,1]")
    if not (0.0 <= args.blend_static_w_ridge <= 1.0):
        raise SystemExit("[ERROR] --blend-static-w-ridge 需在 [0,1]")

    in_path = Path(args.in_path)
    split_path = Path(args.split_manifest)
    feat_manifest_path = Path(args.feature_manifest)

    out_path = Path(args.out)
    metrics_path = Path(args.metrics_out)
    period_metrics_path = Path(args.period_metrics_out)
    report_path = Path(args.report_out)

    if not in_path.exists():
        raise SystemExit(f"[ERROR] 输入不存在: {in_path}")

    df = pd.read_parquet(in_path)
    if "date" not in df.columns or "stock_code" not in df.columns:
        raise SystemExit("[ERROR] 输入缺少 date/stock_code")

    if args.target_col not in df.columns:
        raise SystemExit(f"[ERROR] 目标列不存在: {args.target_col}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)

    train_end, val_end = load_split_boundaries(split_path)
    df = add_split_col(df, train_end=train_end, val_end=val_end)

    feature_cols = resolve_feature_cols(df, args.feature_cols, feat_manifest_path)

    # 保留 industry 以便可选 one-hot / 中性化
    keep_cols = ["date", "stock_code", "split", args.target_col] + feature_cols
    if "industry" in df.columns and "industry" not in keep_cols:
        keep_cols.append("industry")

    work = df[keep_cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["date", "stock_code", args.target_col])

    if args.use_industry_onehot:
        work, feature_cols = _add_industry_onehot(work, feature_cols)

    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(float(args.fillna))

    work = work.sort_values(["date", "stock_code"]).reset_index(drop=True)

    print(f"[INFO] input rows={len(work)} stocks={work['stock_code'].nunique()} days={work['date'].nunique()}")
    print(f"[INFO] feature_cols={len(feature_cols)} target={args.target_col}")

    ridge_pred = walk_forward_predict(
        df=work,
        feature_cols=feature_cols,
        target_col=args.target_col,
        train_end=train_end,
        retrain_every=max(1, args.ridge_retrain_every),
        train_window=max(0, args.ridge_train_window),
        min_train_days=max(1, args.min_train_days),
        model_factory=lambda: RidgeClosedForm(alpha=args.ridge_alpha),
        pred_col_name="pred_ridge",
    )

    tree_pred = walk_forward_predict(
        df=work,
        feature_cols=feature_cols,
        target_col=args.target_col,
        train_end=train_end,
        retrain_every=max(1, args.tree_retrain_every),
        train_window=max(0, args.tree_train_window),
        min_train_days=max(1, args.min_train_days),
        model_factory=lambda: SimpleTreeRegressor(
            max_depth=max(1, args.tree_max_depth),
            min_samples_split=max(2, args.tree_min_split),
            min_samples_leaf=max(1, args.tree_min_leaf),
            n_thresholds=max(2, args.tree_n_thresholds),
            max_features=max(1, args.tree_max_features),
            random_state=args.seed,
        ),
        pred_col_name="pred_tree",
    )

    # 合并预测
    key_cols = ["date", "stock_code", "split", args.target_col]
    merged = ridge_pred.merge(tree_pred, on=key_cols, how="inner", suffixes=("", "_tree"))

    # 标注训练窗口边界（分别保留）
    if "fit_start_date" in ridge_pred.columns:
        merged = merged.merge(
            ridge_pred[["date", "stock_code", "fit_start_date", "fit_end_date"]],
            on=["date", "stock_code"],
            how="left",
            suffixes=("", "_ridge_meta"),
        )
        merged = merged.rename(
            columns={
                "fit_start_date": "ridge_fit_start_date",
                "fit_end_date": "ridge_fit_end_date",
            }
        )

    if "fit_start_date" in tree_pred.columns:
        tree_meta = tree_pred[["date", "stock_code", "fit_start_date", "fit_end_date"]].rename(
            columns={"fit_start_date": "tree_fit_start_date", "fit_end_date": "tree_fit_end_date"}
        )
        merged = merged.merge(tree_meta, on=["date", "stock_code"], how="left")

    # 回填 industry 便于可选中性化
    if "industry" in work.columns:
        merged = merged.merge(work[["date", "stock_code", "industry"]].drop_duplicates(), on=["date", "stock_code"], how="left")

    # 可选行业中性化
    if args.industry_neutralize_pred:
        merged = _neutralize_by_industry(merged, ["pred_ridge", "pred_tree"])

    # 融合前统一尺度
    merged["pred_ridge_z"] = _cs_zscore(merged, "pred_ridge").fillna(0.0)
    merged["pred_tree_z"] = _cs_zscore(merged, "pred_tree").fillna(0.0)

    if args.blend_mode == "static":
        merged["w_ridge"] = float(args.blend_static_w_ridge)
        merged["w_tree"] = 1.0 - merged["w_ridge"]
    else:
        merged = _compute_dynamic_weights(
            merged,
            target_col=args.target_col,
            lookback_days=max(1, int(args.blend_lookback_days)),
            default_w_ridge=float(args.blend_default_w_ridge),
        )

    merged["pred_blend"] = merged["w_ridge"] * merged["pred_ridge_z"] + merged["w_tree"] * merged["pred_tree_z"]

    merged = merged.sort_values(["date", "stock_code"]).reset_index(drop=True)

    # 指标输出
    metrics_rows = []
    model_cols = [("ridge", "pred_ridge"), ("tree", "pred_tree"), ("blend", "pred_blend")]
    for model_name, pred_col in model_cols:
        for split in ["val", "test", "all"]:
            sub = merged if split == "all" else merged[merged["split"] == split]
            m = calc_metrics(sub, pred_col=pred_col, target_col=args.target_col, n_quantiles=max(2, args.n_quantiles))
            m["model"] = model_name
            m["split"] = split
            metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[
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

    period_metrics_df = _build_period_metrics(
        merged,
        target_col=args.target_col,
        pred_cols=model_cols,
        period_freq=args.period_freq,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    period_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    period_metrics_df.to_csv(period_metrics_path, index=False, encoding="utf-8")

    build_report(
        out_path=report_path,
        args=args,
        merged=merged,
        metrics=metrics_df,
        period_metrics=period_metrics_df,
        feature_cols=feature_cols,
    )

    print("[DONE] stage4.1 baseline finished")
    print(f"  result        : {out_path}")
    print(f"  metrics       : {metrics_path}")
    print(f"  period_metrics: {period_metrics_path}")
    print(f"  report        : {report_path}")
    print(f"  rows          : {len(merged)}")
    print(f"  days          : {merged['date'].nunique() if len(merged) else 0}")

    for m in ["ridge", "tree", "blend"]:
        t = metrics_df[(metrics_df["model"] == m) & (metrics_df["split"] == "test")]
        if not t.empty:
            r = t.iloc[0]
            print(
                f"  [{m}] test rank_ic={r['rank_ic_mean']:.6f} icir={r['icir']:.6f} "
                f"long_short={r['long_short_ret']:.6f} hit={r['direction_hit_rate']:.6f}"
            )


if __name__ == "__main__":
    main()
