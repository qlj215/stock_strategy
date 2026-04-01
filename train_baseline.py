#!/usr/bin/env python
"""
阶段4：基线模型实验脚本（train_baseline.py）

目标：
- 在阶段3特征标签表上完成“线性模型 + 树模型”基线实验。
- 使用时间滚动（walk-forward）方式训练并预测，避免未来信息泄漏。
- 输出每日预测分数及核心排序指标，作为阶段5深度学习对照组。

输入（默认）：
- data/features/features_v1.parquet
- data/processed/split_manifest.json

输出（默认）：
- data/baseline/baseline_result.csv
- data/baseline/baseline_metrics.csv
- data/baseline/baseline_report.md

调参速查（优先顺序）：
1) 目标标签：
   - --target-col label_excess_ret_5d（默认）
   - 可切到 label_excess_ret_10d 做对照
2) Ridge 正则：
   - --ridge-alpha 0.1~10（默认 1.0）
3) 树模型复杂度：
   - --tree-max-depth（默认 3）
   - --tree-min-leaf（默认 80）
   - --tree-n-thresholds（默认 8）
4) 滚动训练方式：
   - --retrain-every（默认 20 个交易日）
   - --train-window（默认 0=扩展窗口；>0=固定最近N日）
5) 评估分层：
   - --n-quantiles（默认 5）

运行示例：
- 默认跑法：
  python train_baseline.py
- 使用10日超额收益标签：
  python train_baseline.py --target-col label_excess_ret_10d
- 固定训练窗口（最近500个交易日）：
  python train_baseline.py --train-window 500 --retrain-every 20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd


# --------------------------
# 模型定义：Ridge（闭式解）
# --------------------------
class RidgeClosedForm:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 2:
            raise ValueError("x 必须是二维数组")
        if len(x) == 0:
            raise ValueError("训练样本为空")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        xc = x - x_mean
        yc = y - y_mean

        n_features = x.shape[1]
        a = xc.T @ xc + self.alpha * np.eye(n_features)
        b = xc.T @ yc

        try:
            w = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(a) @ b

        self.coef_ = w
        self.intercept_ = y_mean - float(x_mean @ w)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未 fit")
        x = np.asarray(x, dtype=np.float64)
        return x @ self.coef_ + self.intercept_


# --------------------------
# 模型定义：简化CART回归树
# --------------------------
@dataclass
class _TreeNode:
    is_leaf: bool
    value: float
    feature_idx: int = -1
    threshold: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


class SimpleTreeRegressor:
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 200,
        min_samples_leaf: int = 80,
        n_thresholds: int = 8,
        max_features: int = 10,
        min_gain: float = 1e-10,
        random_state: int = 42,
    ):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_thresholds = int(n_thresholds)
        self.max_features = int(max_features)
        self.min_gain = float(min_gain)
        self.random_state = int(random_state)
        self.root_: _TreeNode | None = None
        self._rng = np.random.default_rng(self.random_state)

    @staticmethod
    def _sse(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        mu = float(y.mean())
        return float(np.square(y - mu).sum())

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        node_value = float(y.mean()) if len(y) else 0.0

        if depth >= self.max_depth:
            return _TreeNode(is_leaf=True, value=node_value)
        if len(y) < self.min_samples_split:
            return _TreeNode(is_leaf=True, value=node_value)
        if float(np.nanstd(y)) < 1e-12:
            return _TreeNode(is_leaf=True, value=node_value)

        n_samples, n_features = x.shape
        parent_sse = self._sse(y)
        if parent_sse <= 0:
            return _TreeNode(is_leaf=True, value=node_value)

        if 0 < self.max_features < n_features:
            feat_idx = self._rng.choice(n_features, size=self.max_features, replace=False)
        else:
            feat_idx = np.arange(n_features)

        best_gain = 0.0
        best_feature = -1
        best_threshold = 0.0
        best_left_mask = None

        # 量化阈值点，控制复杂度
        qs = np.linspace(0.1, 0.9, max(2, self.n_thresholds))

        for j in feat_idx:
            col = x[:, j]
            try:
                thresholds = np.unique(np.quantile(col, qs))
            except Exception:
                continue
            if len(thresholds) == 0:
                continue

            for thr in thresholds:
                left_mask = col <= thr
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                yl = y[left_mask]
                yr = y[~left_mask]
                sse = self._sse(yl) + self._sse(yr)
                gain = parent_sse - sse
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(j)
                    best_threshold = float(thr)
                    best_left_mask = left_mask

        if best_feature < 0 or best_left_mask is None or best_gain <= self.min_gain:
            return _TreeNode(is_leaf=True, value=node_value)

        left = self._build(x[best_left_mask], y[best_left_mask], depth + 1)
        right = self._build(x[~best_left_mask], y[~best_left_mask], depth + 1)
        return _TreeNode(
            is_leaf=False,
            value=node_value,
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left,
            right=right,
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 2:
            raise ValueError("x 必须是二维数组")
        if len(x) == 0:
            raise ValueError("训练样本为空")
        self.root_ = self._build(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), depth=0)
        return self

    def _predict_one(self, row: np.ndarray) -> float:
        if self.root_ is None:
            raise RuntimeError("模型尚未 fit")

        node = self.root_
        while not node.is_leaf:
            if row[node.feature_idx] <= node.threshold:
                node = node.left if node.left is not None else node
            else:
                node = node.right if node.right is not None else node
        return float(node.value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.array([self._predict_one(row) for row in x], dtype=np.float64)


def _parse_cols(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def resolve_feature_cols(df: pd.DataFrame, feature_cols_arg: str, feature_manifest_path: Path) -> List[str]:
    if feature_cols_arg:
        cols = _parse_cols(feature_cols_arg)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"--feature-cols 含不存在列: {missing}")
        return cols

    if feature_manifest_path.exists():
        try:
            m = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
            cols = [c for c in m.get("feature_cols", []) if c in df.columns]
            if cols:
                return cols
        except Exception:
            pass

    base_exclude = {
        "date", "stock_code", "industry", "is_trading", "open", "high", "low", "close", "volume", "amount", "adj_factor"
    }
    cols = []
    for c in df.columns:
        if c in base_exclude:
            continue
        if c.startswith("label_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("无法推断特征列，请显式传 --feature-cols")
    return cols


def load_split_boundaries(split_manifest_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"split_manifest 不存在: {split_manifest_path}")

    m = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    train_end = pd.to_datetime(m["train"]["end"])
    val_end = pd.to_datetime(m["val"]["end"])
    return train_end, val_end


def add_split_col(df: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out["split"] = np.where(
        out["date"] <= train_end,
        "train",
        np.where(out["date"] <= val_end, "val", "test"),
    )
    return out


def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_end: pd.Timestamp,
    retrain_every: int,
    train_window: int,
    min_train_days: int,
    model_factory: Callable[[], object],
    pred_col_name: str,
) -> pd.DataFrame:
    dates = np.array(sorted(df["date"].dropna().unique()))
    if len(dates) == 0:
        return pd.DataFrame(columns=["date", "stock_code", "split", target_col, pred_col_name])

    # 第一个预测日：train_end 之后的首个交易日（一般对应 val 起点）
    train_end_idx = int(np.searchsorted(dates, np.datetime64(train_end), side="right") - 1)
    if train_end_idx < 0:
        raise ValueError("train_end 早于数据起始日期，无法训练")

    pred_chunks = []

    for chunk_start in range(train_end_idx + 1, len(dates), max(1, retrain_every)):
        chunk_end = min(len(dates), chunk_start + max(1, retrain_every))
        pred_dates = dates[chunk_start:chunk_end]

        fit_end_idx = chunk_start - 1
        if train_window > 0:
            fit_start_idx = max(0, fit_end_idx - train_window + 1)
        else:
            fit_start_idx = 0

        fit_dates = dates[fit_start_idx : fit_end_idx + 1]
        if len(fit_dates) < min_train_days:
            continue

        train_df = df[df["date"].isin(fit_dates)]
        pred_df = df[df["date"].isin(pred_dates)]
        if train_df.empty or pred_df.empty:
            continue

        x_train = train_df[feature_cols].to_numpy(dtype=np.float64)
        y_train = train_df[target_col].to_numpy(dtype=np.float64)
        x_pred = pred_df[feature_cols].to_numpy(dtype=np.float64)

        model = model_factory()
        model.fit(x_train, y_train)
        y_hat = model.predict(x_pred)

        out = pred_df[["date", "stock_code", "split", target_col]].copy()
        out[pred_col_name] = y_hat
        out["fit_start_date"] = pd.to_datetime(fit_dates[0])
        out["fit_end_date"] = pd.to_datetime(fit_dates[-1])
        pred_chunks.append(out)

    if not pred_chunks:
        return pd.DataFrame(columns=["date", "stock_code", "split", target_col, pred_col_name])

    pred = pd.concat(pred_chunks, ignore_index=True)
    pred = pred.drop_duplicates(subset=["date", "stock_code"], keep="last")
    pred = pred.sort_values(["date", "stock_code"]).reset_index(drop=True)
    return pred


def calc_metrics(df: pd.DataFrame, pred_col: str, target_col: str, n_quantiles: int) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "days": 0,
            "rank_ic_mean": np.nan,
            "rank_ic_std": np.nan,
            "icir": np.nan,
            "direction_hit_rate": np.nan,
            "top_quantile_ret": np.nan,
            "bottom_quantile_ret": np.nan,
            "long_short_ret": np.nan,
        }

    d = df[["date", pred_col, target_col]].dropna().copy()
    if d.empty:
        return {
            "rows": 0,
            "days": 0,
            "rank_ic_mean": np.nan,
            "rank_ic_std": np.nan,
            "icir": np.nan,
            "direction_hit_rate": np.nan,
            "top_quantile_ret": np.nan,
            "bottom_quantile_ret": np.nan,
            "long_short_ret": np.nan,
        }

    daily_ic = []
    top_rets = []
    bot_rets = []
    long_short_rets = []

    for _, g in d.groupby("date"):
        if len(g) < 3:
            continue

        # RankIC（Spearman）
        if g[pred_col].nunique() > 1 and g[target_col].nunique() > 1:
            ic = g[pred_col].rank(method="average").corr(g[target_col].rank(method="average"))
            if pd.notna(ic):
                daily_ic.append(float(ic))

        # 分层收益
        try:
            q = pd.qcut(g[pred_col], q=max(2, n_quantiles), labels=False, duplicates="drop")
            if q.nunique() >= 2:
                top = float(g.loc[q == q.max(), target_col].mean())
                bot = float(g.loc[q == q.min(), target_col].mean())
                top_rets.append(top)
                bot_rets.append(bot)
                long_short_rets.append(top - bot)
        except Exception:
            pass

    ic_mean = float(np.mean(daily_ic)) if daily_ic else np.nan
    ic_std = float(np.std(daily_ic, ddof=1)) if len(daily_ic) > 1 else np.nan
    icir = float(ic_mean / ic_std) if pd.notna(ic_mean) and pd.notna(ic_std) and abs(ic_std) > 1e-12 else np.nan

    sign_pred = np.sign(d[pred_col].to_numpy(dtype=np.float64))
    sign_true = np.sign(d[target_col].to_numpy(dtype=np.float64))
    direction_hit = float((sign_pred == sign_true).mean()) if len(d) else np.nan

    return {
        "rows": int(len(d)),
        "days": int(d["date"].nunique()),
        "rank_ic_mean": ic_mean,
        "rank_ic_std": ic_std,
        "icir": icir,
        "direction_hit_rate": direction_hit,
        "top_quantile_ret": float(np.mean(top_rets)) if top_rets else np.nan,
        "bottom_quantile_ret": float(np.mean(bot_rets)) if bot_rets else np.nan,
        "long_short_ret": float(np.mean(long_short_rets)) if long_short_rets else np.nan,
    }


def build_markdown_report(
    out_path: Path,
    args,
    feature_cols: List[str],
    target_col: str,
    pred_path: Path,
    metrics_path: Path,
    metrics_df: pd.DataFrame,
    merged: pd.DataFrame,
):
    lines = []
    lines.append("# baseline_report（阶段4）")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat()}")
    lines.append("")
    lines.append("## 1. 运行配置")
    lines.append("")
    lines.append(f"- 输入特征：`{args.in_path}`")
    lines.append(f"- 切分清单：`{args.split_manifest}`")
    lines.append(f"- 目标标签：`{target_col}`")
    lines.append(f"- 特征数：`{len(feature_cols)}`")
    lines.append(f"- 滚动重训周期：`{args.retrain_every}` 日")
    lines.append(f"- 训练窗口：`{'expanding' if args.train_window <= 0 else str(args.train_window)}`")
    lines.append(f"- Ridge alpha：`{args.ridge_alpha}`")
    lines.append(
        f"- Tree depth/min_leaf/thresholds：`{args.tree_max_depth}/{args.tree_min_leaf}/{args.tree_n_thresholds}`"
    )
    lines.append("")
    lines.append("## 2. 输出文件")
    lines.append("")
    lines.append(f"- 预测明细：`{pred_path}`")
    lines.append(f"- 指标汇总：`{metrics_path}`")
    lines.append(f"- 预测样本行数：`{len(merged)}`")
    lines.append("")
    lines.append("## 3. 指标结果")
    lines.append("")

    def _fmt(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.6f}"

    for model in ["ridge", "tree"]:
        sub = metrics_df[metrics_df["model"] == model]
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| split | rows | days | rank_ic_mean | rank_ic_std | icir | direction_hit_rate | top_quantile_ret | bottom_quantile_ret | long_short_ret |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                "| {split} | {rows} | {days} | {rank_ic_mean} | {rank_ic_std} | {icir} | {direction_hit_rate} | {top_quantile_ret} | {bottom_quantile_ret} | {long_short_ret} |".format(
                    split=r["split"],
                    rows=_fmt(r["rows"]),
                    days=_fmt(r["days"]),
                    rank_ic_mean=_fmt(r["rank_ic_mean"]),
                    rank_ic_std=_fmt(r["rank_ic_std"]),
                    icir=_fmt(r["icir"]),
                    direction_hit_rate=_fmt(r["direction_hit_rate"]),
                    top_quantile_ret=_fmt(r["top_quantile_ret"]),
                    bottom_quantile_ret=_fmt(r["bottom_quantile_ret"]),
                    long_short_ret=_fmt(r["long_short_ret"]),
                )
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="阶段4：基线模型实验（Ridge + Tree，时间滚动）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例：\n"
            "  python train_baseline.py\n"
            "  python train_baseline.py --target-col label_excess_ret_10d\n"
            "  python train_baseline.py --train-window 500 --retrain-every 20\n"
        ),
    )

    p.add_argument("--in", dest="in_path", default="data/features/features_v1.parquet", help="阶段3特征标签输入")
    p.add_argument("--split-manifest", default="data/processed/split_manifest.json", help="阶段2切分清单")
    p.add_argument("--feature-manifest", default="data/features/feature_manifest.json", help="阶段3特征清单")
    p.add_argument("--feature-cols", default="", help="显式特征列（逗号分隔）；为空则自动读取 feature_manifest")

    p.add_argument("--target-col", default="label_excess_ret_5d", help="回归目标列")
    p.add_argument("--fillna", type=float, default=0.0, help="特征缺失填充值")

    p.add_argument("--retrain-every", type=int, default=20, help="滚动重训间隔（交易日）")
    p.add_argument("--train-window", type=int, default=0, help="训练窗口长度（交易日）；0 表示扩展窗口")
    p.add_argument("--min-train-days", type=int, default=240, help="最小训练交易日数")

    p.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge L2 强度")

    p.add_argument("--tree-max-depth", type=int, default=3, help="树最大深度")
    p.add_argument("--tree-min-split", type=int, default=200, help="树分裂最小样本")
    p.add_argument("--tree-min-leaf", type=int, default=80, help="树叶子最小样本")
    p.add_argument("--tree-n-thresholds", type=int, default=8, help="每特征候选阈值个数")
    p.add_argument("--tree-max-features", type=int, default=10, help="每节点采样特征数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    p.add_argument("--n-quantiles", type=int, default=5, help="分层收益分位数")

    p.add_argument("--out", default="data/baseline/baseline_result.csv", help="预测明细输出")
    p.add_argument("--metrics-out", default="data/baseline/baseline_metrics.csv", help="指标输出")
    p.add_argument("--report-out", default="data/baseline/baseline_report.md", help="实验说明输出")

    args = p.parse_args()

    in_path = Path(args.in_path)
    split_path = Path(args.split_manifest)
    feat_manifest_path = Path(args.feature_manifest)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics_out)
    report_path = Path(args.report_out)

    if not in_path.exists():
        raise SystemExit(f"[ERROR] 输入不存在: {in_path}")

    df = pd.read_parquet(in_path)
    if "date" not in df.columns or "stock_code" not in df.columns:
        raise SystemExit("[ERROR] 输入缺少 date/stock_code")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)

    if args.target_col not in df.columns:
        raise SystemExit(f"[ERROR] 目标列不存在: {args.target_col}")

    train_end, val_end = load_split_boundaries(split_path)
    df = add_split_col(df, train_end=train_end, val_end=val_end)

    feature_cols = resolve_feature_cols(df, args.feature_cols, feat_manifest_path)

    work = df[["date", "stock_code", "split", args.target_col] + feature_cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["date", "stock_code", args.target_col])
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
        work[c] = work[c].fillna(float(args.fillna))

    work = work.sort_values(["date", "stock_code"]).reset_index(drop=True)

    print(f"[INFO] input rows={len(work)} stocks={work['stock_code'].nunique()} days={work['date'].nunique()}")
    print(f"[INFO] feature_cols={len(feature_cols)} target={args.target_col}")

    ridge_pred = walk_forward_predict(
        df=work,
        feature_cols=feature_cols,
        target_col=args.target_col,
        train_end=train_end,
        retrain_every=max(1, args.retrain_every),
        train_window=max(0, args.train_window),
        min_train_days=max(1, args.min_train_days),
        model_factory=lambda: RidgeClosedForm(alpha=args.ridge_alpha),
        pred_col_name="pred_ridge",
    )

    tree_pred = walk_forward_predict(
        df=work,
        feature_cols=feature_cols,
        target_col=args.target_col,
        train_end=train_end,
        retrain_every=max(1, args.retrain_every),
        train_window=max(0, args.train_window),
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

    key_cols = ["date", "stock_code", "split", args.target_col, "fit_start_date", "fit_end_date"]
    merged = ridge_pred.merge(tree_pred, on=key_cols, how="inner")
    merged = merged.sort_values(["date", "stock_code"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")

    # 指标统计
    metrics_rows = []
    for model, pred_col in [("ridge", "pred_ridge"), ("tree", "pred_tree")]:
        for split_name in ["val", "test", "all"]:
            if split_name == "all":
                sub = merged.copy()
            else:
                sub = merged[merged["split"] == split_name].copy()
            m = calc_metrics(sub, pred_col=pred_col, target_col=args.target_col, n_quantiles=max(2, args.n_quantiles))
            m["model"] = model
            m["split"] = split_name
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
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    build_markdown_report(
        out_path=report_path,
        args=args,
        feature_cols=feature_cols,
        target_col=args.target_col,
        pred_path=out_path,
        metrics_path=metrics_path,
        metrics_df=metrics_df,
        merged=merged,
    )

    print("[DONE] stage4 baseline finished")
    print(f"  result : {out_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  report : {report_path}")
    print(f"  rows   : {len(merged)}")
    print(f"  days   : {merged['date'].nunique() if len(merged) else 0}")

    # 控制台摘要
    for model in ["ridge", "tree"]:
        m_test = metrics_df[(metrics_df["model"] == model) & (metrics_df["split"] == "test")]
        if not m_test.empty:
            r = m_test.iloc[0]
            print(
                f"  [{model}] test rank_ic={r['rank_ic_mean']:.6f} icir={r['icir']:.6f} "
                f"long_short={r['long_short_ret']:.6f} hit={r['direction_hit_rate']:.6f}"
            )


if __name__ == "__main__":
    main()
