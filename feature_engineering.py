#!/usr/bin/env python
"""
阶段3：标签与特征工程脚本（feature_engineering.py）

目标：
- 在阶段2的标准面板数据（daily_panel.parquet）上生成可训练的特征与标签。
- 严格按时间构造，不使用未来信息生成特征。

输入（默认）：
- data/processed/daily_panel.parquet

输出（默认）：
- data/features/features_v1.parquet
- data/features/feature_manifest.json
- data/features/label_spec.md

调参速查：
1) 预测周期：--horizons（默认 5,10）
2) 特征历史窗口：--min-history（默认 60）
3) 分类阈值：--cls-quantile（默认 0.3）
4) 归一化：--normalize xsec_zscore|none（默认 xsec_zscore）
5) 缺失处理：--drop-na-features（默认启用）

说明：
- 特征只由当日及过去数据计算（rolling / pct_change）。
- 标签使用未来收益（shift(-h)）构造，仅用于监督学习目标。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


FEATURE_COLS = [
    # 收益动量
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    # 波动率
    "vol_5", "vol_10", "vol_20", "vol_60",
    # 均线偏离
    "ma_gap_5", "ma_gap_10", "ma_gap_20", "ma_gap_60",
    # 均线结构
    "ma_cross_5_20", "ma_cross_10_60",
    # K线结构
    "hl_spread", "oc_change",
    # 量价变化
    "vol_chg_1", "vol_chg_5", "amt_chg_1", "amt_chg_5",
    # 量能比
    "vol_ratio_5", "vol_ratio_20",
    # 价格区间位置
    "price_pos_20", "price_pos_60",
    # 相对市场/行业
    "rel_mkt_ret_1", "rel_mkt_ret_5", "rel_mkt_ret_20",
    "rel_ind_ret_1", "rel_ind_ret_5",
]


def _parse_horizons(text: str) -> List[int]:
    hs = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        h = int(x)
        if h <= 0:
            continue
        hs.append(h)
    hs = sorted(list(dict.fromkeys(hs)))
    return hs or [5, 10]


def _load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    df = pd.read_parquet(path)
    required = {"date", "stock_code", "open", "high", "low", "close", "volume", "amount"}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"输入缺少必要字段: {miss}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)

    for c in ["open", "high", "low", "close", "volume", "amount"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # 仅保留可形成有效K线的行
    out = out.dropna(subset=["date", "stock_code", "close"])
    out = out.sort_values(["stock_code", "date"]).reset_index(drop=True)
    return out


def _group_rolling_mean(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("stock_code")[col].transform(lambda x: x.rolling(window, min_periods=window).mean())


def _group_rolling_std(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("stock_code")[col].transform(lambda x: x.rolling(window, min_periods=window).std())


def _group_rolling_min(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("stock_code")[col].transform(lambda x: x.rolling(window, min_periods=window).min())


def _group_rolling_max(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("stock_code")[col].transform(lambda x: x.rolling(window, min_periods=window).max())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("stock_code")

    # 收益动量
    out["ret_1"] = g["close"].pct_change(1)
    out["ret_3"] = g["close"].pct_change(3)
    out["ret_5"] = g["close"].pct_change(5)
    out["ret_10"] = g["close"].pct_change(10)
    out["ret_20"] = g["close"].pct_change(20)
    out["ret_60"] = g["close"].pct_change(60)

    # 波动率
    out["vol_5"] = _group_rolling_std(out, "ret_1", 5)
    out["vol_10"] = _group_rolling_std(out, "ret_1", 10)
    out["vol_20"] = _group_rolling_std(out, "ret_1", 20)
    out["vol_60"] = _group_rolling_std(out, "ret_1", 60)

    # 均线
    ma5 = _group_rolling_mean(out, "close", 5)
    ma10 = _group_rolling_mean(out, "close", 10)
    ma20 = _group_rolling_mean(out, "close", 20)
    ma60 = _group_rolling_mean(out, "close", 60)

    out["ma_gap_5"] = out["close"] / ma5 - 1
    out["ma_gap_10"] = out["close"] / ma10 - 1
    out["ma_gap_20"] = out["close"] / ma20 - 1
    out["ma_gap_60"] = out["close"] / ma60 - 1

    out["ma_cross_5_20"] = ma5 / ma20 - 1
    out["ma_cross_10_60"] = ma10 / ma60 - 1

    # K线结构
    out["hl_spread"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_change"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

    # 量价变化
    out["vol_chg_1"] = g["volume"].pct_change(1)
    out["vol_chg_5"] = g["volume"].pct_change(5)
    out["amt_chg_1"] = g["amount"].pct_change(1)
    out["amt_chg_5"] = g["amount"].pct_change(5)

    vol5 = _group_rolling_mean(out, "volume", 5)
    vol20 = _group_rolling_mean(out, "volume", 20)
    out["vol_ratio_5"] = out["volume"] / vol5.replace(0, np.nan)
    out["vol_ratio_20"] = out["volume"] / vol20.replace(0, np.nan)

    # 区间位置
    low20 = _group_rolling_min(out, "close", 20)
    high20 = _group_rolling_max(out, "close", 20)
    low60 = _group_rolling_min(out, "close", 60)
    high60 = _group_rolling_max(out, "close", 60)
    out["price_pos_20"] = (out["close"] - low20) / (high20 - low20).replace(0, np.nan)
    out["price_pos_60"] = (out["close"] - low60) / (high60 - low60).replace(0, np.nan)

    # 相对市场
    market = out.groupby("date")[["ret_1", "ret_5", "ret_20"]].mean().rename(
        columns={"ret_1": "mkt_ret_1", "ret_5": "mkt_ret_5", "ret_20": "mkt_ret_20"}
    )
    out = out.merge(market, left_on="date", right_index=True, how="left")
    out["rel_mkt_ret_1"] = out["ret_1"] - out["mkt_ret_1"]
    out["rel_mkt_ret_5"] = out["ret_5"] - out["mkt_ret_5"]
    out["rel_mkt_ret_20"] = out["ret_20"] - out["mkt_ret_20"]

    # 相对行业（无行业时回退市场）
    if "industry" in out.columns and out["industry"].notna().any():
        ind = out.dropna(subset=["industry"]).groupby(["date", "industry"])[["ret_1", "ret_5"]].mean().rename(
            columns={"ret_1": "ind_ret_1", "ret_5": "ind_ret_5"}
        )
        out = out.merge(ind, left_on=["date", "industry"], right_index=True, how="left")
        out["ind_ret_1"] = out["ind_ret_1"].fillna(out["mkt_ret_1"])
        out["ind_ret_5"] = out["ind_ret_5"].fillna(out["mkt_ret_5"])
    else:
        out["ind_ret_1"] = out["mkt_ret_1"]
        out["ind_ret_5"] = out["mkt_ret_5"]

    out["rel_ind_ret_1"] = out["ret_1"] - out["ind_ret_1"]
    out["rel_ind_ret_5"] = out["ret_5"] - out["ind_ret_5"]

    # 清理中间列
    out = out.drop(columns=["mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "ind_ret_1", "ind_ret_5"], errors="ignore")
    return out


def add_labels(df: pd.DataFrame, horizons: List[int], cls_quantile: float) -> tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    g = out.groupby("stock_code")

    label_cols = []

    for h in horizons:
        fwd_col = f"label_fwd_ret_{h}d"
        bench_col = f"label_bench_ret_{h}d"
        excess_col = f"label_excess_ret_{h}d"
        rank_col = f"label_rank_pct_{h}d"
        cls_col = f"label_cls_{h}d"

        out[fwd_col] = g["close"].shift(-h) / out["close"] - 1
        out[bench_col] = out.groupby("date")[fwd_col].transform("mean")
        out[excess_col] = out[fwd_col] - out[bench_col]

        out[rank_col] = out.groupby("date")[excess_col].rank(method="average", pct=True)
        q = float(cls_quantile)
        out[cls_col] = np.where(
            out[rank_col] >= (1 - q),
            1.0,
            np.where(out[rank_col] <= q, 0.0, np.nan),
        )

        label_cols.extend([fwd_col, bench_col, excess_col, rank_col, cls_col])

    return out, label_cols


def apply_normalization(df: pd.DataFrame, feature_cols: List[str], mode: str) -> pd.DataFrame:
    out = df.copy()
    if mode == "none":
        return out

    if mode != "xsec_zscore":
        raise ValueError(f"不支持的 normalize 模式: {mode}")

    for c in feature_cols:
        mean = out.groupby("date")[c].transform("mean")
        std = out.groupby("date")[c].transform("std")
        std = std.replace(0, np.nan)
        out[c] = (out[c] - mean) / std
    return out


def write_label_spec(path: Path, horizons: List[int], cls_quantile: float, normalize: str):
    htext = ", ".join([str(h) for h in horizons])
    content = f"""# label_spec（阶段3）

## 标签定义

当前标签使用未来收益构造，预测周期：`{htext}` 日。

对每个 horizon=h，定义：
- `label_fwd_ret_{{h}}d`：个股未来 h 日收益
- `label_bench_ret_{{h}}d`：当日横截面平均未来 h 日收益（基准）
- `label_excess_ret_{{h}}d`：超额收益 = 个股未来收益 - 基准未来收益
- `label_rank_pct_{{h}}d`：超额收益横截面分位（0~1）
- `label_cls_{{h}}d`：二分类标签（上分位=1，下分位=0，中间为空）

二分类分位阈值：`{cls_quantile}`
- 上分位阈值：`>= {1-cls_quantile:.2f}`
- 下分位阈值：`<= {cls_quantile:.2f}`

## 特征归一化

- normalize 模式：`{normalize}`
- `xsec_zscore`：按日期做横截面 z-score（不跨期）
- `none`：不做归一化
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="阶段3：标签与特征工程",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例：\n"
            "  python feature_engineering.py\n"
            "  python feature_engineering.py --horizons 5,10,20 --min-history 80\n"
            "  python feature_engineering.py --normalize none --drop-na-features\n"
            "\n"
            "调参建议：\n"
            "  1) 先固定 horizons=5,10 验证流程\n"
            "  2) 稳定后再扩到 20 日标签\n"
            "  3) min-history 建议 60~120\n"
        ),
    )
    p.add_argument("--in", dest="in_path", default="data/processed/daily_panel.parquet", help="阶段2面板输入路径")
    p.add_argument("--out", dest="out_path", default="data/features/features_v1.parquet", help="特征标签表输出路径")
    p.add_argument("--manifest-out", default="data/features/feature_manifest.json", help="特征清单输出路径")
    p.add_argument("--label-spec-out", default="data/features/label_spec.md", help="标签说明输出路径")

    p.add_argument("--horizons", default="5,10", help="标签预测周期（逗号分隔）")
    p.add_argument("--cls-quantile", type=float, default=0.3, help="二分类上下分位阈值（0~0.5）")
    p.add_argument("--min-history", type=int, default=60, help="每只股票最小历史长度过滤阈值")
    p.add_argument("--normalize", choices=["xsec_zscore", "none"], default="xsec_zscore", help="特征归一化方式")
    p.add_argument("--drop-na-features", action="store_true", help="删除含缺失特征或核心标签的样本")
    args = p.parse_args()

    if not (0.0 < args.cls_quantile < 0.5):
        raise SystemExit("[ERROR] --cls-quantile 必须在 (0,0.5) 区间")

    horizons = _parse_horizons(args.horizons)

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    manifest_path = Path(args.manifest_out)
    label_spec_path = Path(args.label_spec_out)

    panel = _load_panel(in_path)
    print(f"[INFO] input rows={len(panel)} stocks={panel['stock_code'].nunique()} days={panel['date'].nunique()}")

    feat = add_features(panel)
    feat, label_cols = add_labels(feat, horizons=horizons, cls_quantile=args.cls_quantile)

    # 最小历史长度过滤（避免前期大量 rolling 缺失）
    feat["_obs"] = feat.groupby("stock_code").cumcount()
    feat = feat[feat["_obs"] >= max(0, int(args.min_history))].copy()
    feat = feat.drop(columns=["_obs"], errors="ignore")

    # 归一化
    feat = apply_normalization(feat, feature_cols=FEATURE_COLS, mode=args.normalize)

    # 清理无限值
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # 默认核心标签以第一个 horizon 作为训练主标签
    primary_excess = f"label_excess_ret_{horizons[0]}d"
    if args.drop_na_features:
        feat = feat.dropna(subset=FEATURE_COLS + [primary_excess])
    else:
        feat = feat.dropna(subset=[primary_excess])

    # 固定列顺序
    base_cols = [c for c in ["date", "stock_code", "industry", "is_trading", "open", "high", "low", "close", "volume", "amount", "adj_factor"] if c in feat.columns]
    ordered_cols = base_cols + FEATURE_COLS + label_cols
    ordered_cols = [c for c in ordered_cols if c in feat.columns]
    feat = feat[ordered_cols].sort_values(["date", "stock_code"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False)

    write_label_spec(label_spec_path, horizons=horizons, cls_quantile=args.cls_quantile, normalize=args.normalize)

    missing_ratio = {
        c: float(feat[c].isna().mean())
        for c in FEATURE_COLS
        if c in feat.columns
    }

    industry_nonnull_ratio = float(1.0 - feat["industry"].isna().mean()) if "industry" in feat.columns and len(feat) else 0.0
    rel_ind_eq_mkt_ratio = {}
    for h in [1, 5]:
        a = f"rel_ind_ret_{h}"
        b = f"rel_mkt_ret_{h}"
        if a in feat.columns and b in feat.columns and len(feat):
            rel_ind_eq_mkt_ratio[f"{h}d"] = float((feat[a].fillna(0).round(10) == feat[b].fillna(0).round(10)).mean())

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "input_path": str(in_path),
        "output_path": str(out_path),
        "rows": int(len(feat)),
        "stocks": int(feat["stock_code"].nunique()) if "stock_code" in feat.columns else 0,
        "days": int(feat["date"].nunique()) if "date" in feat.columns else 0,
        "date_start": str(pd.to_datetime(feat["date"].min()).date()) if len(feat) else "",
        "date_end": str(pd.to_datetime(feat["date"].max()).date()) if len(feat) else "",
        "horizons": horizons,
        "primary_label": primary_excess,
        "normalize": args.normalize,
        "min_history": int(args.min_history),
        "drop_na_features": bool(args.drop_na_features),
        "feature_cols": FEATURE_COLS,
        "label_cols": label_cols,
        "feature_missing_ratio": missing_ratio,
        "industry_nonnull_ratio": industry_nonnull_ratio,
        "rel_ind_eq_rel_mkt_ratio": rel_ind_eq_mkt_ratio,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] stage3 feature engineering built")
    print(f"  features : {out_path}")
    print(f"  manifest : {manifest_path}")
    print(f"  labelspec: {label_spec_path}")
    print(f"  rows     : {len(feat)}")
    print(f"  stocks   : {feat['stock_code'].nunique() if 'stock_code' in feat.columns else 0}")
    print(f"  days     : {feat['date'].nunique() if 'date' in feat.columns else 0}")
    print(f"  industry_nonnull_ratio: {industry_nonnull_ratio:.4f}")
    if rel_ind_eq_mkt_ratio:
        print(f"  rel_ind_eq_rel_mkt_ratio: {rel_ind_eq_mkt_ratio}")


if __name__ == "__main__":
    main()
