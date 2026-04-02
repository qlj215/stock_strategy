#!/usr/bin/env python
"""
阶段5：深度学习模型第一版（LSTM，train_lstm.py）

目标：
- 用阶段3特征（features_v1.parquet）构造固定长度时序样本（默认 lookback=20 日）
- 使用 LSTM 预测未来 5 日超额收益（默认 target=label_excess_ret_5d）
- 按与阶段4一致的口径输出 val/test/all 指标（RankIC、ICIR、Long-Short、命中率）

输入（默认）：
- data/features/features_v1.parquet
- data/processed/split_manifest.json
- data/features/feature_manifest.json

输出（默认）：
- data/dl/dl_result.csv
- data/dl/dl_metrics.csv
- data/dl/dl_trainlog.csv
- data/dl/dl_report.md

调参速查（先调这些）：
1) 时序长度与滚动：
   - --seq-len 20~60
   - --retrain-every 20~60（重训越频繁越慢）
   - --train-window 0（扩展窗口）或 500~1000（固定近期窗口）
2) 网络规模：
   - --hidden-size 32/64/128
   - --num-layers 1/2
   - --dropout 0.0~0.3
3) 训练稳定性：
   - --loss huber（默认）或 mse
   - --lr 1e-3（常用起点）
   - --epochs / --early-stop-patience
4) 内部早停验证：
   - --inner-val-days 40~80（从历史训练窗口尾部切出）

运行示例：
- 默认（先打通链路）：
  python train_lstm.py

- 更轻量快速（重训少、epoch 少）：
  python train_lstm.py --retrain-every 60 --epochs 8

- 近期窗口版本（应对时变）：
  python train_lstm.py --train-window 750 --retrain-every 20
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_baseline import (
    add_split_col,
    calc_metrics,
    load_split_boundaries,
    resolve_feature_cols,
)


@dataclass
class SequenceBundle:
    x: np.ndarray  # [N, T, F], float32
    y: np.ndarray  # [N], float32
    dates: np.ndarray  # [N], datetime64[ns]
    stock_codes: np.ndarray  # [N], str
    splits: np.ndarray  # [N], str


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        mid = max(8, hidden_size // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        y_hat = self.head(h_last).squeeze(-1)
        return y_hat


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("[ERROR] 指定 --device cuda 但当前不可用")
        return torch.device("cuda")
    return torch.device("cpu")


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
) -> SequenceBundle:
    x_list = []
    y_list = []
    d_list = []
    c_list = []
    s_list = []

    for stock_code, g in df.groupby("stock_code", sort=False):
        g = g.sort_values("date")
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        ys = g[target_col].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy(dtype="datetime64[ns]")
        splits = g["split"].astype(str).to_numpy()

        n = len(g)
        if n < seq_len:
            continue

        for end_idx in range(seq_len - 1, n):
            start_idx = end_idx - seq_len + 1
            x_list.append(feats[start_idx : end_idx + 1])
            y_list.append(ys[end_idx])
            d_list.append(dates[end_idx])
            c_list.append(str(stock_code))
            s_list.append(str(splits[end_idx]))

    if not x_list:
        raise ValueError("时序样本为空，请检查 --seq-len 或输入数据")

    x = np.stack(x_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    d = np.asarray(d_list, dtype="datetime64[ns]")
    c = np.asarray(c_list, dtype=object)
    s = np.asarray(s_list, dtype=object)

    return SequenceBundle(x=x, y=y, dates=d, stock_codes=c, splits=s)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else np.nan


def train_one_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args,
    device: torch.device,
):
    model = LSTMRegressor(
        input_size=x_train.shape[2],
        hidden_size=max(8, int(args.hidden_size)),
        num_layers=max(1, int(args.num_layers)),
        dropout=max(0.0, float(args.dropout)),
    ).to(device)

    if args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.HuberLoss(delta=float(args.huber_delta))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    use_val = len(x_val) > 0
    if use_val:
        val_ds = TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    else:
        val_loader = None

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = np.inf
    bad_epochs = 0

    history = []

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))

            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan

        if use_val and val_loader is not None:
            score = evaluate_loss(model, val_loader, criterion=criterion, device=device)
        else:
            score = train_loss

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": score if use_val else np.nan,
            }
        )

        if pd.notna(score) and (best_score - score) > float(args.early_stop_min_delta):
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if use_val and bad_epochs >= max(1, int(args.early_stop_patience)):
            break

    model.load_state_dict(best_state)
    fit_info = {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score) if pd.notna(best_score) else np.nan,
        "trained_epochs": int(history[-1]["epoch"]) if history else 0,
    }
    return model, history, fit_info


def predict_lstm(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.float32)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), max(1, int(batch_size))):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32).to(device)
            y_hat = model(xb).detach().cpu().numpy().astype(np.float32)
            preds.append(y_hat)

    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float32)


def walk_forward_lstm_predict(
    bundle: SequenceBundle,
    target_col: str,
    train_end: pd.Timestamp,
    args,
    device: torch.device,
):
    dates = np.sort(np.unique(bundle.dates))
    if len(dates) == 0:
        return pd.DataFrame(), pd.DataFrame()

    train_end_np = np.datetime64(pd.to_datetime(train_end).to_datetime64())
    train_end_idx = int(np.searchsorted(dates, train_end_np, side="right") - 1)
    if train_end_idx < 0:
        raise ValueError("train_end 早于样本起始日期，无法训练")

    pred_chunks = []
    trainlog_rows = []

    step = max(1, int(args.retrain_every))

    for chunk_start in range(train_end_idx + 1, len(dates), step):
        chunk_end = min(len(dates), chunk_start + step)
        pred_dates = dates[chunk_start:chunk_end]
        if len(pred_dates) == 0:
            continue

        fit_end_idx = chunk_start - 1
        if int(args.train_window) > 0:
            fit_start_idx = max(0, fit_end_idx - int(args.train_window) + 1)
        else:
            fit_start_idx = 0

        fit_dates = dates[fit_start_idx : fit_end_idx + 1]
        if len(fit_dates) < max(1, int(args.min_train_days)):
            continue

        fit_mask = np.isin(bundle.dates, fit_dates)
        pred_mask = np.isin(bundle.dates, pred_dates)
        if fit_mask.sum() == 0 or pred_mask.sum() == 0:
            continue

        # 内部验证：从训练历史尾部切 inner_val_days 天
        train_mask = fit_mask.copy()
        val_mask = np.zeros_like(fit_mask)
        inner_val_days = max(0, int(args.inner_val_days))
        if inner_val_days > 0 and len(fit_dates) > inner_val_days + 5:
            val_dates = fit_dates[-inner_val_days:]
            val_mask = np.isin(bundle.dates, val_dates) & fit_mask
            train_mask = fit_mask & (~val_mask)
            if train_mask.sum() < max(1, int(args.min_train_samples)):
                train_mask = fit_mask
                val_mask = np.zeros_like(fit_mask)

        if train_mask.sum() < max(1, int(args.min_train_samples)):
            continue

        x_train = bundle.x[train_mask]
        y_train = bundle.y[train_mask]
        x_val = bundle.x[val_mask]
        y_val = bundle.y[val_mask]

        model, _history, fit_info = train_one_lstm(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            args=args,
            device=device,
        )

        x_pred = bundle.x[pred_mask]
        y_pred = predict_lstm(
            model=model,
            x=x_pred,
            batch_size=max(1, int(args.pred_batch_size)),
            device=device,
        )

        chunk_df = pd.DataFrame(
            {
                "date": pd.to_datetime(bundle.dates[pred_mask]),
                "stock_code": bundle.stock_codes[pred_mask],
                "split": bundle.splits[pred_mask],
                target_col: bundle.y[pred_mask],
                "pred_lstm": y_pred,
                "fit_start_date": pd.to_datetime(fit_dates[0]),
                "fit_end_date": pd.to_datetime(fit_dates[-1]),
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "best_epoch": int(fit_info.get("best_epoch", 0)),
                "trained_epochs": int(fit_info.get("trained_epochs", 0)),
                "best_score": float(fit_info.get("best_score", np.nan)),
            }
        )
        pred_chunks.append(chunk_df)

        trainlog_rows.append(
            {
                "pred_start_date": pd.to_datetime(pred_dates[0]),
                "pred_end_date": pd.to_datetime(pred_dates[-1]),
                "fit_start_date": pd.to_datetime(fit_dates[0]),
                "fit_end_date": pd.to_datetime(fit_dates[-1]),
                "train_days": int(len(fit_dates)),
                "pred_days": int(len(pred_dates)),
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "best_epoch": int(fit_info.get("best_epoch", 0)),
                "trained_epochs": int(fit_info.get("trained_epochs", 0)),
                "best_score": float(fit_info.get("best_score", np.nan)),
            }
        )

    if not pred_chunks:
        pred_df = pd.DataFrame(
            columns=[
                "date",
                "stock_code",
                "split",
                target_col,
                "pred_lstm",
                "fit_start_date",
                "fit_end_date",
                "train_samples",
                "val_samples",
                "best_epoch",
                "trained_epochs",
                "best_score",
            ]
        )
    else:
        pred_df = pd.concat(pred_chunks, ignore_index=True)
        pred_df = pred_df.drop_duplicates(subset=["date", "stock_code"], keep="last")
        pred_df = pred_df.sort_values(["date", "stock_code"]).reset_index(drop=True)

    trainlog_df = pd.DataFrame(trainlog_rows)
    if not trainlog_df.empty:
        trainlog_df = trainlog_df.sort_values("pred_start_date").reset_index(drop=True)

    return pred_df, trainlog_df


def build_markdown_report(
    out_path: Path,
    args,
    feature_cols: List[str],
    target_col: str,
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    trainlog_df: pd.DataFrame,
):
    def _fmt(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.6f}"

    lines = []
    lines.append("# dl_report（阶段5：LSTM 第一版）")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat()}")
    lines.append("")

    lines.append("## 1. 运行配置")
    lines.append("")
    lines.append(f"- 输入特征：`{args.in_path}`")
    lines.append(f"- split_manifest：`{args.split_manifest}`")
    lines.append(f"- 目标标签：`{target_col}`")
    lines.append(f"- 特征数：`{len(feature_cols)}`")
    lines.append(f"- seq_len：`{args.seq_len}`")
    lines.append(f"- retrain_every：`{args.retrain_every}`")
    lines.append(f"- train_window：`{'expanding' if args.train_window <= 0 else str(args.train_window)}`")
    lines.append(f"- inner_val_days：`{args.inner_val_days}`")
    lines.append(f"- hidden_size/num_layers/dropout：`{args.hidden_size}/{args.num_layers}/{args.dropout}`")
    lines.append(f"- loss：`{args.loss}`")
    lines.append(f"- lr/weight_decay：`{args.lr}/{args.weight_decay}`")
    lines.append(f"- epochs/patience：`{args.epochs}/{args.early_stop_patience}`")
    lines.append("")

    lines.append("## 2. 输出规模")
    lines.append("")
    lines.append(f"- 预测样本行数：`{len(pred_df)}`")
    lines.append(f"- 覆盖交易日：`{pred_df['date'].nunique() if len(pred_df) else 0}`")
    lines.append(f"- 重训轮次：`{len(trainlog_df)}`")
    if not trainlog_df.empty:
        lines.append(f"- 平均 best_epoch：`{trainlog_df['best_epoch'].mean():.2f}`")
        lines.append(f"- 平均 best_score：`{trainlog_df['best_score'].mean():.6f}`")
    lines.append("")

    lines.append("## 3. 指标结果")
    lines.append("")
    lines.append("| model | split | rows | days | rank_ic_mean | rank_ic_std | icir | direction_hit_rate | top_quantile_ret | bottom_quantile_ret | long_short_ret |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in metrics_df.iterrows():
        lines.append(
            "| {model} | {split} | {rows} | {days} | {rank_ic_mean} | {rank_ic_std} | {icir} | {direction_hit_rate} | {top_quantile_ret} | {bottom_quantile_ret} | {long_short_ret} |".format(
                model=r["model"],
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

    lines.append("## 4. 备注")
    lines.append("")
    lines.append("- 当前版本以“链路打通 + 可复现”为优先，后续可继续做结构与超参增强。")
    lines.append("- 指标口径复用阶段4 calc_metrics，便于与 baseline 同口径对照。")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="阶段5：LSTM 第一版（时序输入 + 滚动重训 + 同口径指标）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例：\n"
            "  python train_lstm.py\n"
            "  python train_lstm.py --retrain-every 60 --epochs 8\n"
            "  python train_lstm.py --train-window 750 --retrain-every 20\n"
        ),
    )

    p.add_argument("--in", dest="in_path", default="data/features/features_v1.parquet", help="阶段3特征标签输入")
    p.add_argument("--split-manifest", default="data/processed/split_manifest.json", help="阶段2切分清单")
    p.add_argument("--feature-manifest", default="data/features/feature_manifest.json", help="阶段3特征清单")
    p.add_argument("--feature-cols", default="", help="显式特征列（逗号分隔）；为空则自动读取 feature_manifest")

    p.add_argument("--target-col", default="label_excess_ret_5d", help="回归目标列")
    p.add_argument("--fillna", type=float, default=0.0, help="特征缺失填充值")

    p.add_argument("--seq-len", type=int, default=20, help="时序窗口长度（交易日）")

    p.add_argument("--retrain-every", type=int, default=40, help="滚动重训间隔（交易日）")
    p.add_argument("--train-window", type=int, default=0, help="训练窗口长度（交易日）；0=扩展窗口")
    p.add_argument("--min-train-days", type=int, default=240, help="最小训练交易日数")
    p.add_argument("--min-train-samples", type=int, default=1200, help="最小训练样本数")
    p.add_argument("--inner-val-days", type=int, default=60, help="内部早停验证天数（从历史窗口尾部切出）")

    p.add_argument("--hidden-size", type=int, default=64, help="LSTM 隐层维度")
    p.add_argument("--num-layers", type=int, default=2, help="LSTM 层数")
    p.add_argument("--dropout", type=float, default=0.1, help="dropout")

    p.add_argument("--loss", choices=["mse", "huber"], default="huber", help="训练损失")
    p.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta")

    p.add_argument("--epochs", type=int, default=12, help="每次重训最大 epoch")
    p.add_argument("--batch-size", type=int, default=256, help="训练 batch size")
    p.add_argument("--pred-batch-size", type=int, default=1024, help="预测 batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪阈值（<=0 关闭）")
    p.add_argument("--early-stop-patience", type=int, default=4, help="早停耐心轮数")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-5, help="早停最小改善")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="训练设备")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--n-quantiles", type=int, default=5, help="分层收益分位数")

    p.add_argument("--out", default="data/dl/dl_result.csv", help="预测输出")
    p.add_argument("--metrics-out", default="data/dl/dl_metrics.csv", help="指标输出")
    p.add_argument("--trainlog-out", default="data/dl/dl_trainlog.csv", help="训练日志输出")
    p.add_argument("--report-out", default="data/dl/dl_report.md", help="实验报告输出")

    args = p.parse_args()

    if int(args.seq_len) < 2:
        raise SystemExit("[ERROR] --seq-len 至少为 2")

    set_global_seed(int(args.seed))
    device = resolve_device(args.device)

    in_path = Path(args.in_path)
    split_path = Path(args.split_manifest)
    feat_manifest_path = Path(args.feature_manifest)

    out_path = Path(args.out)
    metrics_path = Path(args.metrics_out)
    trainlog_path = Path(args.trainlog_out)
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

    keep_cols = ["date", "stock_code", "split", args.target_col] + feature_cols
    work = df[keep_cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["date", "stock_code", args.target_col])

    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(float(args.fillna))

    work = work.sort_values(["stock_code", "date"]).reset_index(drop=True)

    print(f"[INFO] device={device}")
    print(f"[INFO] rows={len(work)} stocks={work['stock_code'].nunique()} days={work['date'].nunique()}")
    print(f"[INFO] features={len(feature_cols)} target={args.target_col} seq_len={args.seq_len}")

    bundle = build_sequences(
        df=work,
        feature_cols=feature_cols,
        target_col=args.target_col,
        seq_len=max(2, int(args.seq_len)),
    )
    print(f"[INFO] sequence_samples={len(bundle.y)}")

    pred_df, trainlog_df = walk_forward_lstm_predict(
        bundle=bundle,
        target_col=args.target_col,
        train_end=train_end,
        args=args,
        device=device,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False, encoding="utf-8")

    trainlog_path.parent.mkdir(parents=True, exist_ok=True)
    trainlog_df.to_csv(trainlog_path, index=False, encoding="utf-8")

    metrics_rows = []
    for split_name in ["val", "test", "all"]:
        sub = pred_df if split_name == "all" else pred_df[pred_df["split"] == split_name]
        m = calc_metrics(
            sub,
            pred_col="pred_lstm",
            target_col=args.target_col,
            n_quantiles=max(2, int(args.n_quantiles)),
        )
        m["model"] = "lstm"
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
        pred_df=pred_df,
        metrics_df=metrics_df,
        trainlog_df=trainlog_df,
    )

    print("[DONE] stage5 lstm finished")
    print(f"  result : {out_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  trainlog: {trainlog_path}")
    print(f"  report : {report_path}")
    print(f"  rows   : {len(pred_df)}")
    print(f"  days   : {pred_df['date'].nunique() if len(pred_df) else 0}")

    m_test = metrics_df[(metrics_df["model"] == "lstm") & (metrics_df["split"] == "test")]
    if not m_test.empty:
        r = m_test.iloc[0]
        print(
            f"  [lstm] test rank_ic={r['rank_ic_mean']:.6f} icir={r['icir']:.6f} "
            f"long_short={r['long_short_ret']:.6f} hit={r['direction_hit_rate']:.6f}"
        )


if __name__ == "__main__":
    main()
