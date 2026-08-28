"""
真实 DL 推理入口（PyTorch 版）。

用途：
- 作为 `probability_backend` 的 dl 后端入口
- 支持模型文件加载 + 特征预处理 + batch 推理
- 当模型文件缺失时，自动训练一个轻量 bootstrap 模型并保存

配置（config/model_backend.json）：
{
  "default_backend": "rule",
  "dl_entrypoint": "stock_strategy.dl_entry_template:predict_proba"
}

可选环境变量：
- STOCK_DL_MODEL_PATH: 自定义模型文件路径（默认 stock_strategy/models/simple_prob_mlp.pt）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch 未安装或当前服务进程无法导入 PyTorch，无法启用 DL 后端。"
        f"原始错误：{type(e).__name__}: {e}"
    ) from e


WINDOW = 30
LONG_HORIZON = 20
FEATURE_NAMES = [
    "ret1",
    "ret3",
    "ret5",
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    "vol5_ratio",
    "vol20_ratio",
    "hl_spread",
    "oc_change",
]

BOOTSTRAP_SYMBOLS = ["000001", "600519", "000858", "600036", "300750", "002594", "600276", "603986"]


def _default_model_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "models", "simple_prob_mlp.pt")


def _model_path() -> str:
    return os.getenv("STOCK_DL_MODEL_PATH", "").strip() or _default_model_path()


@dataclass
class ModelBundle:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str]
    window: int
    meta: Dict[str, Any]


class SimpleProbMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.sigmoid(logits)


def _safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.fillna(default)


def _ensure_daily_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    d = daily_df.copy()

    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    else:
        d = d.reset_index().rename(columns={d.index.name or d.columns[0]: "date"})
        d["date"] = pd.to_datetime(d["date"], errors="coerce")

    if "close" not in d.columns:
        # 尝试从最后一列兜底，不建议长期依赖
        d["close"] = _safe_numeric(d.iloc[:, -1], default=np.nan)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in d.columns:
            if col == "volume":
                d[col] = 0.0
            else:
                d[col] = d["close"]
        d[col] = _safe_numeric(d[col], default=0.0 if col == "volume" else np.nan)

    d = d.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return d[["date", "open", "high", "low", "close", "volume"]]


def _feature_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_daily_df(daily_df)
    if d.empty:
        return d

    close = d["close"].astype(float)
    volume = d["volume"].astype(float).replace(0, np.nan)

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()

    vol5 = volume.rolling(5).mean()
    vol20 = volume.rolling(20).mean()

    out = pd.DataFrame({
        "date": d["date"],
        "close": close,
        "ret1": close.pct_change(1),
        "ret3": close.pct_change(3),
        "ret5": close.pct_change(5),
        "ma5_gap": (close / ma5) - 1,
        "ma10_gap": (close / ma10) - 1,
        "ma20_gap": (close / ma20) - 1,
        "vol5_ratio": volume / vol5,
        "vol20_ratio": volume / vol20,
        "hl_spread": (d["high"] - d["low"]) / close.replace(0, np.nan),
        "oc_change": (close - d["open"]) / d["open"].replace(0, np.nan),
    })

    out = out.replace([np.inf, -np.inf], np.nan)
    out[FEATURE_NAMES] = out[FEATURE_NAMES].ffill().bfill().fillna(0.0)
    return out


def _extract_latest_feature_vector(daily_df: pd.DataFrame, window: int = WINDOW) -> Optional[np.ndarray]:
    f = _feature_frame(daily_df)
    if f.empty or len(f) < window:
        return None

    x = f[FEATURE_NAMES].iloc[-window:].values.astype(np.float32)
    return x.reshape(-1)


def _build_training_samples(daily_df: pd.DataFrame, window: int = WINDOW, long_horizon: int = LONG_HORIZON) -> Tuple[np.ndarray, np.ndarray]:
    f = _feature_frame(daily_df)
    if f.empty:
        return np.empty((0, len(FEATURE_NAMES) * window), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    close = f["close"].values.astype(np.float32)
    feat = f[FEATURE_NAMES].values.astype(np.float32)

    X: List[np.ndarray] = []
    Y: List[np.ndarray] = []

    max_i = len(f) - long_horizon - 1
    for i in range(window - 1, max_i):
        c0 = close[i]
        if c0 <= 0:
            continue

        y1 = 1.0 if close[i + 1] > c0 else 0.0
        y5 = 1.0 if close[i + 5] > c0 else 0.0
        yl = 1.0 if close[i + long_horizon] > c0 else 0.0

        x = feat[i - window + 1 : i + 1].reshape(-1)
        X.append(x)
        Y.append(np.array([y1, y5, yl], dtype=np.float32))

    if not X:
        return np.empty((0, len(FEATURE_NAMES) * window), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)


def _collect_bootstrap_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    from stock_strategy.data.fetcher import fetch_stock_data

    today = datetime.now().strftime("%Y%m%d")
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    used_symbols: List[str] = []

    for s in BOOTSTRAP_SYMBOLS:
        try:
            df = fetch_stock_data(s, "20190101", today, adjust="qfq", retries=2)
            if df is None or df.empty:
                continue

            # fetch_stock_data 返回 index=date 的 OHLCV
            d = df.copy().reset_index()
            if "date" not in d.columns:
                d = d.rename(columns={d.columns[0]: "date"})

            if "volume" not in d.columns:
                d["volume"] = 0.0
            if "open" not in d.columns:
                d["open"] = d["close"]
            if "high" not in d.columns:
                d["high"] = d["close"]
            if "low" not in d.columns:
                d["low"] = d["close"]

            X, Y = _build_training_samples(d)
            if len(X) == 0:
                continue

            all_x.append(X)
            all_y.append(Y)
            used_symbols.append(s)
        except Exception:
            continue

    if not all_x:
        # 极端兜底：生成微型随机集，确保有可用 .pt 文件
        x = np.random.randn(1024, len(FEATURE_NAMES) * WINDOW).astype(np.float32)
        y = (np.random.rand(1024, 3) > 0.5).astype(np.float32)
        return x, y, {"symbols": [], "synthetic": True}

    X_all = np.concatenate(all_x, axis=0)
    Y_all = np.concatenate(all_y, axis=0)

    # 限制样本上限，避免初次训练过慢
    max_samples = 60000
    if len(X_all) > max_samples:
        idx = np.random.choice(len(X_all), size=max_samples, replace=False)
        X_all = X_all[idx]
        Y_all = Y_all[idx]

    return X_all, Y_all, {"symbols": used_symbols, "synthetic": False}


def _train_bootstrap_model(model_path: str) -> Dict[str, Any]:
    X, Y, data_meta = _collect_bootstrap_dataset()

    input_dim = X.shape[1]
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    Xn = ((X - mean) / std).astype(np.float32)

    tensor_x = torch.from_numpy(Xn)
    tensor_y = torch.from_numpy(Y.astype(np.float32))
    ds = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model = SimpleProbMLP(input_dim=input_dim, hidden_dim=128, dropout=0.15)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    model.train()
    epochs = 8
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            p = model(xb)
            loss = loss_fn(p, yb)
            loss.backward()
            opt.step()

    model.eval()

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": int(input_dim),
        "window": int(WINDOW),
        "feature_names": list(FEATURE_NAMES),
        "mean": mean,
        "std": std,
        "meta": {
            "trained_at": datetime.now().isoformat(),
            "epochs": epochs,
            "samples": int(len(Xn)),
            **data_meta,
        },
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(ckpt, model_path)
    return ckpt


def _load_checkpoint(model_path: str) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        return _train_bootstrap_model(model_path)

    try:
        ckpt = torch.load(model_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise TypeError("checkpoint 结构非法")
        if "state_dict" not in ckpt:
            raise KeyError("checkpoint 缺少 state_dict")
        return ckpt
    except Exception:
        # 文件损坏或结构异常时，自动重训覆盖
        return _train_bootstrap_model(model_path)


@lru_cache(maxsize=1)
def _load_model_bundle() -> ModelBundle:
    path = _model_path()
    ckpt = _load_checkpoint(path)

    input_dim = int(ckpt.get("input_dim", len(FEATURE_NAMES) * WINDOW))
    model = SimpleProbMLP(input_dim=input_dim, hidden_dim=128, dropout=0.15)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mean = np.asarray(ckpt.get("mean", np.zeros(input_dim, dtype=np.float32)), dtype=np.float32)
    std = np.asarray(ckpt.get("std", np.ones(input_dim, dtype=np.float32)), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    feature_names = list(ckpt.get("feature_names", FEATURE_NAMES))
    window = int(ckpt.get("window", WINDOW))
    meta = dict(ckpt.get("meta", {}))
    meta["model_path"] = path

    return ModelBundle(
        model=model,
        mean=mean,
        std=std,
        feature_names=feature_names,
        window=window,
        meta=meta,
    )


def _neutral_result(reason: str) -> Dict[str, Any]:
    return {
        "p_up_today": 0.50,
        "p_up_5d": 0.50,
        "p_up_long": 0.50,
        "reasons": [reason],
        "features": {},
    }


def predict_proba_batch(daily_dfs: Sequence[pd.DataFrame]) -> List[Dict[str, Any]]:
    bundle = _load_model_bundle()

    outputs: List[Dict[str, Any]] = [_neutral_result("样本不足，返回中性概率") for _ in range(len(daily_dfs))]

    xs: List[np.ndarray] = []
    valid_idx: List[int] = []

    for i, df in enumerate(daily_dfs):
        x = _extract_latest_feature_vector(df, window=bundle.window)
        if x is None:
            outputs[i] = _neutral_result("样本不足（少于窗口长度），返回中性概率")
            continue
        xs.append(x)
        valid_idx.append(i)

    if not xs:
        return outputs

    X = np.stack(xs).astype(np.float32)
    Xn = ((X - bundle.mean) / bundle.std).astype(np.float32)

    with torch.no_grad():
        p = bundle.model(torch.from_numpy(Xn)).cpu().numpy()

    for i, row in zip(valid_idx, p):
        p1, p5, pl = [float(np.clip(v, 0.05, 0.95)) for v in row.tolist()]
        outputs[i] = {
            "p_up_today": p1,
            "p_up_5d": p5,
            "p_up_long": pl,
            "reasons": [
                "DL 推理结果（PyTorch MLP）",
                f"bootstrap_samples={bundle.meta.get('samples', 'NA')}",
            ],
            "features": {
                "model_type": "simple_prob_mlp",
                "window": int(bundle.window),
                "model_path": bundle.meta.get("model_path", ""),
                "trained_at": bundle.meta.get("trained_at", ""),
            },
        }

    return outputs


def predict_proba(daily_df: pd.DataFrame) -> Dict[str, Any]:
    return predict_proba_batch([daily_df])[0]
