"""
概率预测后端：支持 rule / dl / auto 三种模式切换。

说明：
- rule：使用项目内置规则模型（默认）
- dl：调用用户配置的深度学习推理入口
- auto：优先 dl，失败自动回退 rule

环境变量：
- STOCK_PROB_BACKEND=rule|dl|auto
- STOCK_DL_ENTRYPOINT=python.module:function
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd


def _config_candidates() -> Tuple[str, ...]:
    base = os.path.dirname(os.path.abspath(__file__))
    return (
        os.path.join(base, "config", "model_backend.json"),
        os.path.join(base, "model_backend.json"),
    )


@lru_cache(maxsize=1)
def _load_backend_config() -> Dict[str, Any]:
    for path in _config_candidates():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data["_path"] = path
                    return data
        except Exception:
            continue
    return {"default_backend": "rule", "dl_entrypoint": "", "_path": ""}


def _clamp_prob(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.5
    return float(max(0.05, min(0.95, v)))


def _sigmoid(x: float) -> float:
    return 1 / (1 + pow(2.718281828, -5 * float(x)))


class RuleProbabilityBackend:
    name = "rule"

    def predict(self, daily_df: pd.DataFrame) -> Dict[str, Any]:
        if daily_df is None or len(daily_df) < 30:
            return {
                "p_up_today": 0.50,
                "p_up_5d": 0.50,
                "p_up_long": 0.50,
                "reasons": ["历史样本较少，采用中性先验概率。"],
                "features": {},
            }

        d = daily_df.copy()
        if "volume" not in d.columns:
            d["volume"] = 0.0

        d["ret1"] = d["close"].pct_change()
        d["ma5"] = d["close"].rolling(5).mean()
        d["ma20"] = d["close"].rolling(20).mean()
        d["vol5"] = d["volume"].rolling(5).mean()
        d["vol20"] = d["volume"].rolling(20).mean()

        last = d.iloc[-1]
        momentum = (last["close"] / d["close"].iloc[-6]) - 1 if len(d) >= 6 else 0
        ma_bias = (
            (last["ma5"] - last["ma20"]) / last["ma20"]
            if pd.notna(last["ma5"]) and pd.notna(last["ma20"]) and last["ma20"]
            else 0
        )
        vol_ratio = (
            (last["vol5"] / last["vol20"])
            if pd.notna(last["vol5"]) and pd.notna(last["vol20"]) and last["vol20"]
            else 1
        )
        recent_win = (d["ret1"].tail(10) > 0).mean()

        score_today = 0.35 * momentum + 0.45 * ma_bias + 0.15 * (vol_ratio - 1) + 0.25 * (recent_win - 0.5)
        score_5d = 0.45 * momentum + 0.55 * ma_bias + 0.20 * (vol_ratio - 1) + 0.35 * (recent_win - 0.5)
        score_long = 0.25 * momentum + 0.70 * ma_bias + 0.10 * (vol_ratio - 1) + 0.20 * (recent_win - 0.5)

        p_today = _clamp_prob(_sigmoid(score_today))
        p_5d = _clamp_prob(_sigmoid(score_5d))
        p_long = _clamp_prob(_sigmoid(score_long))

        return {
            "p_up_today": p_today,
            "p_up_5d": p_5d,
            "p_up_long": p_long,
            "reasons": [
                f"短期动量（近5日）为 {momentum * 100:.2f}%",
                f"均线结构（MA5-MA20）偏离 {ma_bias * 100:.2f}%",
                f"量能比（VOL5/VOL20）为 {vol_ratio:.2f}",
                f"近10日上涨胜率 {recent_win * 100:.1f}%",
            ],
            "features": {
                "momentum_5d": round(momentum * 100, 2),
                "ma_bias_pct": round(ma_bias * 100, 2),
                "vol_ratio": round(vol_ratio, 2),
                "win_rate_10d": round(recent_win * 100, 1),
            },
        }


@dataclass
class DLEntry:
    fn: Optional[Callable[[pd.DataFrame], Dict[str, Any]]] = None
    entrypoint: str = ""
    error: str = ""


def _load_dl_entry(entrypoint: str) -> DLEntry:
    ep = (entrypoint or "").strip()
    if not ep:
        return DLEntry(fn=None, entrypoint="", error="未配置 STOCK_DL_ENTRYPOINT")

    if ":" not in ep:
        return DLEntry(fn=None, entrypoint=ep, error="入口格式应为 module:function")

    mod_name, fn_name = ep.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        if not callable(fn):
            return DLEntry(fn=None, entrypoint=ep, error=f"{ep} 不是可调用函数")
        return DLEntry(fn=fn, entrypoint=ep, error="")
    except Exception as e:
        return DLEntry(fn=None, entrypoint=ep, error=str(e))


def _normalize_dl_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("DL 输出必须是 dict")

    return {
        "p_up_today": _clamp_prob(raw.get("p_up_today", 0.5)),
        "p_up_5d": _clamp_prob(raw.get("p_up_5d", 0.5)),
        "p_up_long": _clamp_prob(raw.get("p_up_long", 0.5)),
        "reasons": list(raw.get("reasons", ["DL 模型未提供解释信息"]))[:8],
        "features": dict(raw.get("features", {})),
    }


class DLProbabilityBackend:
    name = "dl"

    def __init__(self, entrypoint: Optional[str] = None):
        cfg = _load_backend_config()
        self.entrypoint = (
            entrypoint
            or os.getenv("STOCK_DL_ENTRYPOINT", "")
            or str(cfg.get("dl_entrypoint", ""))
        ).strip()
        self._entry = _load_dl_entry(self.entrypoint)

    @property
    def available(self) -> bool:
        return self._entry.fn is not None

    @property
    def error(self) -> str:
        return self._entry.error

    def predict(self, daily_df: pd.DataFrame) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError(self.error or "DL 入口不可用")
        raw = self._entry.fn(daily_df.copy())
        return _normalize_dl_output(raw)


def _resolve_requested_backend(requested: Optional[str]) -> str:
    cfg = _load_backend_config()
    mode = (
        requested
        or os.getenv("STOCK_PROB_BACKEND", "")
        or str(cfg.get("default_backend", "rule"))
    ).strip().lower()
    if mode not in {"rule", "dl", "auto"}:
        return "rule"
    return mode


def _choose_backend(mode: str) -> Tuple[str, Any, DLProbabilityBackend]:
    rule = RuleProbabilityBackend()
    dl = DLProbabilityBackend()

    if mode == "rule":
        return mode, rule, dl
    if mode == "dl":
        return mode, dl, dl
    # auto
    if dl.available:
        return mode, dl, dl
    return mode, rule, dl


def predict_probability(
    daily_df: pd.DataFrame,
    backend: Optional[str] = None,
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    mode = _resolve_requested_backend(backend)
    requested_mode, chosen_backend, dl_backend = _choose_backend(mode)

    try:
        out = chosen_backend.predict(daily_df)
        out.update(
            {
                "backend": chosen_backend.name,
                "backend_requested": requested_mode,
                "backend_fallback": False,
                "backend_error": "",
            }
        )
        return out
    except Exception as e:
        if allow_fallback and chosen_backend.name != "rule":
            rule = RuleProbabilityBackend()
            out = rule.predict(daily_df)
            out["reasons"] = [f"DL 后端不可用，已回退 rule：{e}"] + list(out.get("reasons", []))
            out.update(
                {
                    "backend": "rule",
                    "backend_requested": requested_mode,
                    "backend_fallback": True,
                    "backend_error": str(e),
                }
            )
            return out
        raise


def get_backend_runtime_status(requested: Optional[str] = None) -> Dict[str, Any]:
    mode = _resolve_requested_backend(requested)
    _, chosen_backend, dl_backend = _choose_backend(mode)

    selected_pre_fallback = chosen_backend.name
    selected_with_fallback = selected_pre_fallback
    if mode in {"dl", "auto"} and not dl_backend.available:
        selected_with_fallback = "rule"

    cfg = _load_backend_config()
    return {
        "requested": mode,
        "env_default": os.getenv("STOCK_PROB_BACKEND", ""),
        "config_default": str(cfg.get("default_backend", "rule")),
        "config_path": str(cfg.get("_path", "")),
        "selected_pre_fallback": selected_pre_fallback,
        "selected_with_fallback": selected_with_fallback,
        "dl_entrypoint": dl_backend.entrypoint or "",
        "dl_entrypoint_env": os.getenv("STOCK_DL_ENTRYPOINT", ""),
        "dl_entrypoint_config": str(cfg.get("dl_entrypoint", "")),
        "dl_available": bool(dl_backend.available),
        "dl_error": dl_backend.error,
    }
