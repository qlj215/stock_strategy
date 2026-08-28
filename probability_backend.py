"""
概率预测后端：支持 rule / dl / auto 三种模式切换。

说明：
- rule：使用项目内置规则模型（默认）
- dl：调用用户配置的深度学习推理入口
- auto：优先 dl，失败自动回退 rule

配置优先级：
1) 请求参数 backend
2) 环境变量 STOCK_PROB_BACKEND / STOCK_DL_ENTRYPOINT
3) 配置文件 config/model_backend.json
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_PROB_MODEL = "rule_basic"
PROBABILITY_MODEL_LABELS = {
    "rule_basic": "基础规则模型",
    "rule_no_chase": "低追高倾向模型",
    "dl_default": "DL 默认模型",
}


def get_probability_model_options() -> List[Dict[str, str]]:
    return [
        {"value": key, "label": PROBABILITY_MODEL_LABELS[key]}
        for key in ("rule_basic", "rule_no_chase")
    ]


def get_probability_model_label(model_name: Optional[str]) -> str:
    key = str(model_name or "").strip().lower()
    return PROBABILITY_MODEL_LABELS.get(key, PROBABILITY_MODEL_LABELS[DEFAULT_PROB_MODEL])


def _resolve_requested_prob_model(requested: Optional[str]) -> str:
    key = str(requested or DEFAULT_PROB_MODEL).strip().lower()
    if key not in {"rule_basic", "rule_no_chase"}:
        return DEFAULT_PROB_MODEL
    return key


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

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = _resolve_requested_prob_model(model_name)

    def _neutral_result(self, reason: str) -> Dict[str, Any]:
        return {
            "p_up_today": 0.50,
            "p_up_5d": 0.50,
            "p_up_long": 0.50,
            "reasons": [reason],
            "features": {
                "prob_model": self.model_name,
                "prob_model_label": get_probability_model_label(self.model_name),
            },
        }

    def predict(self, daily_df: pd.DataFrame) -> Dict[str, Any]:
        if daily_df is None or len(daily_df) < 30:
            return self._neutral_result("历史样本较少，采用中性先验概率。")

        d = daily_df.copy()
        if "volume" not in d.columns:
            d["volume"] = 0.0

        d["ret1"] = d["close"].pct_change()
        d["ma5"] = d["close"].rolling(5).mean()
        d["ma20"] = d["close"].rolling(20).mean()
        d["vol5"] = d["volume"].rolling(5).mean()
        d["vol20"] = d["volume"].rolling(20).mean()

        last = d.iloc[-1]
        close = float(last["close"])
        momentum = (close / d["close"].iloc[-6]) - 1 if len(d) >= 6 and d["close"].iloc[-6] else 0.0
        ma_bias = (
            (float(last["ma5"]) - float(last["ma20"])) / float(last["ma20"])
            if pd.notna(last["ma5"]) and pd.notna(last["ma20"]) and float(last["ma20"]) != 0
            else 0.0
        )
        vol_ratio = (
            float(last["vol5"]) / float(last["vol20"])
            if pd.notna(last["vol5"]) and pd.notna(last["vol20"]) and float(last["vol20"]) != 0
            else 1.0
        )
        recent_win = float((d["ret1"].tail(10) > 0).mean())
        max20 = float(d["close"].tail(20).max()) if not d["close"].tail(20).empty else close
        max60 = float(d["close"].tail(60).max()) if not d["close"].tail(60).empty else close
        dist_to_high_20 = close / max20 if max20 > 0 else 1.0
        dist_to_high_60 = close / max60 if max60 > 0 else 1.0
        ret_10d = (close / d["close"].iloc[-11]) - 1 if len(d) >= 11 and d["close"].iloc[-11] else 0.0
        ma20_gap = (
            (close / float(last["ma20"])) - 1
            if pd.notna(last["ma20"]) and float(last["ma20"]) != 0
            else 0.0
        )

        score_today = 0.35 * momentum + 0.45 * ma_bias + 0.15 * (vol_ratio - 1) + 0.25 * (recent_win - 0.5)
        score_5d = 0.45 * momentum + 0.55 * ma_bias + 0.20 * (vol_ratio - 1) + 0.35 * (recent_win - 0.5)
        score_long = 0.25 * momentum + 0.70 * ma_bias + 0.10 * (vol_ratio - 1) + 0.20 * (recent_win - 0.5)

        chase_penalty = 0.0
        penalty_reason = "基础规则评分。"
        if self.model_name == "rule_no_chase":
            high20_penalty = max(0.0, (dist_to_high_20 - 0.97) / 0.03)
            high60_penalty = max(0.0, (dist_to_high_60 - 0.92) / 0.08)
            ret10_penalty = max(0.0, (ret_10d - 0.10) / 0.10)
            ma20_gap_penalty = max(0.0, (ma20_gap - 0.08) / 0.08)
            chase_penalty = (
                0.12 * high20_penalty
                + 0.08 * high60_penalty
                + 0.10 * ret10_penalty
                + 0.10 * ma20_gap_penalty
            )
            score_today -= chase_penalty
            score_5d -= chase_penalty * 0.85
            score_long -= chase_penalty * 0.50
            if chase_penalty > 0:
                penalty_reason = f"触发高位抑制，综合惩罚系数 {chase_penalty:.3f}。"
            else:
                penalty_reason = "未触发明显追高抑制，仍按偏谨慎规则评分。"

        p_today = _clamp_prob(_sigmoid(score_today))
        p_5d = _clamp_prob(_sigmoid(score_5d))
        p_long = _clamp_prob(_sigmoid(score_long))

        return {
            "p_up_today": p_today,
            "p_up_5d": p_5d,
            "p_up_long": p_long,
            "reasons": [
                f"模型：{get_probability_model_label(self.model_name)}",
                penalty_reason,
                f"短期动量（近5日）为 {momentum * 100:.2f}%",
                f"均线结构（MA5-MA20）偏离 {ma_bias * 100:.2f}%",
                f"量能比（VOL5/VOL20）为 {vol_ratio:.2f}",
                f"近10日上涨胜率 {recent_win * 100:.1f}%",
            ],
            "features": {
                "prob_model": self.model_name,
                "prob_model_label": get_probability_model_label(self.model_name),
                "momentum_5d": round(momentum * 100, 2),
                "ma_bias_pct": round(ma_bias * 100, 2),
                "vol_ratio": round(vol_ratio, 2),
                "win_rate_10d": round(recent_win * 100, 1),
                "dist_to_20d_high": round(dist_to_high_20, 4),
                "dist_to_60d_high": round(dist_to_high_60, 4),
                "ret_10d_pct": round(ret_10d * 100, 2),
                "ma20_gap_pct": round(ma20_gap * 100, 2),
                "chase_penalty": round(chase_penalty, 4),
            },
        }

    def predict_batch(self, daily_dfs: Sequence[pd.DataFrame]) -> List[Dict[str, Any]]:
        return [self.predict(df) for df in daily_dfs]


@dataclass
class DLEntry:
    fn: Optional[Callable[[pd.DataFrame], Dict[str, Any]]] = None
    batch_fn: Optional[Callable[[List[pd.DataFrame]], List[Dict[str, Any]]]] = None
    entrypoint: str = ""
    error: str = ""


def _load_dl_entry(entrypoint: str) -> DLEntry:
    ep = (entrypoint or "").strip()
    if not ep:
        return DLEntry(fn=None, batch_fn=None, entrypoint="", error="未配置 STOCK_DL_ENTRYPOINT")

    if ":" not in ep:
        return DLEntry(fn=None, batch_fn=None, entrypoint=ep, error="入口格式应为 module:function")

    mod_name, fn_name = ep.split(":", 1)
    try:
        importlib.invalidate_caches()
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        if not callable(fn):
            return DLEntry(fn=None, batch_fn=None, entrypoint=ep, error=f"{ep} 不是可调用函数")

        batch_fn = getattr(mod, "predict_proba_batch", None)
        if batch_fn is not None and not callable(batch_fn):
            batch_fn = None

        return DLEntry(fn=fn, batch_fn=batch_fn, entrypoint=ep, error="")
    except Exception as e:
        return DLEntry(fn=None, batch_fn=None, entrypoint=ep, error=str(e))


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

    def predict_batch(self, daily_dfs: Sequence[pd.DataFrame]) -> List[Dict[str, Any]]:
        if not self.available:
            raise RuntimeError(self.error or "DL 入口不可用")

        if self._entry.batch_fn is not None:
            raw_list = self._entry.batch_fn([df.copy() for df in daily_dfs])
            if not isinstance(raw_list, list):
                raise TypeError("DL batch 输出必须是 list")
            if len(raw_list) != len(daily_dfs):
                raise ValueError(f"DL batch 输出长度不匹配: expect={len(daily_dfs)} got={len(raw_list)}")
            return [_normalize_dl_output(item) for item in raw_list]

        return [self.predict(df) for df in daily_dfs]


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


def _choose_backend(mode: str, prob_model: Optional[str] = None) -> Tuple[str, Any, DLProbabilityBackend, str]:
    resolved_prob_model = _resolve_requested_prob_model(prob_model)
    rule = RuleProbabilityBackend(model_name=resolved_prob_model)
    dl = DLProbabilityBackend()

    if mode == "rule":
        return mode, rule, dl, resolved_prob_model
    if mode == "dl":
        return mode, dl, dl, resolved_prob_model
    if dl.available:
        return mode, dl, dl, resolved_prob_model
    return mode, rule, dl, resolved_prob_model


def _attach_meta(
    out: Dict[str, Any],
    backend: str,
    requested: str,
    fallback: bool,
    prob_model: str,
    prob_model_requested: str,
    error: str = "",
) -> Dict[str, Any]:
    out = dict(out)
    out.update(
        {
            "backend": backend,
            "backend_requested": requested,
            "backend_fallback": bool(fallback),
            "backend_error": error or "",
            "prob_model": prob_model,
            "prob_model_label": get_probability_model_label(prob_model),
            "prob_model_requested": prob_model_requested,
            "prob_model_requested_label": get_probability_model_label(prob_model_requested),
        }
    )
    return out


def predict_probability(
    daily_df: pd.DataFrame,
    backend: Optional[str] = None,
    prob_model: Optional[str] = None,
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    mode = _resolve_requested_backend(backend)
    requested_mode, chosen_backend, _dl_backend, requested_prob_model = _choose_backend(mode, prob_model)
    selected_prob_model = getattr(chosen_backend, "model_name", "dl_default") if chosen_backend.name == "rule" else "dl_default"

    try:
        out = chosen_backend.predict(daily_df)
        return _attach_meta(
            out,
            backend=chosen_backend.name,
            requested=requested_mode,
            fallback=False,
            prob_model=selected_prob_model,
            prob_model_requested=requested_prob_model,
        )
    except Exception as e:
        if allow_fallback and chosen_backend.name != "rule":
            rule = RuleProbabilityBackend(model_name=requested_prob_model)
            out = rule.predict(daily_df)
            out["reasons"] = [f"DL 后端不可用，已回退 rule：{e}"] + list(out.get("reasons", []))
            return _attach_meta(
                out,
                backend="rule",
                requested=requested_mode,
                fallback=True,
                prob_model=rule.model_name,
                prob_model_requested=requested_prob_model,
                error=str(e),
            )
        raise


def predict_probability_batch(
    daily_dfs: Sequence[pd.DataFrame],
    backend: Optional[str] = None,
    prob_model: Optional[str] = None,
    allow_fallback: bool = True,
) -> List[Dict[str, Any]]:
    mode = _resolve_requested_backend(backend)
    requested_mode, chosen_backend, _dl_backend, requested_prob_model = _choose_backend(mode, prob_model)

    daily_dfs = list(daily_dfs)
    if not daily_dfs:
        return []

    selected_prob_model = getattr(chosen_backend, "model_name", "dl_default") if chosen_backend.name == "rule" else "dl_default"

    try:
        raw_list = chosen_backend.predict_batch(daily_dfs)
        return [
            _attach_meta(
                item,
                backend=chosen_backend.name,
                requested=requested_mode,
                fallback=False,
                prob_model=selected_prob_model,
                prob_model_requested=requested_prob_model,
            )
            for item in raw_list
        ]
    except Exception as e:
        if allow_fallback and chosen_backend.name != "rule":
            rule = RuleProbabilityBackend(model_name=requested_prob_model)
            raw_list = rule.predict_batch(daily_dfs)
            out = []
            for item in raw_list:
                item = dict(item)
                item["reasons"] = [f"DL 后端不可用，已回退 rule：{e}"] + list(item.get("reasons", []))
                out.append(
                    _attach_meta(
                        item,
                        backend="rule",
                        requested=requested_mode,
                        fallback=True,
                        prob_model=rule.model_name,
                        prob_model_requested=requested_prob_model,
                        error=str(e),
                    )
                )
            return out
        raise


def get_backend_runtime_status(requested: Optional[str] = None) -> Dict[str, Any]:
    importlib.invalidate_caches()
    mode = _resolve_requested_backend(requested)
    _, chosen_backend, dl_backend, _ = _choose_backend(mode)

    selected_pre_fallback = chosen_backend.name
    selected_with_fallback = selected_pre_fallback
    if mode in {"dl", "auto"} and not dl_backend.available:
        selected_with_fallback = "rule"

    cfg = _load_backend_config()
    torch_version = ""
    torch_error = ""
    try:
        import torch  # type: ignore

        torch_version = str(getattr(torch, "__version__", "unknown"))
    except Exception as e:
        torch_version = ""
        torch_error = f"{type(e).__name__}: {e}"

    return {
        "requested": mode,
        "process_id": os.getpid(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "working_directory": os.getcwd(),
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
        "torch_version": torch_version,
        "torch_error": torch_error,
    }
