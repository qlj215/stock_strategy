"""
方案A（独立可删）：本地行业映射注入模块。

设计目标：
- 在 MiniQMT 动态行业板块（方案B）不可用时，给阶段2面板数据补充行业字段。
- 与主线/交易逻辑解耦，仅用于阶段2~7研究分支。
- 未来方案B打通后可整体删除本模块与对应映射文件。

使用方式：
- build_dataset.py 中设置 --industry-scheme scheme_a_local
- 指定 --industry-map-path（默认 data/meta/industry_map_scheme_a.csv）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


REQUIRED_COLS = ["stock_code", "industry"]


@dataclass
class IndustryAttachStats:
    total_symbols: int
    mapped_symbols: int
    mapping_coverage: float
    row_coverage: float
    missing_symbols: List[str]
    map_path: str


def _norm_code(x: str) -> str:
    s = str(x).strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]
    if s.startswith(("SH", "SZ", "BJ")) and s[2:].isdigit():
        s = s[2:]
    if s.isdigit():
        return s.zfill(6)
    return ""


def load_industry_map(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"行业映射文件不存在: {p}")

    df = pd.read_csv(p, dtype=str)
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"行业映射缺少必要字段: {miss}")

    out = df.copy()
    out["stock_code"] = out["stock_code"].map(_norm_code)
    out["industry"] = out["industry"].astype(str).str.strip()
    out = out[(out["stock_code"] != "") & (out["industry"] != "")]
    out = out.drop_duplicates(subset=["stock_code"], keep="last")

    keep_cols = [c for c in ["stock_code", "industry", "source", "note", "updated_at"] if c in out.columns]
    return out[keep_cols].reset_index(drop=True)


def attach_industry_scheme_a(
    panel: pd.DataFrame,
    map_path: str | Path,
    strict: bool = False,
) -> Tuple[pd.DataFrame, IndustryAttachStats]:
    """
    对 daily_panel 注入 industry 字段。

    strict=True 时，如果有股票未映射则报错。
    """
    if panel is None or panel.empty:
        empty_stats = IndustryAttachStats(
            total_symbols=0,
            mapped_symbols=0,
            mapping_coverage=0.0,
            row_coverage=0.0,
            missing_symbols=[],
            map_path=str(map_path),
        )
        return panel, empty_stats

    p = panel.copy()
    p["stock_code"] = p["stock_code"].astype(str).map(_norm_code)

    m = load_industry_map(map_path)
    p = p.drop(columns=["industry"], errors="ignore")
    p = p.merge(m[["stock_code", "industry"]], on="stock_code", how="left")

    symbols = sorted(p["stock_code"].dropna().unique().tolist())
    mapped_set = set(m["stock_code"].tolist())
    missing_symbols = [s for s in symbols if s not in mapped_set]

    total = len(symbols)
    mapped = total - len(missing_symbols)
    mapping_coverage = float(mapped / total) if total else 0.0
    row_coverage = float(1.0 - p["industry"].isna().mean()) if len(p) else 0.0

    if strict and missing_symbols:
        raise RuntimeError(
            f"行业映射不完整：缺少 {len(missing_symbols)} 只股票，示例={missing_symbols[:10]}"
        )

    stats = IndustryAttachStats(
        total_symbols=total,
        mapped_symbols=mapped,
        mapping_coverage=mapping_coverage,
        row_coverage=row_coverage,
        missing_symbols=missing_symbols,
        map_path=str(map_path),
    )
    return p, stats
