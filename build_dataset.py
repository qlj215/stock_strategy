#!/usr/bin/env python
"""
阶段2:研究数据集构建脚本(build_dataset.py)

目标:
- 把 MiniQMT/xtdata 的日频数据整理为可训练的标准面板数据。

输入:
- 行情来源:stock_strategy.data.fetcher.fetch_stock_data(MiniQMT + xtdata)
- 股票池:--symbols(手工指定)或 --limit(自动拉取上限)
- 时间区间:--start / --end

输出(默认路径):
- data/processed/daily_panel.parquet
- data/processed/split_manifest.json
- data/processed/data_dictionary.md

行业补齐(方案A,独立可删):
- 默认启用 `--industry-scheme scheme_a_local`
- 行业映射文件默认:`data/meta/industry_map_scheme_a.csv`
- 未来方案B打通后,可切换 `--industry-scheme none` 并删除方案A相关文件。

调参速查(优先顺序):
1) 样本规模:
   - 快速验证:--limit 20
   - 正式研究:--limit 100~500(视机器与时长)
2) 时间跨度:
   - 先 3~5 年验证,再扩到 8~10 年
3) 复权口径:
   - 默认 qfq;与下游训练/回测口径保持一致
4) 稳定性:
   - 网络不稳时提高 --retries(如 3~5)
5) 日历补齐:
   - 默认补齐并集交易日历(推荐)
   - 用 --no-union-calendar 可关闭
6) 数据切分:
   - 默认 train/val/test = 0.70/0.15/0.15
   - 调参时保证 train_ratio + val_ratio < 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# 确保能以脚本方式导入 stock_strategy 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_strategy.data.fetcher import fetch_stock_data, list_a_share_symbols
from stock_strategy.data.industry_scheme_a import attach_industry_scheme_a

CORE_SYMBOLS = [
    "000001", "600036", "600519", "000858", "300750",
    "002594", "600276", "603986", "688981", "300760",
]


def _norm_symbol(x: str) -> str:
    s = str(x).strip().upper()
    if not s:
        return ""
    if "." in s:
        s = s.split(".", 1)[0]
    if s.startswith(("SH", "SZ", "BJ")) and s[2:].isdigit():
        s = s[2:]
    if s.isdigit():
        return s.zfill(6)
    return ""


def resolve_symbols(symbols: str, limit: int) -> List[str]:
    if symbols:
        raw = [_norm_symbol(x) for x in symbols.split(",")]
        out = [x for x in raw if x]
    else:
        out = []
        try:
            out = [_norm_symbol(x) for x in list_a_share_symbols(limit=limit)]
            out = [x for x in out if x]
        except Exception:
            out = []

        if not out:
            out = CORE_SYMBOLS.copy()

    # 去重保序
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    if limit > 0:
        uniq = uniq[:limit]
    return uniq


def fetch_symbol_daily(symbol: str, start: str, end: str, adjust: str, retries: int, cache_dir: str) -> pd.DataFrame:
    """拉取单只股票日线并规整为标准字段。"""
    df = fetch_stock_data(
        symbol=symbol,
        start_date=start,
        end_date=end,
        adjust=adjust,
        retries=retries,
        cache_dir=cache_dir,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy().reset_index()
    if "date" not in d.columns:
        d = d.rename(columns={d.columns[0]: "date"})

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in d.columns:
            d[col] = np.nan if col != "volume" else 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # fetch_stock_data 当前主线输出不含 amount,这里采用 close*volume 近似
    d["amount"] = (d["close"] * d["volume"]).astype(float)

    out = pd.DataFrame(
        {
            "date": d["date"],
            "stock_code": str(symbol).zfill(6),
            "open": d["open"],
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            "volume": d["volume"],
            "amount": d["amount"],
            "adj_factor": 1.0,
            "is_trading": 1,
            "industry": pd.NA,
        }
    )

    out = out.dropna(subset=["date", "close"]).sort_values("date")
    return out.reset_index(drop=True)


def build_panel(
    symbols: List[str],
    start: str,
    end: str,
    adjust: str,
    retries: int,
    cache_dir: str,
    use_union_calendar: bool,
) -> pd.DataFrame:
    chunks = []
    for s in symbols:
        try:
            one = fetch_symbol_daily(s, start, end, adjust, retries, cache_dir)
            if not one.empty:
                chunks.append(one)
            else:
                print(f"[WARN] {s} 无可用数据")
        except Exception as e:
            print(f"[WARN] {s} 拉取失败: {e}")

    if not chunks:
        return pd.DataFrame(
            columns=[
                "date", "stock_code", "open", "high", "low", "close",
                "volume", "amount", "adj_factor", "is_trading", "industry",
            ]
        )

    panel = pd.concat(chunks, ignore_index=True)
    panel = panel.drop_duplicates(subset=["date", "stock_code"], keep="last")

    if not use_union_calendar:
        return panel.sort_values(["date", "stock_code"]).reset_index(drop=True)

    # 统一日历:使用样本中所有交易日并对每只股票补齐
    all_dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    completed = []
    for s in symbols:
        one = panel[panel["stock_code"] == s].set_index("date").reindex(all_dates)
        one["stock_code"] = s
        one["is_trading"] = one["close"].notna().astype(int)

        # 停牌/缺失行处理:价格保持 NaN,成交量/成交额置 0
        for c in ["volume", "amount"]:
            one[c] = pd.to_numeric(one[c], errors="coerce")
            one.loc[one["is_trading"] == 0, c] = 0.0

        if "adj_factor" not in one.columns:
            one["adj_factor"] = 1.0
        one["adj_factor"] = pd.to_numeric(one["adj_factor"], errors="coerce").fillna(1.0)

        if "industry" not in one.columns:
            one["industry"] = pd.NA

        one = one.reset_index().rename(columns={"index": "date"})
        completed.append(one)

    panel = pd.concat(completed, ignore_index=True)
    panel = panel.drop_duplicates(subset=["date", "stock_code"], keep="last")
    panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)
    return panel


def apply_industry_enrichment(
    panel: pd.DataFrame,
    industry_scheme: str,
    industry_map_path: str,
    industry_strict: bool,
):
    """
    行业补齐入口(方案A独立注入)。

    - none: 不做行业补齐
    - scheme_a_local: 使用本地映射表注入 industry
    """
    if industry_scheme == "none":
        stats = {
            "industry_scheme": "none",
            "industry_map_path": "",
            "industry_symbol_coverage": 0.0,
            "industry_row_coverage": float(1.0 - panel["industry"].isna().mean()) if ("industry" in panel.columns and len(panel)) else 0.0,
            "industry_missing_symbols": [],
        }
        return panel, stats

    if industry_scheme == "scheme_a_local":
        enriched, s = attach_industry_scheme_a(
            panel=panel,
            map_path=industry_map_path,
            strict=industry_strict,
        )
        stats = {
            "industry_scheme": "scheme_a_local",
            "industry_map_path": s.map_path,
            "industry_symbol_coverage": round(s.mapping_coverage, 6),
            "industry_row_coverage": round(s.row_coverage, 6),
            "industry_missing_symbols": s.missing_symbols,
        }
        return enriched, stats

    raise ValueError(f"不支持的 industry_scheme: {industry_scheme}")


def make_split_manifest(panel: pd.DataFrame, train_ratio: float, val_ratio: float) -> dict:
    dates = sorted(panel["date"].dropna().unique())
    if not dates:
        return {}

    n = len(dates)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, n - n_train - n_val)

    # 修正边界
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_val = max(1, n - n_train - n_test)

    train_end = dates[n_train - 1]
    val_end = dates[min(n - 1, n_train + n_val - 1)]

    train_mask = panel["date"] <= train_end
    val_mask = (panel["date"] > train_end) & (panel["date"] <= val_end)
    test_mask = panel["date"] > val_end

    return {
        "split_by": "date",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 4),
        "train": {
            "start": str(pd.to_datetime(dates[0]).date()),
            "end": str(pd.to_datetime(train_end).date()),
            "rows": int(train_mask.sum()),
            "days": int((panel.loc[train_mask, "date"].nunique())),
        },
        "val": {
            "start": str(pd.to_datetime(train_end + np.timedelta64(1, "D")).date()),
            "end": str(pd.to_datetime(val_end).date()),
            "rows": int(val_mask.sum()),
            "days": int((panel.loc[val_mask, "date"].nunique())),
        },
        "test": {
            "start": str(pd.to_datetime(val_end + np.timedelta64(1, "D")).date()),
            "end": str(pd.to_datetime(dates[-1]).date()),
            "rows": int(test_mask.sum()),
            "days": int((panel.loc[test_mask, "date"].nunique())),
        },
    }


def write_data_dictionary(path: Path):
    content = """# data_dictionary(阶段2)

## 表:daily_panel.parquet

每行代表某只股票在某个交易日的一条观测(股票代码 + 日期 唯一)。

### 字段说明
- `date`:交易日期(datetime)
- `stock_code`:6位股票代码(string)
- `open`:开盘价(float)
- `high`:最高价(float)
- `low`:最低价(float)
- `close`:收盘价(float)
- `volume`:成交量(float)
- `amount`:成交额(float,当前由 close*volume 近似)
- `adj_factor`:复权因子(float,当前固定 1.0)
- `is_trading`:是否可交易(1=有交易,0=停牌/缺失补齐行)
- `industry`：行业名称（string，可空；方案A可由本地映射注入）

### 数据处理规则
1. 统一字段类型,日期转 datetime,价格量转数值。
2. 使用样本并集交易日历对每只股票补齐日期。
3. 补齐行(停牌/缺失)规则:
   - `is_trading=0`
   - `open/high/low/close` 保持缺失
   - `volume/amount` 置 0
4. 去重规则:`date + stock_code` 保留最后一条。

### 时间切分
见 `split_manifest.json`,按日期切分 train/val/test,避免未来信息泄漏。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="阶段2:研究数据集构建",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  python build_dataset.py --start 20200101 --end 20260331 --limit 20\n"
            "  python build_dataset.py --symbols 000001,600519,300750 --start 20180101 --end 20260331\n"
            "\n"
            "调参建议:\n"
            "  1) 先小样本验证:--limit 20\n"
            "  2) 再扩样本规模:--limit 100~500\n"
            "  3) train/val 需满足 train_ratio + val_ratio < 1\n"
            "  4) 方案A行业注入:--industry-scheme scheme_a_local --industry-map-path data/meta/industry_map_scheme_a.csv\n"
        ),
    )
    p.add_argument("--start", default="20200101", help="开始日期 YYYYMMDD")
    p.add_argument("--end", default=datetime.now().strftime("%Y%m%d"), help="结束日期 YYYYMMDD")
    p.add_argument("--symbols", default="", help="逗号分隔股票代码;为空时自动从 MiniQMT 股票池获取")
    p.add_argument("--limit", type=int, default=60, help="自动股票池数量上限(建议:快速=20,研究=100~500)")
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none"], help="复权口径")
    p.add_argument("--retries", type=int, default=2, help="单标的数据拉取重试次数(网络不稳可提高到 3~5)")
    p.add_argument("--cache-dir", default="output/cache", help="拉取阶段缓存目录")
    p.add_argument("--out", default="data/processed/daily_panel.parquet", help="面板数据输出路径")
    p.add_argument("--split-out", default="data/processed/split_manifest.json", help="时间切分清单输出路径")
    p.add_argument("--dict-out", default="data/processed/data_dictionary.md", help="字段字典输出路径")
    p.add_argument("--train-ratio", type=float, default=0.7, help="训练集日期占比")
    p.add_argument("--val-ratio", type=float, default=0.15, help="验证集日期占比")
    p.add_argument("--no-union-calendar", action="store_true", help="不做并集日历补齐(默认开启补齐)")

    # 方案A(本地行业映射): 与主线解耦,后续可整体删除
    p.add_argument(
        "--industry-scheme",
        choices=["scheme_a_local", "none"],
        default="scheme_a_local",
        help="行业补齐方案(默认 scheme_a_local;方案B打通后可切 none)",
    )
    p.add_argument(
        "--industry-map-path",
        default="data/meta/industry_map_scheme_a.csv",
        help="方案A行业映射文件路径",
    )
    p.add_argument(
        "--industry-strict",
        action="store_true",
        help="方案A严格模式:存在未映射股票时直接报错",
    )

    args = p.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("[ERROR] --train-ratio 必须在 (0,1) 区间")
    if not (0.0 <= args.val_ratio < 1.0):
        raise SystemExit("[ERROR] --val-ratio 必须在 [0,1) 区间")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("[ERROR] --train-ratio + --val-ratio 必须 < 1")

    symbols = resolve_symbols(args.symbols, args.limit)
    if not symbols:
        raise SystemExit("[ERROR] 无可用股票池")

    print(f"[INFO] symbols={len(symbols)} start={args.start} end={args.end} adjust={args.adjust}")

    panel = build_panel(
        symbols=symbols,
        start=args.start,
        end=args.end,
        adjust=args.adjust,
        retries=args.retries,
        cache_dir=args.cache_dir,
        use_union_calendar=not args.no_union_calendar,
    )

    if panel.empty:
        raise SystemExit("[ERROR] 面板数据为空")

    # 保证唯一键
    dup = panel.duplicated(subset=["date", "stock_code"]).sum()
    if dup > 0:
        panel = panel.drop_duplicates(subset=["date", "stock_code"], keep="last")

    # 方案A:行业补齐(独立注入层,便于未来删除)
    panel, industry_stats = apply_industry_enrichment(
        panel=panel,
        industry_scheme=args.industry_scheme,
        industry_map_path=args.industry_map_path,
        industry_strict=args.industry_strict,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    split = make_split_manifest(panel, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    split["generated_at"] = datetime.now().isoformat()
    split["rows"] = int(len(panel))
    split["stocks"] = int(panel["stock_code"].nunique())
    split["days"] = int(panel["date"].nunique())
    split["symbols"] = symbols
    split["start"] = args.start
    split["end"] = args.end
    split["adjust"] = args.adjust
    split["union_calendar"] = not args.no_union_calendar
    split["industry"] = industry_stats

    split_path = Path(args.split_out)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    write_data_dictionary(Path(args.dict_out))

    print("[DONE] stage2 dataset built")
    print(f"  panel : {out_path}")
    print(f"  split : {split_path}")
    print(f"  dict  : {Path(args.dict_out)}")
    print(f"  rows  : {len(panel)}")
    print(f"  stocks: {panel['stock_code'].nunique()}")
    print(f"  days  : {panel['date'].nunique()}")
    print(f"  industry_scheme        : {industry_stats.get('industry_scheme')}")
    print(f"  industry_symbol_cover  : {industry_stats.get('industry_symbol_coverage')}")
    print(f"  industry_row_cover     : {industry_stats.get('industry_row_coverage')}")
    if industry_stats.get("industry_missing_symbols"):
        miss = industry_stats.get("industry_missing_symbols", [])
        print(f"  industry_missing_symbols(sample): {miss[:10]}")


if __name__ == "__main__":
    main()
