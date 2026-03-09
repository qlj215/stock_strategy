"""
研究型批量回测脚本
目标：多行业 + 多策略 + 长周期，输出结构化研究结果
"""
import os
import sys
import json
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_strategy.data.fetcher import fetch_stock_data
from stock_strategy.strategies.signals import STRATEGY_MAP
from stock_strategy.backtest.engine import run_backtest
from stock_strategy.backtest.metrics import calc_metrics

SECTOR_SYMBOLS = {
    "银行": ["000001", "600036"],
    "白酒": ["600519", "000858"],
    "新能源": ["002594", "300750"],
    "半导体": ["603986", "688981"],
    "医药": ["600276", "300760"],
}


def ensure_dirs(base: str):
    for d in ["metrics", "trades", "logs"]:
        os.makedirs(os.path.join(base, d), exist_ok=True)


def run_research(start_date="20180101", end_date="20241231", init_cash=100000.0, commission=0.0003):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join("output", f"research_{ts}")
    ensure_dirs(out_base)

    config = {
        "start_date": start_date,
        "end_date": end_date,
        "init_cash": init_cash,
        "commission": commission,
        "sectors": SECTOR_SYMBOLS,
        "strategies": list(STRATEGY_MAP.keys()),
    }
    with open(os.path.join(out_base, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    rows = []
    logs = []

    for sector, symbols in SECTOR_SYMBOLS.items():
        for symbol in symbols:
            try:
                df = fetch_stock_data(symbol, start_date, end_date)
            except Exception as e:
                logs.append(f"[ERROR] 拉取数据失败 {sector}-{symbol}: {e}")
                continue

            for strat_name, strat_fn in STRATEGY_MAP.items():
                try:
                    df_sig = strat_fn(df.copy())
                    result = run_backtest(df_sig, init_cash=init_cash, commission=commission)
                    metrics = calc_metrics(result["portfolio"])

                    row = {
                        "行业": sector,
                        "股票": symbol,
                        "策略": strat_name,
                        **metrics,
                        "交易次数": int((result["trades"].shape[0] // 2) if not result["trades"].empty else 0),
                    }
                    rows.append(row)

                    if not result["trades"].empty:
                        trade_path = os.path.join(out_base, "trades", f"{sector}_{symbol}_{strat_name}_trades.csv")
                        result["trades"].to_csv(trade_path, encoding="utf-8-sig")

                except Exception as e:
                    logs.append(f"[ERROR] 回测失败 {sector}-{symbol}-{strat_name}: {e}")

    results = pd.DataFrame(rows)
    if results.empty:
        with open(os.path.join(out_base, "logs", "run.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(logs) if logs else "No result generated")
        return out_base

    results = results.sort_values(["夏普比率", "总收益率"], ascending=False)
    results.to_csv(os.path.join(out_base, "metrics", "all_results.csv"), index=False, encoding="utf-8-sig")

    strategy_summary = (
        results.groupby("策略")[["总收益率", "年化收益率", "夏普比率", "最大回撤", "胜率(%)", "交易次数"]]
        .mean()
        .sort_values("夏普比率", ascending=False)
        .round(3)
    )
    strategy_summary.to_csv(os.path.join(out_base, "metrics", "strategy_summary.csv"), encoding="utf-8-sig")

    sector_summary = (
        results.groupby("行业")[["总收益率", "年化收益率", "夏普比率", "最大回撤", "胜率(%)", "交易次数"]]
        .mean()
        .sort_values("夏普比率", ascending=False)
        .round(3)
    )
    sector_summary.to_csv(os.path.join(out_base, "metrics", "sector_summary.csv"), encoding="utf-8-sig")

    top20 = results.head(20)
    top20.to_csv(os.path.join(out_base, "metrics", "top20.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(out_base, "logs", "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs) if logs else "run ok")

    # 研究报告
    best = top20.iloc[0]
    report = []
    report.append("# 多行业多策略长周期研究报告\n")
    report.append(f"- 时间区间：{start_date} ~ {end_date}")
    report.append(f"- 行业数：{len(SECTOR_SYMBOLS)}，股票数：{sum(len(v) for v in SECTOR_SYMBOLS.values())}")
    report.append(f"- 策略数：{len(STRATEGY_MAP)}")
    report.append(f"- 样本总组合数：{len(results)}\n")

    report.append("## 关键结论")
    report.append(f"- 综合最佳组合：{best['行业']} / {best['股票']} / {best['策略']}（夏普 {best['夏普比率']}，总收益 {best['总收益率']}%）")
    report.append("- 详细明细见 metrics/all_results.csv 与 metrics/top20.csv")

    report.append("\n## 按策略平均表现（见 strategy_summary.csv）")
    report.append(strategy_summary.to_string())

    report.append("\n## 按行业平均表现（见 sector_summary.csv）")
    report.append(sector_summary.to_string())

    with open(os.path.join(out_base, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return out_base


if __name__ == "__main__":
    out = run_research()
    print(f"[DONE] 研究输出目录: {out}")
