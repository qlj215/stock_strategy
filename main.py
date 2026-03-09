"""
股票交易策略回测系统 - 主程序入口

用法示例:
    python main.py                          # 默认：000001 MA策略
    python main.py --symbol 600519 --strategy macd --start 20200101 --end 20241231
    python main.py --symbol 000001 --strategy all  # 运行所有策略对比
    python main.py --symbol 000001 --strategy ma --optimize  # 参数优化
"""

import argparse
import sys
import os

# 确保包路径正确（将父目录加入 sys.path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_strategy.data.fetcher import fetch_stock_data, save_data
from stock_strategy.strategies.signals import get_strategy, STRATEGY_MAP
from stock_strategy.backtest.engine import run_backtest
from stock_strategy.backtest.metrics import calc_metrics, print_metrics
from stock_strategy.backtest.optimizer import grid_search, print_top_params
from stock_strategy.visualization.plotter import plot_strategy, plot_optimization

# ── 各策略默认参数网格（用于优化） ──
OPTIMIZE_GRIDS = {
    "ma": {
        "fast": [3, 5, 10],
        "slow": [15, 20, 30, 60],
    },
    "rsi": {
        "period":    [7, 14, 21],
        "oversold":  [25, 30, 35],
        "overbought":[65, 70, 75],
    },
    "macd": {
        "fast":          [8, 12, 16],
        "slow":          [20, 26, 32],
        "signal_period": [7, 9, 12],
    },
    "bollinger": {
        "period":  [10, 15, 20, 25],
        "std_dev": [1.5, 2.0, 2.5],
    },
}


def run_single_strategy(
    symbol: str,
    strategy_name: str,
    start_date: str,
    end_date: str,
    init_cash: float,
    commission: float,
    output_dir: str,
    show_plot: bool,
    do_optimize: bool,
    opt_metric: str,
):
    print(f"\n[INFO] 获取 {symbol} 数据 ({start_date} ~ {end_date}) ...")
    df = fetch_stock_data(symbol, start_date, end_date)
    print(f"[INFO] 共 {len(df)} 条记录")

    strategy_fn = get_strategy(strategy_name)
    print(f"[INFO] 运行策略: {strategy_name}")
    df_sig = strategy_fn(df.copy())

    result = run_backtest(df_sig, init_cash=init_cash, commission=commission)
    metrics = calc_metrics(result["portfolio"])

    strategy_display = df_sig["strategy_name"].iloc[0] if "strategy_name" in df_sig.columns else strategy_name
    print_metrics(metrics, strategy_display)

    # 保存交易记录
    if not result["trades"].empty:
        trade_path = os.path.join(output_dir, f"{symbol}_{strategy_name}_trades.csv")
        result["trades"].to_csv(trade_path, encoding="utf-8-sig")
        print(f"[INFO] 交易记录已保存: {trade_path}")

    # 绘图
    save_path = plot_strategy(
        df_sig, result["portfolio"],
        symbol=symbol, strategy_name=strategy_display,
        save_dir=output_dir, show=show_plot
    )
    print(f"[INFO] 图表已保存: {save_path}")

    # 参数优化
    if do_optimize and strategy_name in OPTIMIZE_GRIDS:
        print(f"\n[INFO] 开始参数优化 ({strategy_name})，目标: {opt_metric}")
        results_df = grid_search(
            df.copy(), strategy_fn,
            OPTIMIZE_GRIDS[strategy_name],
            opt_metric=opt_metric,
            init_cash=init_cash, commission=commission,
            verbose=True
        )
        print_top_params(results_df, n=5, metric=opt_metric)

        # 保存优化结果
        opt_path = os.path.join(output_dir, f"{symbol}_{strategy_name}_optimization.csv")
        results_df.to_csv(opt_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 优化结果已保存: {opt_path}")

        # 优化图（取前两个参数作二维热力图）
        param_keys = list(OPTIMIZE_GRIDS[strategy_name].keys())
        if len(param_keys) >= 2:
            opt_fig = plot_optimization(
                results_df, x_col=param_keys[0], y_col=param_keys[1],
                metric=opt_metric, save_dir=output_dir,
                strategy_name=f"{symbol}_{strategy_name}", show=show_plot
            )
            print(f"[INFO] 优化热力图已保存: {opt_fig}")


def run_all_strategies(
    symbol: str, start_date: str, end_date: str,
    init_cash: float, commission: float,
    output_dir: str, show_plot: bool
):
    """运行所有策略并打印对比表格"""
    print(f"\n[INFO] 获取 {symbol} 数据 ({start_date} ~ {end_date}) ...")
    df = fetch_stock_data(symbol, start_date, end_date)
    print(f"[INFO] 共 {len(df)} 条记录\n")

    summary = []
    for name, fn in STRATEGY_MAP.items():
        try:
            df_sig = fn(df.copy())
            result = run_backtest(df_sig, init_cash=init_cash, commission=commission)
            metrics = calc_metrics(result["portfolio"])
            metrics["策略"] = name
            summary.append(metrics)

            plot_strategy(
                df_sig, result["portfolio"],
                symbol=symbol,
                strategy_name=df_sig["strategy_name"].iloc[0] if "strategy_name" in df_sig.columns else name,
                save_dir=output_dir, show=show_plot
            )
        except Exception as e:
            print(f"[WARN] 策略 {name} 运行失败: {e}")

    if summary:
        import pandas as pd
        compare_df = pd.DataFrame(summary).set_index("策略")
        print("\n" + "=" * 60)
        print("  所有策略绩效对比")
        print("=" * 60)
        print(compare_df.to_string())
        print("=" * 60)

        compare_path = os.path.join(output_dir, f"{symbol}_strategy_compare.csv")
        compare_df.to_csv(compare_path, encoding="utf-8-sig")
        print(f"\n[INFO] 对比结果已保存: {compare_path}")


def main():
    parser = argparse.ArgumentParser(description="股票交易策略回测系统")
    parser.add_argument("--symbol",    default="000001", help="股票代码，如 000001")
    parser.add_argument("--strategy",  default="ma",
                        choices=list(STRATEGY_MAP.keys()) + ["all"],
                        help="策略名称 (ma/rsi/macd/bollinger/all)")
    parser.add_argument("--start",     default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end",       default="20241231", help="结束日期 YYYYMMDD")
    parser.add_argument("--cash",      type=float, default=100000.0, help="初始资金")
    parser.add_argument("--commission",type=float, default=0.0003,   help="手续费率")
    parser.add_argument("--output",    default="output",  help="输出目录")
    parser.add_argument("--no-plot",   action="store_true",           help="不显示图形窗口（仅保存文件）")
    parser.add_argument("--optimize",  action="store_true",           help="执行参数优化")
    parser.add_argument("--opt-metric",default="夏普比率",             help="参数优化目标指标")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    show_plot = not args.no_plot

    if args.strategy == "all":
        run_all_strategies(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            init_cash=args.cash,
            commission=args.commission,
            output_dir=args.output,
            show_plot=show_plot,
        )
    else:
        run_single_strategy(
            symbol=args.symbol,
            strategy_name=args.strategy,
            start_date=args.start,
            end_date=args.end,
            init_cash=args.cash,
            commission=args.commission,
            output_dir=args.output,
            show_plot=show_plot,
            do_optimize=args.optimize,
            opt_metric=args.opt_metric,
        )


if __name__ == "__main__":
    main()
