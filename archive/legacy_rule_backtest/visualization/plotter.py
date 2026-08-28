"""
可视化模块
- 绘制价格走势 + 技术指标 + 买卖信号
- 绘制净值曲线对比图
- 绘制回撤曲线
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import os

# 解决中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_strategy(
    df: pd.DataFrame,
    portfolio: pd.DataFrame,
    symbol: str = "",
    strategy_name: str = "",
    save_dir: str = "output",
    show: bool = True,
) -> str:
    """
    绘制综合策略图，包含：
      - 价格 + 技术指标 + 买卖信号
      - 成交量
      - 净值曲线（策略 vs 买入持有）
      - 回撤曲线
    """
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 2, 1.5], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])  # 价格 + 指标 + 信号
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # 成交量
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # 净值曲线
    ax4 = fig.add_subplot(gs[3], sharex=ax1)  # 回撤

    title = f"{symbol} {strategy_name} 策略回测"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── ax1: 价格与指标 ──
    ax1.plot(df.index, df["close"], label="收盘价", color="black", linewidth=1.2, zorder=3)

    # 均线
    for col_name, color in [
        ("ma5",  "#e67e22"), ("ma10", "#9b59b6"), ("ma20", "#2980b9"),
        ("ema12", "#27ae60"), ("ema26", "#c0392b")
    ]:
        if col_name in df.columns:
            ax1.plot(df.index, df[col_name], label=col_name.upper(), linewidth=0.9, alpha=0.8, color=color)

    # 布林带
    if "boll_upper" in df.columns:
        ax1.fill_between(df.index, df["boll_lower"], df["boll_upper"], alpha=0.1, color="steelblue", label="布林带")
        ax1.plot(df.index, df["boll_mid"], linewidth=0.8, color="steelblue", linestyle="--")

    # 买卖信号
    buy_signals  = portfolio[portfolio["signal"] == 1]
    sell_signals = portfolio[portfolio["signal"] == -1]
    if not buy_signals.empty:
        ax1.scatter(buy_signals.index, df.loc[buy_signals.index, "close"],
                    marker="^", color="red", s=80, zorder=5, label="买入")
    if not sell_signals.empty:
        ax1.scatter(sell_signals.index, df.loc[sell_signals.index, "close"],
                    marker="v", color="green", s=80, zorder=5, label="卖出")

    ax1.set_ylabel("价格 (元)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.3)

    # ── ax2: 成交量 ──
    if "volume" in df.columns:
        colors = ["red" if df["close"].iloc[i] >= df["open"].iloc[i] else "green"
                  for i in range(len(df))]
        ax2.bar(df.index, df["volume"] / 1e4, color=colors, alpha=0.7, width=1)
        ax2.set_ylabel("成交量\n(万股)", fontsize=9)
        ax2.grid(True, alpha=0.3)

    # ── ax3: 净值曲线 ──
    ax3.plot(portfolio.index, (portfolio["total_value"] / portfolio["total_value"].iloc[0]),
             label="策略净值", color="royalblue", linewidth=1.4)
    ax3.plot(portfolio.index, (portfolio["bh_value"] / portfolio["bh_value"].iloc[0]),
             label="买入持有", color="gray", linewidth=1.0, linestyle="--")
    ax3.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax3.set_ylabel("净值", fontsize=10)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── ax4: 最大回撤 ──
    cum_max = portfolio["total_value"].cummax()
    drawdown = (portfolio["total_value"] - cum_max) / cum_max * 100
    ax4.fill_between(portfolio.index, drawdown, 0, color="salmon", alpha=0.7, label="回撤")
    ax4.set_ylabel("回撤 (%)", fontsize=9)
    ax4.legend(loc="lower left", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # X轴格式
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)

    save_path = os.path.join(save_dir, f"{symbol}_{strategy_name}_backtest.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return save_path


def plot_optimization(results_df: pd.DataFrame, x_col: str, y_col: str,
                      metric: str = "夏普比率", save_dir: str = "output",
                      strategy_name: str = "", show: bool = True) -> str:
    """绘制参数优化热力图（二维参数时）或折线图（一维参数时）"""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    if x_col == y_col or y_col not in results_df.columns:
        # 一维：折线图
        ax.plot(results_df[x_col], results_df[metric], marker="o", color="royalblue")
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric)
        ax.set_title(f"{strategy_name} 参数优化: {x_col} vs {metric}")
        ax.grid(True, alpha=0.3)
    else:
        # 二维：热力图
        pivot = results_df.pivot_table(index=y_col, columns=x_col, values=metric)
        import matplotlib.cm as cm
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{strategy_name} 参数优化热力图: {metric}")
        plt.colorbar(im, ax=ax, label=metric)

    save_path = os.path.join(save_dir, f"{strategy_name}_optimization.png")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return save_path
