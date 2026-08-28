# Stock Strategy —— A股量化研究与交易辅助系统

从数据接入、特征/标签构建、模型训练、组合回测，到实时行情辅助决策的**完整闭环**。
包含一个可运行的 Flask Web 服务端和一条七阶段研究管线，核心 Python 代码约 1 万行（另有全功能前端页面）。

> 本项目用于个人学习与研究，不构成任何投资建议。

## 功能总览

- **实时行情工作台**（Web）：日线 + 1 分钟分时合成，按开盘/午休/收盘进度外推估算当日量能；rule / 深度学习双后端上涨概率预测（可解释规则评分或 PyTorch 模型，auto 模式自动回退）；全市场扫描（行业/板块过滤、四种候选模式、异步任务进度）；历史组合回测；K 线训练挑战与 AI 复盘。
- **组合回测引擎**：手动/全市场主板/单行业股票池；5 类规则策略 + 模型 Top-K；T+1 信号执行、按板块区分涨停（10%/20%/30%）不追买、停牌跳过、非重叠持仓批次；买卖佣金/最低佣金/过户费/印花税/滑点成本模型；净值曲线 vs 等权基准、年化/夏普/回撤/胜率。
- **七阶段研究管线**：数据集构建 → 31 维特征与前瞻标签 → Ridge/CART 基线 → 行业中性化稳健版 → LSTM 滚动训练（GPU 自动辨识）→ 动态模型选择 → 横截面组合回测，全程 walk-forward、按时间切分、防前视偏差。
- **组合收益试算工具**：按金额/手数、按天/精确到分的假想交易收益试算（独立小 Web 应用）。

## 技术栈

Python 3.12 · pandas / numpy · PyTorch · Flask · MiniQMT / xtquant（Windows 行情客户端）· pyarrow / parquet

## 目录结构

```
stock_strategy/
├── trainer_app.py              Web 服务端：行情工作台 / 全市场扫描 / 组合回测 / 训练挑战
├── probability_backend.py      概率预测后端：rule / dl / auto 三模式与回退机制
├── dl_entry_template.py        PyTorch DL 推理入口（模型加载、bootstrap 自训练、批量推理）
├── portfolio_whatif_web.py     假想组合收益试算工具（独立小 Web 应用）
├── config/
│   └── model_backend.json      预测后端配置（默认后端 / DL 入口）
├── data/
│   ├── fetcher.py              MiniQMT/xtdata 行情封装（复权、缓存、WSL 兼容、自动拉起终端、连接就绪探测）
│   ├── industry_scheme_a.py    本地行业映射注入（独立可删的补齐方案）
│   └── meta/                   行业映射参考表
├── research_pipeline/          七阶段研究管线（脚本化、可独立复现）
│   ├── build_dataset.py        阶段2  标准日线面板数据集
│   ├── feature_engineering.py  阶段3  特征与前瞻收益标签
│   ├── train_baseline.py       阶段4  Ridge + CART 基线（walk-forward）
│   ├── train_baseline_v41.py   阶段4.1 稳健版（行业中性化 + 动态融合）
│   ├── train_lstm.py           阶段5  LSTM 滚动训练 + GPU 自动辨识
│   ├── train_lstm_v52.py       阶段5.2 LSTM/anchor 动态模型选择
│   ├── portfolio_backtest.py   阶段6  横截面信号组合回测
│   └── model_config.yaml       各阶段推荐参数模板
├── tests/
│   └── test_smoke.py           14 个离线冒烟回归用例（无需行情环境/显卡/网络）
└── web/                        前端页面（训练挑战页 + 行情工作台页）
```

## 核心设计

### 概率预测双后端

- **rule**：可解释规则评分——短期动量、均线结构、量能比、近 10 日胜率加权映射为当日 / 5 日 / 长期上涨概率，`rule_no_chase` 变体叠加高位追高抑制；输出附带自然语言理由与特征值。
- **dl**：插件式接入（配置声明 `module:function` 入口），支持单条与批量推理；模型缺失时自动 bootstrap 训练兜底。
- **auto**：优先 DL，失败自动回退 rule；每条预测统一标注 `backend / backend_fallback / prob_model` 元信息，来源可追溯。

### 防前视 / 防泄漏贯穿研究管线

- 特征仅由当日及历史数据计算，标签使用未来收益且仅作监督目标；
- 训练/验证/测试按时间切分（70/15/15），walk-forward 滚动训练；
- 标签为个股未来收益减当日横截面均值（超额收益），横截面分位构造二分类标签；
- 阶段 5.2 动态选择器按滞后标签滚动评估 LSTM 与 anchor 的近期 IC 并动态切换。

### 行情连接的可靠性

- MiniQMT 不可用时按"动态行业板块 → 本地面板 → 内置股票池"三级回退，接口优雅降级而非报错；
- xtdata 连接带端口候选（环境变量 / `~/.xtquant/xtdata.cfg` / 常见默认端口）与行情服务就绪探活，区分"未连接"与"已连接但行情源不可用"并给出诊断；
- 支持 `--auto-start-qmt` 通过 PowerShell 自动拉起 XtMiniQmt.exe 与 miniquote.exe（WSL 兼容）。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 离线自测（无需 MiniQMT / GPU / 网络，约 1 分钟）
python tests/test_smoke.py

# 2) 启动 Web 工作台（默认端口读 STOCK_STRATEGY_PORT，否则 8789）
python trainer_app.py
python trainer_app.py --port 8789 --auto-start-qmt

# 3) 组合收益试算（独立端口）
python portfolio_whatif_web.py --no-browser

# 4) 研究管线示例（需 MiniQMT 行情；在仓库根目录运行）
python research_pipeline/build_dataset.py --start 20200101 --end 20260331 --limit 20
python research_pipeline/feature_engineering.py --drop-na-features
python research_pipeline/train_baseline.py
python research_pipeline/train_lstm.py --retrain-every 60 --epochs 8
python research_pipeline/portfolio_backtest.py \
  --signal-config data/dl/dl_v52_result.csv::pred_v52::stage5_2_selector \
  --split test --rebalance-every 5
```

说明：

- 实时行情功能依赖本机 MiniQMT 客户端（Windows，WSL 下可自动探活/拉起）；无行情环境下，离线自测与"本地面板"回测链路仍可完整运行；
- 运行产物（日线面板 parquet、训练结果 CSV、模型 checkpoint 等）体积较大且可再生，均不入库，可按上述管线命令重新生成（研究阶段 2 起）。
