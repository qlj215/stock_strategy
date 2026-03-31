# 阶段3：feature_engineering.py 调参与输入输出说明（详细版）

> 分支：`feat/dl-plan-stage2`  
> 模块：`stock_strategy/feature_engineering.py`

---

## 1. 模块目标

阶段3模块负责在阶段2面板数据基础上构建：

1) 训练特征（X）  
2) 监督标签（y）

并产出后续阶段（基线模型 / DL 模型 / 组合回测）可直接消费的标准数据文件。

---

## 2. 输入与输出（I/O）

## 2.1 输入（默认）

- `data/processed/daily_panel.parquet`

关键输入字段（必须存在）：
- `date, stock_code, open, high, low, close, volume, amount`

行业相关特征依赖：
- `industry` 字段建议尽量完整（推荐先通过阶段2方案A补齐）

---

## 2.2 输出（默认）

- `data/features/features_v1.parquet`  
  - 含基础字段 + 特征字段 + 标签字段
- `data/features/feature_manifest.json`  
  - 记录参数、列名、样本规模、缺失率等元信息
  - 包含行业质量指标：`industry_nonnull_ratio`、`rel_ind_eq_rel_mkt_ratio`
- `data/features/label_spec.md`  
  - 标签定义文档（horizon、分位阈值、归一化模式）

---

## 3. 核心特征与标签设计

### 3.1 特征（当前31个）

主要分为：
- 动量收益：`ret_1/3/5/10/20/60`
- 波动率：`vol_5/10/20/60`
- 均线结构：`ma_gap_*`, `ma_cross_*`
- K线形态：`hl_spread`, `oc_change`
- 成交变化：`vol_chg_*`, `amt_chg_*`, `vol_ratio_*`
- 区间位置：`price_pos_20/60`
- 相对收益：`rel_mkt_ret_*`, `rel_ind_ret_*`

### 3.2 标签（按 horizon）

默认 horizon：`5,10`

每个 h 生成：
- `label_fwd_ret_{h}d`：未来收益
- `label_bench_ret_{h}d`：当日横截面平均未来收益（基准）
- `label_excess_ret_{h}d`：超额收益
- `label_rank_pct_{h}d`：横截面分位
- `label_cls_{h}d`：上下分位二分类（中间留空）

---

## 4. 如何调参（重点）

下面是你最常调的参数和建议：

### A) 预测周期（horizon）

参数：`--horizons`

- 快速版：`5,10`
- 稳健性扩展：`5,10,20`

示例：
```bash
python feature_engineering.py --horizons 5,10,20
```

---

### B) 最小历史长度过滤

参数：`--min-history`（默认 60）

作用：
- 控制每只股票从第几条历史开始纳入样本
- 避免 rolling 特征前段大量缺失

建议：
- `60`：默认稳定
- `80~120`：更稳但样本更少

示例：
```bash
python feature_engineering.py --min-history 90
```

---

### C) 二分类阈值

参数：`--cls-quantile`（默认 0.3）

含义：
- top 30% => 1
- bottom 30% => 0
- 中间 40% => NaN

建议：
- `0.3`：样本相对均衡
- `0.2`：信号更纯但样本更少

示例：
```bash
python feature_engineering.py --cls-quantile 0.2
```

---

### D) 归一化方式

参数：`--normalize`

- `xsec_zscore`（默认）：按日期横截面标准化
- `none`：不标准化

建议：
- 基线实验先用 `xsec_zscore`
- 对照实验再跑 `none`

示例：
```bash
python feature_engineering.py --normalize none
```

---

### E) 缺失值处理

参数：`--drop-na-features`

- 开启：删除含缺失特征或主标签缺失样本（更干净）
- 关闭：保留更多样本，后续模型侧自行处理缺失

建议：
- 阶段4先开启（便于快速出 baseline）
- 若开启后样本骤降，优先检查阶段2行业覆盖和 `feature_manifest.json` 的行业指标

示例：
```bash
python feature_engineering.py --drop-na-features
```

---

## 5. 推荐运行配方

### 配方1：默认稳定版（推荐）

```bash
python feature_engineering.py --drop-na-features
```

### 配方2：稳健性探索版

```bash
python feature_engineering.py \
  --horizons 5,10,20 \
  --min-history 90 \
  --cls-quantile 0.2 \
  --normalize xsec_zscore \
  --drop-na-features
```

---

## 6. 当前实跑结果（本次）

- 输入：`data/processed/daily_panel.parquet`
- 命令：`python feature_engineering.py --drop-na-features`
- 输出：
  - `data/features/features_v1.parquet`
  - `data/features/feature_manifest.json`
  - `data/features/label_spec.md`
- 样本：
  - rows = `22590`
  - stocks = `19`
  - days = `1431`

---

## 7. 后续衔接（阶段4）

阶段4基线模型可直接读取：
- `features_v1.parquet`
- `feature_manifest.json`

建议优先使用：
- 主标签：`label_excess_ret_5d`
- 对照标签：`label_excess_ret_10d`

这样可以无缝进入 Ridge/树模型实验与 RankIC 评价。