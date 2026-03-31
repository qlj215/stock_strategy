# 阶段2：build_dataset.py 调参与输入输出说明（详细版）

> 适用分支：`feat/dl-plan-stage2`  
> 模块文件：`stock_strategy/build_dataset.py`

---

## 1. 模块定位

`build_dataset.py` 是七阶段计划中的 **阶段2（研究数据集构建）** 核心脚本。  
目标是把 MiniQMT/xtdata 拉到的日频数据，整理成后续阶段可直接使用的标准面板数据。

它不改 `trainer_app.py` 主线逻辑，只服务于独立研究分支。

---

## 2. 输入形式（Input）

### 2.1 数据源输入（程序内部）

- 数据接口：`stock_strategy.data.fetcher.fetch_stock_data`
- 实际来源：MiniQMT + xtdata
- 频率：日频 K 线

### 2.2 命令行输入（CLI）

| 参数 | 说明 | 默认值 | 调参建议 |
|---|---|---|---|
| `--start` | 起始日期（YYYYMMDD） | `20200101` | 先短后长：先 3~5 年再扩 8~10 年 |
| `--end` | 结束日期（YYYYMMDD） | 当天 | 与回测窗口一致 |
| `--symbols` | 手工股票池（逗号分隔） | 空 | 复现实验优先手工指定 |
| `--limit` | 自动股票池上限 | `60` | 快速验证 20；正式研究 100~500 |
| `--adjust` | 复权口径 | `qfq` | 与训练/回测保持一致 |
| `--retries` | 单标拉取重试次数 | `2` | 网络波动时升到 3~5 |
| `--cache-dir` | 拉取缓存目录 | `output/cache` | 常用默认 |
| `--train-ratio` | 训练集比例 | `0.7` | 常用 0.6~0.75 |
| `--val-ratio` | 验证集比例 | `0.15` | 常用 0.1~0.2 |
| `--no-union-calendar` | 关闭并集日历补齐 | 关闭（即默认补齐） | 一般不建议关闭 |
| `--industry-scheme` | 行业补齐方案 | `scheme_a_local` | 方案A可用，方案B打通后切 `none` |
| `--industry-map-path` | 方案A本地行业映射CSV | `data/meta/industry_map_scheme_a.csv` | 映射维护入口 |
| `--industry-strict` | 映射不全时是否报错 | 关闭 | 数据质量闸门，训练前建议开启一次 |

> 校验规则：`train_ratio + val_ratio < 1`，否则脚本会报错退出。

---

## 3. 输出形式（Output）

默认输出目录：`stock_strategy/data/processed/`

### 3.1 `daily_panel.parquet`

- 路径：`data/processed/daily_panel.parquet`
- 结构：每行 = 某股票某交易日
- 唯一键：`(date, stock_code)`
- 主要字段：
  - `date, stock_code, open, high, low, close, volume, amount, adj_factor, is_trading, industry`

### 3.2 `split_manifest.json`

- 路径：`data/processed/split_manifest.json`
- 内容：按日期切分的 train/val/test 边界、行数、天数、参数快照
- 新增行业覆盖信息：
  - `industry.industry_scheme`
  - `industry.industry_symbol_coverage`
  - `industry.industry_row_coverage`
  - `industry.industry_missing_symbols`

### 3.3 `data_dictionary.md`

- 路径：`data/processed/data_dictionary.md`
- 内容：字段定义、补齐规则、去重规则、时间切分说明

---

## 4. 调参建议（按场景）

### 场景 A：快速冒烟（确认链路）

```bash
python build_dataset.py --start 20220101 --end 20260331 --limit 20
```

- 目标：几分钟内看到 parquet 产出
- 关注：字段完整性、唯一键、切分是否正确

### 场景 B：阶段2正式版

```bash
python build_dataset.py --start 20180101 --end 20260331 --limit 200 --retries 3
```

- 目标：用于阶段3特征工程
- 关注：样本覆盖、缺失占比、停牌补齐行为

### 场景 C：严格可复现

```bash
python build_dataset.py \
  --symbols 000001,600036,600519,000858,300750,002594,600276,603986,688981,300760 \
  --start 20190101 --end 20260331 --train-ratio 0.7 --val-ratio 0.15
```

- 目标：保证不同机器上股票池一致

### 场景 D：方案A行业补齐（当前推荐）

```bash
python build_dataset.py \
  --start 20200101 --end 20260331 --limit 20 \
  --industry-scheme scheme_a_local \
  --industry-map-path data/meta/industry_map_scheme_a.csv \
  --industry-strict
```

- 目标：在方案B不可用时，让 `industry` 字段可用。
- 关注：`split_manifest.json` 中行业覆盖率是否接近 1。

---

## 5. 当前实现细节与注意事项

1. `amount` 当前是 `close * volume` 的近似值（非原生成交额）  
   - 后续可在 fetcher 层补真实 `amount` 后替换。

2. 默认启用“并集交易日历补齐”  
   - 缺失/停牌行会设置 `is_trading=0`，`volume/amount=0`，价格列保持缺失。

3. 先去重再输出  
   - 对 `(date, stock_code)` 重复行保留最后一条。

4. 时间切分按日期而非随机  
   - 避免未来信息泄漏，符合后续时序训练要求。

5. 行业补齐方案A是独立可删层  
   - 只依赖 `data/industry_scheme_a.py` 与 `data/meta/industry_map_scheme_a.csv`。
   - 方案B打通后可切 `--industry-scheme none` 并删除方案A文件。

---

## 6. 模块与后续阶段衔接

- 阶段3：直接读取 `daily_panel.parquet` 构建标签和特征
- 阶段4：基线模型用 `split_manifest.json` 保持统一切分
- 阶段5：DL 模型训练输入同一份 panel，保证对比公平

---

## 7. 一句话总结

`build_dataset.py` 已经具备“可复现、可调参、可衔接后续阶段”的阶段2基础能力；你后续只要固定股票池和时间窗口，就可以稳定地推进阶段3/4/5。