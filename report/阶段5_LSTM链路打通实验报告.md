# 阶段5：LSTM链路打通实验报告

> 分支：`feat/dl-plan-stage2`  
> 关联阶段：阶段3（特征标签）→ 阶段4（基线）→ 阶段5（本报告）

---

## 1. 阶段目标

按计划书 05 的要求，先用 **LSTM** 完成阶段5第一版闭环：

1. 构造过去 20 日时序输入；
2. 输出未来 5 日超额收益预测（`label_excess_ret_5d`）；
3. 沿用阶段4同口径指标（RankIC / ICIR / 分层收益 / 命中率）；
4. 形成可复现脚本、结果文件、说明文档，并更新进度。

本次优先级：**先把链路打通并稳定可跑**，再做深度调参与结构增强。

---

## 2. 本次新增内容

### 2.1 新增脚本与配置

- `research_pipeline/train_lstm.py`：阶段5主脚本（LSTM + 滚动重训 + 指标输出）
- `research_pipeline/model_config.yaml`：阶段5参数模板（便于后续复用与调参记录）

### 2.2 新增数据产物

- `data/dl/dl_result.csv`：逐日逐股票预测结果
- `data/dl/dl_metrics.csv`：val/test/all 指标汇总
- `data/dl/dl_trainlog.csv`：每次重训的训练窗口与早停信息
- `data/dl/dl_report.md`：脚本自动生成报告

### 2.3 文档与进度

- 新增本报告：`report/阶段5_LSTM链路打通实验报告.md`
- 更新：`plans/a_stock_dl_project_plans/PROGRESS.md`
  - 阶段5状态已从 `[ ]` 更新为 `[x]`

---

## 3. 代码使用说明（已写入脚本）

`research_pipeline/train_lstm.py` 顶部已补充“输入输出说明 + 调参速查 + 运行示例”。

### 3.1 默认运行

```bash
python research_pipeline/train_lstm.py
```

### 3.2 本次实跑命令

```bash
python research_pipeline/train_lstm.py \
  --in data/features/features_v1.parquet \
  --split-manifest data/processed/split_manifest.json \
  --feature-manifest data/features/feature_manifest.json \
  --out data/dl/dl_result.csv \
  --metrics-out data/dl/dl_metrics.csv \
  --trainlog-out data/dl/dl_trainlog.csv \
  --report-out data/dl/dl_report.md
```

### 3.3 本版关键参数（默认）

- `seq_len=20`
- `retrain_every=40`
- `train_window=0`（扩展窗口）
- `inner_val_days=60`
- `hidden_size=64`，`num_layers=2`，`dropout=0.1`
- `loss=huber`，`epochs=12`，`early_stop_patience=4`

---

## 4. 实现要点

### 4.1 时序样本构造

- 按 `stock_code` 分组，按 `date` 排序；
- 每个样本用过去 `seq_len` 天的特征序列预测当日目标标签；
- 采用固定窗口滑动构造，输出 `N x T x F` 张量（本次 `T=20`，`F=31`）。

### 4.2 训练与预测方式

- 沿用阶段4的时间滚动思想：
  - 每隔 `retrain_every` 天重训一次；
  - 用重训后模型预测下一个区间；
- 每次重训在历史窗口尾部切出 `inner_val_days` 做早停验证；
- 预测输出保留 `fit_start_date / fit_end_date`，便于回溯每段模型训练区间。

### 4.3 指标口径

- 直接复用阶段4 `calc_metrics`：
  - RankIC（按日横截面 Spearman）
  - ICIR
  - Top/Bottom/Long-Short
  - Direction Hit Rate

---

## 5. 本次结果（2026-04-02）

### 5.1 输出规模

- 预测样本：`8483` 行
- 覆盖交易日：`448` 天（val+test）
- 重训轮次：`12`

### 5.2 LSTM 指标

| split | rows | days | rank_ic_mean | icir | long_short_ret | direction_hit_rate |
|---|---:|---:|---:|---:|---:|---:|
| val | 4246 | 225 | 0.073435 | 0.262638 | 0.005168 | 0.521196 |
| test | 4237 | 223 | 0.004885 | 0.018072 | -0.007630 | 0.459287 |
| all | 8483 | 448 | 0.039313 | 0.142009 | -0.001202 | 0.490275 |

---

## 6. 与阶段4基线的对照（test）

- 阶段4.1 Tree：RankIC `0.030204`，Long-Short `0.007594`
- 阶段4.1 Blend：RankIC `0.013315`，Long-Short `0.003334`
- 本次 LSTM：RankIC `0.004885`，Long-Short `-0.007630`

解读：
1. LSTM 首版已完成技术闭环和可复现实验流程；
2. 但当前 test 表现尚未超过 4.1 稳健基线；
3. 初步判断仍存在过拟合/时变适配不足问题，需要继续调优。

---

## 7. 下一步建议（承接阶段6前）

建议先做一轮阶段5.1强化，再进入阶段6组合映射：

1. **训练策略增强**
   - 缩短 `train_window`（如 750）对抗时变；
   - 提高重训频率（`retrain_every=20`）；
   - 增加多随机种子复现实验（避免单次偶然性）。

2. **结构增强**
   - 对比 `1层LSTM` vs `2层LSTM`；
   - 尝试 GRU / TCN 轻量替代；
   - 增加输出头正则（dropout / weight decay）稳定泛化。

3. **目标与损失增强**
   - 在回归损失基础上加入排序项（如 rank-aware 目标）；
   - 对极端收益样本做稳健处理（clip/winsor）。

4. **融合增强**
   - 先做 `LSTM + baseline_v41_blend` 简单融合，评估增量价值。

---

## 8. 结论

阶段5（LSTM第一版）已实现：

- ✅ 代码链路打通（可运行、可复现）
- ✅ 同口径指标可对照
- ✅ 报告与进度已更新

当前模型效果仍偏“验证集可用、测试集偏弱”，后续将按“窗口/重训/结构/融合”四条线继续迭代。
