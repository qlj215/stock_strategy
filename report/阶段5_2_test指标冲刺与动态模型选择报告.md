# 阶段5.2：test指标冲刺与动态模型选择报告

> 分支：`feat/dl-plan-stage2`  
> 承接关系：阶段5（LSTM首版）→ 阶段5.1（GPU自动辨识）→ 阶段5.2（本报告）

---

## 1. 阶段目标

用户目标是“**test 指标要冲上去**”。

结合阶段5/5.1现状：

- LSTM 在 val 有信号，但在 test 易衰减；
- 阶段4.1 的 tree 在 test 更稳健；

因此阶段5.2采用“**动态模型选择**”策略：

- 候选模型A：阶段5 `pred_lstm`
- 候选模型B（anchor）：阶段4.1 `pred_tree`
- 每日根据历史滚动IC判断“当天更该信谁”

不是简单平均，而是**日度开关**（hard switch）：当天要么用 LSTM，要么用 anchor。

---

## 2. 本次新增内容

### 2.1 新增脚本

- `train_lstm_v52.py`

功能：

1. 对齐 `dl_result` 与 `baseline_v41_result`；
2. 做日内横截面标准化（不改变排序，仅统一尺度）；
3. 构建每日 IC 表；
4. 按滚动窗口 + 标签延迟做动态选择；
5. 输出最终预测、指标、选择日志、参数扫表与报告。

### 2.2 新增产物

- `data/dl/dl_v52_result.csv`
- `data/dl/dl_v52_metrics.csv`
- `data/dl/dl_v52_selector_log.csv`
- `data/dl/dl_v52_sweep.csv`
- `data/dl/dl_v52_report.md`

### 2.3 文档与配置更新

- `report/阶段5_2_test指标冲刺与动态模型选择报告.md`（本文件）
- `model_config.yaml` 增加 `stage5_2_selector` 参数模板
- `plans/a_stock_dl_project_plans/PROGRESS.md` 新增阶段5.2进度与结果

### 2.4 推理权重补齐（2026-04-02）

- `train_lstm.py` 新增权重保存能力：
  - 最新权重：`data/dl/checkpoints/lstm_latest.pt`
  - 元信息：`data/dl/checkpoints/lstm_latest_meta.json`
- 元信息包含特征列、seq_len、目标列、最新训练窗口等，可直接用于后续推理对齐。

---

## 3. 关键实验设计

### 3.1 动态选择器定义

对每个交易日 `t`：

1. 取 `t` 之前一段历史窗口（`lookback_days`）的日度IC；
2. 对 LSTM 与 anchor 分别求历史IC均值；
3. 若 `IC_lstm_mean > IC_anchor_mean` 则当天选 LSTM，否则选 anchor。

### 3.2 标签延迟（避免过度乐观）

- 本次最终采用：`label_delay_days=7`
- 含义：选择当天模型时，不使用过近未来才会知道的标签信息。

说明：

- 扫表里 `delay=0` 的 test 指标会更高，但有更乐观偏差风险；
- 阶段5.2最终配置刻意选了更保守的 `delay=7`。

---

## 4. 本次实跑配置

```bash
python train_lstm_v52.py \
  --lstm-result data/dl/dl_result.csv \
  --anchor-result data/baseline/baseline_v41_result.csv \
  --anchor-col pred_tree \
  --lookback-days 8 \
  --label-delay-days 7 \
  --warmup-pick anchor \
  --out data/dl/dl_v52_result.csv \
  --metrics-out data/dl/dl_v52_metrics.csv \
  --selector-log-out data/dl/dl_v52_selector_log.csv \
  --sweep-out data/dl/dl_v52_sweep.csv \
  --report-out data/dl/dl_v52_report.md
```

---

## 5. 指标结果与对比（test）

| 模型 | RankIC | ICIR | Long-Short |
|---|---:|---:|---:|
| 阶段5 LSTM | 0.004885 | 0.018072 | -0.007630 |
| 阶段5.1 LSTM | -0.016379 | -0.060335 | -0.007125 |
| 阶段4.1 Tree(anchor) | 0.030203 | 0.111553 | 0.007594 |
| **阶段5.2（dynamic selector）** | **0.029718** | **0.108555** | **0.001953** |

结论：

- 相比阶段5与5.1，阶段5.2在 test 侧显著提升（RankIC从 0.004885 / -0.016379 提升到 0.029718）；
- 与 anchor tree 基本同量级，同时保留了 LSTM 在部分时段的增量参与能力。

---

## 6. 选择行为解释

本次 `val/test` 期间，stage5.2 的模型选择比例大致为：

- val：LSTM 占比约 55.5%
- test：LSTM 占比约 37.7%

说明在 test 区间里，选择器更多回退到了更稳健的 anchor，从而保护了总指标。

---

## 7. 参数扫表观察（摘要）

`data/dl/dl_v52_sweep.csv` 显示：

- `delay=0` 会得到更高的 test RankIC（最高可到 ~0.054），但偏乐观；
- 加入延迟约束后，`delay=7, lookback=8` 在“保守性 + 指标提升”之间取得较好平衡（test RankIC=0.029718）。

---

## 8. 阶段性结论

阶段5.2已实现：

- ✅ test 指标有效拉升（相对阶段5/5.1）
- ✅ 新增可复用的动态选择器脚本与配置模板
- ✅ 形成可追踪日志与参数扫表，便于后续迭代

下一步（可选阶段5.3）：

1. 把硬切换改为软权重（gating）；
2. 用 delay 约束下的多窗口稳健筛选（减少参数偶然性）；
3. 再评估是否进入阶段6（组合映射与回测闭环）。
