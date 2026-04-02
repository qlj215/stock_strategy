# 阶段5.1：LSTM稳健增强与GPU自动辨识报告

> 分支：`feat/dl-plan-stage2`  
> 承接关系：阶段5（LSTM链路打通）→ 阶段5.1（本报告）

---

## 1. 阶段5.1目标

在已完成阶段5链路打通的基础上，继续做两件事：

1. **工程增强**：新增“是否可用 GPU”的自动辨识能力，避免人工反复试错；
2. **实验增强**：用近期窗口+更高重训频率做一版稳健性实验（v5.1）。

---

## 2. 本次代码增强（train_lstm.py）

### 2.1 新增 GPU 自动辨识逻辑

- 新增设备解析函数：`resolve_device_with_profile(...)`
- 行为规则：
  - `--device cpu`：固定 CPU
  - `--device cuda`：强制 CUDA（不可用直接报错）
  - `--device auto`：优先尝试可用 GPU，失败自动回退 CPU

### 2.2 新增 GPU 选择参数

- `--gpu-index`：优先尝试的 GPU 序号
- `--gpu-min-memory-gb`：GPU 最小显存阈值（GB）

### 2.3 新增运行时探针与记录

- 对候选 GPU 做轻量算子探针，确认“可见且可算”；
- 控制台输出 `requested/selected/reason`；
- 自动写入 `dl_report` 的“设备自动辨识”小节，便于复盘环境。

---

## 3. 阶段5.1实跑配置

### 3.1 运行命令

```bash
python train_lstm.py \
  --in data/features/features_v1.parquet \
  --split-manifest data/processed/split_manifest.json \
  --feature-manifest data/features/feature_manifest.json \
  --device auto \
  --gpu-min-memory-gb 4 \
  --train-window 750 \
  --retrain-every 20 \
  --epochs 10 \
  --inner-val-days 40 \
  --out data/dl/dl_v51_result.csv \
  --metrics-out data/dl/dl_v51_metrics.csv \
  --trainlog-out data/dl/dl_v51_trainlog.csv \
  --report-out data/dl/dl_v51_report.md
```

### 3.2 设备自动辨识结果

- requested_device：`auto`
- torch_cuda_available：`False`
- selected_device：`cpu`
- 回退原因：`未检测到可用 CUDA，自动回退 CPU`

---

## 4. 新增产物

- `data/dl/dl_v51_result.csv`
- `data/dl/dl_v51_metrics.csv`
- `data/dl/dl_v51_trainlog.csv`
- `data/dl/dl_v51_report.md`
- `model_config.yaml`（新增 `stage5_1_lstm` 配置模板）

---

## 5. 指标结果（v5.1）

| split | rows | days | rank_ic_mean | icir | long_short_ret | direction_hit_rate |
|---|---:|---:|---:|---:|---:|---:|
| val | 4246 | 225 | 0.052068 | 0.200686 | 0.001912 | 0.494348 |
| test | 4237 | 223 | -0.016379 | -0.060335 | -0.007125 | 0.482889 |
| all | 8483 | 448 | 0.017997 | 0.067301 | -0.002586 | 0.488624 |

---

## 6. 与阶段5首版对照（test）

- 阶段5首版：RankIC `0.004885`，ICIR `0.018072`，Long-Short `-0.007630`
- 阶段5.1本次：RankIC `-0.016379`，ICIR `-0.060335`，Long-Short `-0.007125`

解读：

1. 本次阶段5.1主要目标（GPU自动辨识）已完成并验证可用；
2. 在当前参数下，test 排序表现未提升，说明“近期窗口+高频重训”不必然带来收益；
3. 下一步应继续做系统调参/结构对照，而不是仅依赖窗口缩短。

---

## 7. 下一步建议（阶段5.2候选）

1. 多种子重复（至少3个 seed）评估稳定性；
2. 对比 `num_layers=1` 与 `hidden_size=32/64` 的小模型，降低过拟合；
3. 试验 `train_window=0/500/750` 三档，配合统一 `retrain_every`；
4. 增加“LSTM + baseline_v41_blend”融合实验，优先看 test 的 RankIC/Long-Short 是否转正。
