# dl_report（阶段5：LSTM 第一版）

生成时间：2026-04-02T16:23:57.165812

## 1. 运行配置

- 输入特征：`data/features/features_v1.parquet`
- split_manifest：`data/processed/split_manifest.json`
- 目标标签：`label_excess_ret_5d`
- 特征数：`31`
- seq_len：`20`
- retrain_every：`40`
- train_window：`expanding`
- inner_val_days：`60`
- hidden_size/num_layers/dropout：`64/2/0.1`
- loss：`huber`
- lr/weight_decay：`0.001/0.0001`
- epochs/patience：`12/4`

### 设备自动辨识

- requested_device：`auto`
- selected_device：`cpu`
- torch_cuda_available：`False`
- cuda_device_count：`0`
- gpu_min_memory_gb：`0.0`
- selection_reason：`未检测到可用 CUDA，自动回退 CPU`

## 2. 输出规模

- 预测样本行数：`8483`
- 覆盖交易日：`448`
- 重训轮次：`12`
- 平均 best_epoch：`1.42`
- 平均 best_score：`0.001725`
- 最新模型权重：`data\dl\checkpoints\lstm_latest.pt`
- 最新模型元信息：`data\dl\checkpoints\lstm_latest_meta.json`
- 最新模型训练区间：`2020-05-08 -> 2026-03-12`
- 最新模型预测覆盖：`2026-03-13 -> 2026-03-24`

## 3. 指标结果

| model | split | rows | days | rank_ic_mean | rank_ic_std | icir | direction_hit_rate | top_quantile_ret | bottom_quantile_ret | long_short_ret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lstm | val | 4246 | 225 | 0.073435 | 0.279608 | 0.262638 | 0.521196 | 0.000956 | -0.004212 | 0.005168 |
| lstm | test | 4237 | 223 | 0.004885 | 0.270289 | 0.018072 | 0.459287 | -0.002979 | 0.004651 | -0.007630 |
| lstm | all | 8483 | 448 | 0.039313 | 0.276836 | 0.142009 | 0.490275 | -0.001003 | 0.000199 | -0.001202 |

## 4. 备注

- 当前版本以“链路打通 + 可复现”为优先，后续可继续做结构与超参增强。
- 指标口径复用阶段4 calc_metrics，便于与 baseline 同口径对照。