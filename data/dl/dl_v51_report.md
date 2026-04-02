# dl_report（阶段5：LSTM 第一版）

生成时间：2026-04-02T12:20:50.075505

## 1. 运行配置

- 输入特征：`data/features/features_v1.parquet`
- split_manifest：`data/processed/split_manifest.json`
- 目标标签：`label_excess_ret_5d`
- 特征数：`31`
- seq_len：`20`
- retrain_every：`20`
- train_window：`750`
- inner_val_days：`40`
- hidden_size/num_layers/dropout：`64/2/0.1`
- loss：`huber`
- lr/weight_decay：`0.001/0.0001`
- epochs/patience：`10/4`

### 设备自动辨识

- requested_device：`auto`
- selected_device：`cpu`
- torch_cuda_available：`False`
- cuda_device_count：`0`
- gpu_min_memory_gb：`4.0`
- selection_reason：`未检测到可用 CUDA，自动回退 CPU`

## 2. 输出规模

- 预测样本行数：`8483`
- 覆盖交易日：`448`
- 重训轮次：`23`
- 平均 best_epoch：`2.39`
- 平均 best_score：`0.001690`

## 3. 指标结果

| model | split | rows | days | rank_ic_mean | rank_ic_std | icir | direction_hit_rate | top_quantile_ret | bottom_quantile_ret | long_short_ret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lstm | val | 4246 | 225 | 0.052068 | 0.259448 | 0.200686 | 0.494348 | -0.000007 | -0.001918 | 0.001912 |
| lstm | test | 4237 | 223 | -0.016379 | 0.271472 | -0.060335 | 0.482889 | -0.002525 | 0.004600 | -0.007125 |
| lstm | all | 8483 | 448 | 0.017997 | 0.267408 | 0.067301 | 0.488624 | -0.001260 | 0.001326 | -0.002586 |

## 4. 备注

- 当前版本以“链路打通 + 可复现”为优先，后续可继续做结构与超参增强。
- 指标口径复用阶段4 calc_metrics，便于与 baseline 同口径对照。