# baseline_report（阶段4）

生成时间：2026-04-01T17:43:42.302350

## 1. 运行配置

- 输入特征：`C:\Users\24333\Desktop\university\股票交易策略项目\stock_strategy\data\features\features_v1.parquet`
- 切分清单：`C:\Users\24333\Desktop\university\股票交易策略项目\stock_strategy\data\processed\split_manifest.json`
- 目标标签：`label_excess_ret_5d`
- 特征数：`31`
- 滚动重训周期：`20` 日
- 训练窗口：`expanding`
- Ridge alpha：`1.0`
- Tree depth/min_leaf/thresholds：`3/80/8`

## 2. 输出文件

- 预测明细：`C:\Users\24333\Desktop\university\股票交易策略项目\stock_strategy\data\baseline\baseline_result.csv`
- 指标汇总：`C:\Users\24333\Desktop\university\股票交易策略项目\stock_strategy\data\baseline\baseline_metrics.csv`
- 预测样本行数：`8502`

## 3. 指标结果

### ridge

| split | rows | days | rank_ic_mean | rank_ic_std | icir | direction_hit_rate | top_quantile_ret | bottom_quantile_ret | long_short_ret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 4265 | 225 | 0.090652 | 0.309642 | 0.292766 | 0.521219 | 0.002794 | -0.006671 | 0.009466 |
| test | 4237 | 223 | 0.008209 | 0.268950 | 0.030521 | 0.478405 | -0.001333 | 0.002720 | -0.004053 |
| all | 8502 | 448 | 0.049615 | 0.292700 | 0.169507 | 0.499882 | 0.000740 | -0.001997 | 0.002737 |

### tree

| split | rows | days | rank_ic_mean | rank_ic_std | icir | direction_hit_rate | top_quantile_ret | bottom_quantile_ret | long_short_ret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 4265 | 225 | 0.028106 | 0.283086 | 0.099285 | 0.496366 | -0.001801 | 0.001968 | -0.003768 |
| test | 4237 | 223 | -0.013286 | 0.266346 | -0.049881 | 0.452443 | -0.009532 | 0.003519 | -0.013051 |
| all | 8502 | 448 | 0.007503 | 0.275354 | 0.027248 | 0.474477 | -0.005365 | 0.002683 | -0.008048 |
