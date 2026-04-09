# 阶段6.0 组合回测结果摘要

## 运行配置

- split: `test`
- features_file: `data/features/features_v1.parquet`
- label_col: `label_fwd_ret_5d`
- bench_col: `label_bench_ret_5d`
- rebalance_every: `5`
- top_n: `0`
- top_quantile: `0.2`
- buy_cost_rate: `0.0008`
- sell_cost_rate: `0.0018`

## 指标汇总

| signal_alias | signal_path | score_col | periods | gross_total_return | net_total_return | benchmark_total_return | annual_return | benchmark_annual_return | annual_volatility | sharpe | max_drawdown | win_rate | avg_turnover_buy | avg_turnover_sell | avg_selected | mean_gross_return | mean_net_return | mean_benchmark_return | excess_total_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage5_2_selector | data/dl/dl_v52_result.csv | pred_v52 | 45 | 0.39264 | 0.298905 | 0.269941 | 0.349266 | 0.314859 | 0.274341 | 1.230961 | -0.1121 | 0.577778 | 0.622222 | 0.6 | 4.0 | 0.008278 | 0.0067 | 0.005163 | 0.049031 |
| stage4_1_tree | data/baseline/baseline_v41_result.csv | pred_tree | 45 | 0.311917 | 0.235492 | 0.269941 | 0.274085 | 0.314859 | 0.29107 | 0.983019 | -0.117809 | 0.555556 | 0.538889 | 0.516667 | 4.0 | 0.007038 | 0.005677 | 0.005163 | -0.002183 |
| stage4_1_blend | data/baseline/baseline_v41_result.csv | pred_blend | 45 | 0.028608 | -0.048619 | 0.269941 | -0.055491 | 0.314859 | 0.305738 | 0.065887 | -0.240853 | 0.466667 | 0.688889 | 0.666667 | 4.0 | 0.002151 | 0.0004 | 0.005163 | -0.216459 |

## 说明

- 本版优先打通链路，默认使用 `label_fwd_ret_5d` 作为未来收益口径。
- 成本按换手近似扣减，属于研究级简化版，不是最终实盘撮合结果。