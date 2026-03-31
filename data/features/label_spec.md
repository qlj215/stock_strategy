# label_spec（阶段3）

## 标签定义

当前标签使用未来收益构造，预测周期：`5, 10` 日。

对每个 horizon=h，定义：
- `label_fwd_ret_{h}d`：个股未来 h 日收益
- `label_bench_ret_{h}d`：当日横截面平均未来 h 日收益（基准）
- `label_excess_ret_{h}d`：超额收益 = 个股未来收益 - 基准未来收益
- `label_rank_pct_{h}d`：超额收益横截面分位（0~1）
- `label_cls_{h}d`：二分类标签（上分位=1，下分位=0，中间为空）

二分类分位阈值：`0.3`
- 上分位阈值：`>= 0.70`
- 下分位阈值：`<= 0.30`

## 特征归一化

- normalize 模式：`xsec_zscore`
- `xsec_zscore`：按日期做横截面 z-score（不跨期）
- `none`：不做归一化
