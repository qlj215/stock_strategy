# A股DL计划执行进度（独立分支）

- 分支：`feat/dl-plan-stage2`
- 原则：不改主线 `trainer_app.py` 业务逻辑；阶段化推进。

## 阶段状态

- [x] 阶段1：数据通路打通（已有 MiniQMT + xtdata）
- [x] 阶段2：研究数据集构建（本次完成首版）
- [x] 阶段3：标签与特征工程（本次完成首版）
- [x] 阶段4：基线模型实验（本次完成首版）
- [x] 阶段5：深度学习模型第一版（LSTM链路打通首版）
- [x] 阶段5.1：LSTM稳健增强 + GPU自动辨识（本次完成）
- [x] 阶段5.2：LSTM test指标冲刺（动态模型选择）
- [ ] 阶段6：组合回测系统
- [ ] 阶段7：稳健性分析与最终总结

## 阶段2交付

- 脚本：`build_dataset.py`
- 数据：`data/processed/daily_panel.parquet`
- 切分：`data/processed/split_manifest.json`
- 字典：`data/processed/data_dictionary.md`
- 说明报告：`report/阶段2_build_dataset模块调参与输入输出说明.md`
- 方案A独立补丁：
  - `data/industry_scheme_a.py`
  - `data/meta/industry_map_scheme_a.csv`
  - `report/阶段2_方案A本地行业映射独立补丁说明.md`

### 本次实跑结果（2026-03-31）

- 参数：`--start 20200101 --end 20260331 --limit 20`
- 输出行数：`30220`
- 股票数：`20`
- 交易日数：`1511`
- 训练/验证/测试：`1057 / 226 / 228` 日
- 行业覆盖（方案A）：symbol=1.0，row=1.0

### 更新实跑结果（2026-04-01）

- 参数：`--start 20200101 --end 20260331 --limit 20 --prelisting-null-mode drop`
- 输出行数：`24105`
- 股票数：`20`
- 交易日数：`1511`
- 上市前空K线剔除：`6115` 行，影响 `10` 只股票
- 行业覆盖（方案A）：symbol=1.0，row=1.0

## 阶段3交付

- 脚本：`feature_engineering.py`
- 特征标签表：`data/features/features_v1.parquet`
- 特征清单：`data/features/feature_manifest.json`
- 标签说明：`data/features/label_spec.md`
- 说明报告：`report/阶段3_feature_engineering模块调参与输入输出说明.md`

### 本次实跑结果（2026-03-31）

- 运行命令：`python feature_engineering.py --drop-na-features`
- 输入（阶段2）：`data/processed/daily_panel.parquet`
- 输出行数：`22590`
- 股票数：`19`
- 交易日数：`1431`
- 特征数：`31`
- 标签周期：`5, 10` 日
- 行业有效率：`industry_nonnull_ratio=1.0`
- 行业相对收益非退化：`rel_ind_eq_rel_mkt_ratio={'1d': 0.0, '5d': 0.0}`

### 更新实跑结果（2026-04-01）

- 运行命令：`python feature_engineering.py --drop-na-features --industry-feature-mode zero`
- 输入（阶段2，drop 后）：`data/processed/daily_panel.parquet`
- 输出行数：`22590`
- 股票数：`19`
- 交易日数：`1431`
- 行业特征模式：`zero`（`rel_ind_ret_1/5` 全为 0）

## 阶段4交付

- 脚本：`train_baseline.py`
- 预测结果：`data/baseline/baseline_result.csv`
- 指标汇总：`data/baseline/baseline_metrics.csv`
- 实验报告：`data/baseline/baseline_report.md`
- 阶段总报告：`report/阶段4_基线模型实验整体报告.md`

### 本次实跑结果（2026-04-01）

- 运行命令：`python train_baseline.py --in data/features/features_v1.parquet --split-manifest data/processed/split_manifest.json --feature-manifest data/features/feature_manifest.json --out data/baseline/baseline_result.csv --metrics-out data/baseline/baseline_metrics.csv --report-out data/baseline/baseline_report.md`
- 目标标签：`label_excess_ret_5d`
- 预测样本：`8502` 行，覆盖 `448` 个交易日（val+test）
- Ridge(test)：RankIC=`0.008209`，ICIR=`0.030521`，Long-Short=`-0.004053`，命中率=`0.478405`
- Tree(test)：RankIC=`-0.013286`，ICIR=`-0.049881`，Long-Short=`-0.013051`，命中率=`0.452443`

### 阶段4.1稳健版更新（2026-04-01）

- 保留 4.0 脚本不变，新增 `train_baseline_v41.py`：
  - Ridge/Tree 支持独立窗口与重训频率
  - 动态融合（`blend`）
  - 时间分段诊断（季度/月份）
  - 可选行业感知（one-hot / 预测行业中性化）
- 新增产物：
  - `data/baseline/baseline_v41_result.csv`
  - `data/baseline/baseline_v41_metrics.csv`
  - `data/baseline/baseline_v41_period_metrics.csv`
  - `data/baseline/baseline_v41_report.md`
  - `report/阶段4_1_稳健版基线模型实验报告.md`
- 本次实跑（默认稳健配置）test 结果：
  - Ridge：RankIC=`0.008209`，ICIR=`0.030521`，Long-Short=`-0.004053`
  - Tree：RankIC=`0.030204`，ICIR=`0.111556`，Long-Short=`0.007594`
  - Blend：RankIC=`0.013315`，ICIR=`0.049641`，Long-Short=`0.003334`

## 阶段5交付

- 脚本：`train_lstm.py`
- 配置模板：`model_config.yaml`
- 预测结果：`data/dl/dl_result.csv`
- 指标汇总：`data/dl/dl_metrics.csv`
- 训练日志：`data/dl/dl_trainlog.csv`
- 实验报告：`data/dl/dl_report.md`
- 最新可推理权重：`data/dl/checkpoints/lstm_latest.pt`
- 最新权重元信息：`data/dl/checkpoints/lstm_latest_meta.json`
- 阶段总报告：`report/阶段5_LSTM链路打通实验报告.md`

### 本次实跑结果（2026-04-02）

- 运行命令：`python train_lstm.py --in data/features/features_v1.parquet --split-manifest data/processed/split_manifest.json --feature-manifest data/features/feature_manifest.json --out data/dl/dl_result.csv --metrics-out data/dl/dl_metrics.csv --trainlog-out data/dl/dl_trainlog.csv --report-out data/dl/dl_report.md`
- 目标标签：`label_excess_ret_5d`
- 时序窗口：`seq_len=20`
- 滚动重训：`retrain_every=40`，扩展窗口（`train_window=0`）
- 预测样本：`8483` 行，覆盖 `448` 个交易日（val+test）
- test（LSTM）：RankIC=`0.004885`，ICIR=`0.018072`，Long-Short=`-0.007630`，命中率=`0.459287`
- val（LSTM）：RankIC=`0.073435`，ICIR=`0.262638`，Long-Short=`0.005168`，命中率=`0.521196`
- 追加（2026-04-02）：补齐可推理模型落盘
  - 最新权重：`data/dl/checkpoints/lstm_latest.pt`
  - 元信息：`data/dl/checkpoints/lstm_latest_meta.json`

### 阶段5.1增强（2026-04-02）

- 代码增强（`train_lstm.py`）：
  - 新增 GPU 自动辨识与回退：`--device auto|cpu|cuda`（增强版）
  - 新增 GPU 筛选参数：`--gpu-index`、`--gpu-min-memory-gb`
  - 运行期增加设备探针日志（requested/selected/reason）
  - 报告新增“设备自动辨识”小节（记录 GPU 检测与回退原因）
- 新增参数模板：`model_config.yaml`（补充 stage5.1 推荐配置）
- 新增产物：
  - `data/dl/dl_v51_result.csv`
  - `data/dl/dl_v51_metrics.csv`
  - `data/dl/dl_v51_trainlog.csv`
  - `data/dl/dl_v51_report.md`
  - `report/阶段5_1_LSTM稳健增强与GPU自动辨识报告.md`
- 本次实跑命令：`python train_lstm.py --in data/features/features_v1.parquet --split-manifest data/processed/split_manifest.json --feature-manifest data/features/feature_manifest.json --device auto --gpu-min-memory-gb 4 --train-window 750 --retrain-every 20 --epochs 10 --inner-val-days 40 --out data/dl/dl_v51_result.csv --metrics-out data/dl/dl_v51_metrics.csv --trainlog-out data/dl/dl_v51_trainlog.csv --report-out data/dl/dl_v51_report.md`
- 设备识别结果：`torch_cuda_available=False`，自动回退 `cpu`
- test（LSTM v5.1）：RankIC=`-0.016379`，ICIR=`-0.060335`，Long-Short=`-0.007125`，命中率=`0.482889`

### 阶段5.2冲刺（2026-04-02）

- 新增脚本：`train_lstm_v52.py`（LSTM + anchor 动态模型选择器）
- 策略思路：
  - 候选模型：`pred_lstm`（阶段5）与 `pred_tree`（阶段4.1）
  - 每个交易日按“历史滚动IC”选择当日使用 LSTM 或 anchor
  - 采用标签可得性延迟：`label_delay_days=7`（避免过于乐观）
- 新增产物：
  - `data/dl/dl_v52_result.csv`
  - `data/dl/dl_v52_metrics.csv`
  - `data/dl/dl_v52_selector_log.csv`
  - `data/dl/dl_v52_sweep.csv`
  - `data/dl/dl_v52_report.md`
  - `report/阶段5_2_test指标冲刺与动态模型选择报告.md`
- 推理所需 LSTM 权重：`data/dl/checkpoints/lstm_latest.pt`（元信息：`data/dl/checkpoints/lstm_latest_meta.json`）
- 本次实跑命令：`python train_lstm_v52.py --lstm-result data/dl/dl_result.csv --anchor-result data/baseline/baseline_v41_result.csv --anchor-col pred_tree --lookback-days 8 --label-delay-days 7 --warmup-pick anchor --out data/dl/dl_v52_result.csv --metrics-out data/dl/dl_v52_metrics.csv --selector-log-out data/dl/dl_v52_selector_log.csv --sweep-out data/dl/dl_v52_sweep.csv --report-out data/dl/dl_v52_report.md`
- 指标对比（test，label_excess_ret_5d）：
  - 阶段5 LSTM：RankIC=`0.004885`，ICIR=`0.018072`，Long-Short=`-0.007630`
  - 阶段5.1 LSTM：RankIC=`-0.016379`，ICIR=`-0.060335`，Long-Short=`-0.007125`
  - 阶段5.2（stage5_2）：RankIC=`0.029718`，ICIR=`0.108555`，Long-Short=`0.001953`

## 说明

- `daily_panel.parquet` 采用（date, stock_code）唯一键。
- 默认使用并集交易日历补齐，`is_trading` 标识可交易行。
- `amount` 当前由 `close*volume` 近似；后续可切到原生成交额字段。
- 阶段3默认采用横截面 z-score 归一化，可通过 `--normalize none` 关闭。
