# Output 目录说明（研究型组织）

## 目录结构

- `research_YYYYMMDD_HHMMSS/`：一次完整研究实验输出
  - `config.json`：实验配置（时间区间、行业、股票、策略、资金、费率）
  - `metrics/all_results.csv`：全样本结果明细
  - `metrics/strategy_summary.csv`：按策略聚合统计
  - `metrics/sector_summary.csv`：按行业聚合统计
  - `metrics/top20.csv`：Top 20 组合
  - `trades/*.csv`：每个组合交易流水
  - `logs/run.log`：运行日志与失败原因
  - `report.md`：自动生成研究摘要

- `legacy_runs/`：历史单次测试与验证输出
  - `single_run_20260309/`
  - `test_runs_20260309/`

## 当前最新研究

- `research_20260309_114409`
