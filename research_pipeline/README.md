# Research Pipeline

本目录存放七阶段 A 股深度学习研究支线脚本。

运行约定：
- 从项目根目录运行脚本；
- 默认输入输出路径仍指向根目录下的 `data/`、`report/` 等目录；
- 示例：`python research_pipeline/train_baseline.py`。

阶段对应：
- 阶段2：`research_pipeline/build_dataset.py`
- 阶段3：`research_pipeline/feature_engineering.py`
- 阶段4：`research_pipeline/train_baseline.py`
- 阶段4.1：`research_pipeline/train_baseline_v41.py`
- 阶段5 / 5.1：`research_pipeline/train_lstm.py`
- 阶段5.2：`research_pipeline/train_lstm_v52.py`
- 阶段6.0：`research_pipeline/portfolio_backtest.py`

主线网页端仍由项目根目录下的 `trainer_app.py` 驱动；独立收益试算工具保留在 `portfolio_whatif_web.py`。
