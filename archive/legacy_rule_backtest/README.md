# Legacy Rule Backtest Archive

本目录归档的是项目早期的传统技术指标策略回测体系。

归档原因：
- 当前主线网页端由 `trainer_app.py` 和 `web/market.html` 驱动；
- 当前研究支线由 `plans/a_stock_dl_project_plans/` 七阶段计划驱动；
- 本目录下代码不再属于上述两条当前维护链路，继续放在主目录会增加识别成本。

原始位置：
- `main.py`
- `research_runner.py`
- `backtest/`
- `strategies/`
- `visualization/`
- `test_quick.py`
- `test_integration.py`

内容说明：
- `main.py`：旧 CLI 单股传统策略回测入口；
- `research_runner.py`：旧多行业、多策略批量研究脚本；
- `strategies/`：MA、RSI、MACD、Bollinger 等传统指标信号；
- `backtest/`：旧信号驱动回测引擎、指标和参数优化；
- `visualization/`：旧回测图表绘制；
- `test_quick.py` / `test_integration.py`：旧体系测试脚本。

如需恢复使用，建议先确认其 MiniQMT 数据接口、导入路径和输出目录是否仍适配当前项目结构。
