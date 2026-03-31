# A股DL计划执行进度（独立分支）

- 分支：`feat/dl-plan-stage2`
- 原则：不改主线 `trainer_app.py` 业务逻辑；阶段化推进。

## 阶段状态

- [x] 阶段1：数据通路打通（已有 MiniQMT + xtdata）
- [x] 阶段2：研究数据集构建（本次完成首版）
- [ ] 阶段3：标签与特征工程
- [ ] 阶段4：基线模型实验
- [ ] 阶段5：深度学习模型第一版
- [ ] 阶段6：组合回测系统
- [ ] 阶段7：稳健性分析与最终总结

## 阶段2交付

- 脚本：`build_dataset.py`
- 数据：`data/processed/daily_panel.parquet`
- 切分：`data/processed/split_manifest.json`
- 字典：`data/processed/data_dictionary.md`

### 本次实跑结果（2026-03-31）

- 参数：`--start 20200101 --end 20260331 --limit 20`
- 输出行数：`30220`
- 股票数：`20`
- 交易日数：`1511`
- 训练/验证/测试：`1057 / 226 / 228` 日

## 说明

- `daily_panel.parquet` 采用（date, stock_code）唯一键。
- 默认使用并集交易日历补齐，`is_trading` 标识可交易行。
- `amount` 当前由 `close*volume` 近似；后续可切到原生成交额字段。
