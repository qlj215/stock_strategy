# data_dictionary（阶段2）

## 表：daily_panel.parquet

每行代表某只股票在某个交易日的一条观测（股票代码 + 日期 唯一）。

### 字段说明
- `date`：交易日期（datetime）
- `stock_code`：6位股票代码（string）
- `open`：开盘价（float）
- `high`：最高价（float）
- `low`：最低价（float）
- `close`：收盘价（float）
- `volume`：成交量（float）
- `amount`：成交额（float，当前由 close*volume 近似）
- `adj_factor`：复权因子（float，当前固定 1.0）
- `is_trading`：是否可交易（1=有交易，0=停牌/缺失补齐行）
- `industry`：行业名称（string，可空）

### 数据处理规则
1. 统一字段类型，日期转 datetime，价格量转数值。
2. 使用样本并集交易日历对每只股票补齐日期。
3. 补齐行（停牌/缺失）规则：
   - `is_trading=0`
   - `open/high/low/close` 保持缺失
   - `volume/amount` 置 0
4. 去重规则：`date + stock_code` 保留最后一条。

### 时间切分
见 `split_manifest.json`，按日期切分 train/val/test，避免未来信息泄漏。
