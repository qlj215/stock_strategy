# rule / dl 双后端可切换设计方案（已落地首版）

## 1. 设计目标

在不改前端主要接口的前提下，让概率预测支持三种模式：

- `rule`：规则模型（默认）
- `dl`：深度学习推理入口
- `auto`：优先 dl，失败自动回退 rule

这样可以做到：

1. 先稳定跑规则模型（当前可用）
2. 随时插入深度学习模型进行 A/B 对比
3. 深度学习出故障时不中断业务

---

## 2. 本次改造位置

### 新增核心模块

- `stock_strategy/probability_backend.py`

主要能力：

- `predict_probability(...)`：统一预测入口
- `get_backend_runtime_status(...)`：运行时后端状态
- `RuleProbabilityBackend`：内置规则模型
- `DLProbabilityBackend`：加载外部 DL 推理入口（`module:function`）

### 新增模板

- `stock_strategy/dl_entry_template.py`

提供标准函数签名 `predict_proba(daily_df)`，可直接替换成真实模型推理。

### trainer_app 侧改造

- `_probability_model` 改为统一代理 `predict_probability`
- 新接口：`/api/market/model_backend_status`
- 以下接口支持 `backend=rule|dl|auto` 参数：
  - `/api/market/overview`
  - `/api/market/scan`
  - `/api/market/backtest`
  - `/api/market/codex_reason`（POST body）

---

## 3. 切换机制

### 3.1 配置文件（推荐，WSL + Windows Python 更稳）

- 文件：`stock_strategy/config/model_backend.json`

```json
{
  "default_backend": "rule",
  "dl_entrypoint": "stock_strategy.dl_entry_template:predict_proba"
}
```

> 说明：你当前在 WSL 调 Windows Python，环境变量透传可能不稳定，推荐优先用配置文件。

### 3.2 环境变量（可选）

- `STOCK_PROB_BACKEND=rule|dl|auto`
- `STOCK_DL_ENTRYPOINT=python.module:function`

### 3.3 请求级覆盖（局部）

可在接口参数里传 `backend`，覆盖全局默认：

- `.../overview?backend=dl`
- `.../backtest?backend=rule`

---

## 4. 回退策略（关键稳定性）

当请求 `backend=dl` 但 DL 不可用（未配置/导入失败/推理异常）时：

- 自动回退到 `rule`
- 在响应里给出：
  - `backend_fallback: true`
  - `backend_error: ...`

因此即使 DL 还没完全就绪，系统也不会挂。

---

## 5. 输出协议（统一格式）

无论 rule 还是 dl，最终都归一到同一字段：

- `p_up_today`
- `p_up_5d`
- `p_up_long`
- `reasons`
- `features`
- `backend / backend_requested / backend_fallback / backend_error`

这保证前端和回测代码不需要区分具体后端实现。

---

## 6. 如何接入真实深度学习模型

你只需要实现一个函数并配置入口：

```python
def predict_proba(daily_df: pd.DataFrame) -> dict:
    # 1) 特征预处理
    # 2) 模型推理
    # 3) 返回统一输出字段
```

建议先从：

- 输入窗口：最近 60~120 天 OHLCV
- 输出：1日/5日/long 三个概率头（多任务）
- 保存格式：`torchscript` 或 `onnx`

---

## 7. 已验证结果（首版）

在当前环境下（未配置真实 DL 入口）：

- `backend=rule`：正常使用规则模型
- `backend=dl`：自动回退 rule，接口仍返回 200
- 状态接口可显示 dl 是否可用与错误原因

说明双后端开关链路已打通。

---

## 8. 下一步建议

1. 先把真实 DL 推理函数接到 `STOCK_DL_ENTRYPOINT`
2. 增加离线评测脚本：`rule vs dl` 同窗对比
3. 在 `/api/market/backtest` 增加对比输出（同一请求返回两套结果）
4. 再做灰度：线上默认 `rule`，部分请求 `dl`
