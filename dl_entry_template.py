"""
DL 后端推理入口模板。

配置方式（Windows / WSL 都可）：
- STOCK_PROB_BACKEND=dl
- STOCK_DL_ENTRYPOINT=stock_strategy.dl_entry_template:predict_proba

实际接入时请把本文件中的示例逻辑替换成你的真实模型推理。
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def predict_proba(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """
    约定输入:
        daily_df: 至少包含 close, volume 列的日线 DataFrame

    约定输出:
        {
            "p_up_today": float(0~1),
            "p_up_5d": float(0~1),
            "p_up_long": float(0~1),
            "reasons": [str, ...],      # 可选
            "features": {k: v, ...},    # 可选
        }
    """

    # TODO: 在这里加载并调用你的深度学习模型（Torch/ONNX/TensorFlow 都行）
    # 这里先给一个占位输出，便于验证双后端切换链路。
    if daily_df is None or daily_df.empty:
        return {
            "p_up_today": 0.50,
            "p_up_5d": 0.50,
            "p_up_long": 0.50,
            "reasons": ["DL 模板：输入为空，返回中性概率"],
            "features": {},
        }

    return {
        "p_up_today": 0.52,
        "p_up_5d": 0.54,
        "p_up_long": 0.56,
        "reasons": ["DL 模板输出：请替换为真实模型推理结果"],
        "features": {"template_rows": int(len(daily_df))},
    }
