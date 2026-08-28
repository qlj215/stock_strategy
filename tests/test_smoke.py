#!/usr/bin/env python
"""
冒烟回归测试：覆盖概率后端、trainer_app 回测链路、组合试算与研究管线核心函数。

运行（项目根目录，任一装有 pandas/numpy/flask 的环境）：
    python stock_strategy/tests/test_smoke.py

不依赖 xtquant/torch/网络：
- fetcher/事件等外部依赖一律用猴子补丁替换；
- 历史面板用临时 parquet 构造（无 pyarrow 时自动跳过相关用例）。

输出：逐项 PASS/FAIL；任何 FAIL 都以非零码退出。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

RESULTS = []


def check(name: str, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"[PASS] {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


def _synthetic_daily(n: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.02, n))
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=n),
        "open": close * (1 + rng.normal(0, 0.004, n)),
        "high": close * (1 + rng.normal(0.006, 0.004, n)),
        "low": close * (1 - rng.normal(0.006, 0.004, n)).clip(min=0.001),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


def _synthetic_panel(symbols_with_meta: list[tuple[str, str]], days: int = 260, seed: int = 11) -> pd.DataFrame:
    """构造带行业/停牌/涨停特征的历史面板（列结构同 daily_panel.parquet）。"""
    dates = pd.bdate_range("2024-01-02", periods=days)
    rng = np.random.default_rng(seed)
    frames = []
    for i, (code, industry) in enumerate(symbols_with_meta):
        close = 50.0 * np.cumprod(1.0 + rng.normal(0.0004 + 0.0001 * i, 0.018, days))
        open_ = close * (1 + rng.normal(0, 0.004, days))
        high = np.maximum(close, open_) * (1 + rng.normal(0.004, 0.003, days))
        low = np.minimum(close, open_) * (1 - rng.normal(0.004, 0.003, days)).clip(min=0.001)
        volume = rng.integers(800_000, 6_000_000, days).astype(float)
        one = pd.DataFrame({
            "date": dates,
            "stock_code": code,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "is_trading": 1,
            "industry": industry,
        })
        # 少量停牌行，制造 is_trading=0 且价格为空的补齐形态
        suspend_idx = rng.choice(days, size=4, replace=False)
        for c in ["open", "high", "low", "close"]:
            one.loc[one.index[suspend_idx], c] = np.nan
        one["is_trading"] = 0
        one.loc[one["close"].notna(), "is_trading"] = 1
        frames.append(one)
    return pd.concat(frames, ignore_index=True)


PANEL_META = [
    ("000001", "银行"), ("600036", "银行"), ("600519", "白酒"),
    ("000858", "白酒"), ("300750", "新能源"), ("002594", "新能源"),
    ("600276", "医药"), ("603986", "半导体"),
]


def test_rule_backend_predict():
    from stock_strategy.probability_backend import RuleProbabilityBackend

    df = _synthetic_daily()
    out = RuleProbabilityBackend("rule_basic").predict(df)
    for k in ["p_up_today", "p_up_5d", "p_up_long"]:
        assert 0.05 <= out[k] <= 0.95, f"{k} 越界: {out[k]}"
    assert out["features"]["prob_model"] == "rule_basic"
    assert len(out["reasons"]) >= 4

    out2 = RuleProbabilityBackend("rule_no_chase").predict(df)
    assert out2["features"]["prob_model"] == "rule_no_chase"

    short = df.head(20)
    neutral = RuleProbabilityBackend().predict(short)
    assert neutral["p_up_today"] == 0.5 and "中性" in neutral["reasons"][0]


def test_backend_switch_and_meta():
    from stock_strategy import probability_backend as pb

    df = _synthetic_daily()
    out = pb.predict_probability(df, backend="rule", prob_model="rule_no_chase")
    assert out["backend"] == "rule"
    assert out["backend_requested"] == "rule"
    assert out["prob_model"] == "rule_no_chase"
    assert out["prob_model_label"] == "低追高倾向模型"
    assert out["backend_fallback"] is False

    batch = pb.predict_probability_batch([df, df.head(10)], backend="rule")
    assert len(batch) == 2 and batch[1]["p_up_today"] == 0.5

    status = pb.get_backend_runtime_status("auto")
    assert status["requested"] == "auto"
    assert status["selected_with_fallback"] in {"rule", "dl"}


def test_symbol_helpers():
    from stock_strategy.trainer_app import (
        _board_filter_matches,
        _classify_symbol_board,
        _normalize_plain_symbol,
        _parse_manual_symbols,
        _symbol_limit_up_rate,
    )

    assert _normalize_plain_symbol("600519.SH") == "600519"
    assert _normalize_plain_symbol("sz000001") == "000001"
    assert _normalize_plain_symbol("1") == "000001"
    assert _normalize_plain_symbol("abc") == ""

    assert _classify_symbol_board("300750") == "创业板"
    assert _classify_symbol_board("688981") == "科创板"
    assert _classify_symbol_board("600519") == "主板"
    assert _classify_symbol_board("830001") == "北交所"

    assert _board_filter_matches("300750", "gem_only")
    assert not _board_filter_matches("600519", "gem_only")
    assert _board_filter_matches("600519", "exclude_gem")

    assert _symbol_limit_up_rate("300750") == 0.20
    assert _symbol_limit_up_rate("600519") == 0.10

    syms, bad = _parse_manual_symbols("600519, 000001；SZ300750；xyz")
    assert syms == ["600519", "000001", "300750"]
    assert bad == ["xyz"]


def test_scan_anchor_and_cost_config():
    from stock_strategy.trainer_app import _parse_backtest_cost_config, _parse_scan_anchor

    ts, meta = _parse_scan_anchor("2024-06-03", "10:31")
    assert meta["is_custom"] is True
    assert meta["effective_date"] == "2024-06-03"
    assert meta["effective_time"] == "10:31"

    ts2, meta2 = _parse_scan_anchor("", "")
    assert meta2["is_custom"] is False

    try:
        _parse_scan_anchor("bad-date", "")
        raise AssertionError("坏日期应当抛错")
    except ValueError:
        pass

    cfg = _parse_backtest_cost_config({})
    assert cfg["preset"] == "huatai_a_share_default"
    cfg2 = _parse_backtest_cost_config({"buy_commission_rate": "0.0005"})
    assert abs(cfg2["buy_commission_rate"] - 0.0005) < 1e-12 and cfg2["preset"] == "custom"
    cfg3 = _parse_backtest_cost_config({"buy_commission_rate": "99"})
    assert cfg3["buy_commission_rate"] == 0.05  # 上限钳制


def test_rule_signals_on_history():
    from stock_strategy.trainer_app import (
        _generate_realtime_signal_marks,
        _prepare_rule_indicator_frame,
        _rule_signal_breakout_20d_high,
        _rule_signal_limit_up_follow,
    )

    df = _synthetic_daily(120)
    df["symbol"] = "000001"
    marks = _generate_realtime_signal_marks(df, "breakout_20d_high")
    assert marks["preset"] == "breakout_20d_high"
    assert isinstance(marks["signal_count"], int) and marks["signal_count"] >= 0

    prepared = _prepare_rule_indicator_frame(df)
    assert {"ret_1d", "ma5", "ma20", "high20_prev"} <= set(prepared.columns)
    _ = _rule_signal_breakout_20d_high(prepared.iloc[-1])
    _ = _rule_signal_limit_up_follow(prepared.iloc[-1])

    try:
        _generate_realtime_signal_marks(df, "no_such_preset")
        raise AssertionError("未知 preset 应当抛错")
    except ValueError:
        pass


def _make_panel_parquet(tmpdir: str) -> str:
    path = os.path.join(tmpdir, "daily_panel_smoke.parquet")
    panel = _synthetic_panel(PANEL_META)
    panel.to_parquet(path, index=False)
    return path


def _patched_panel_app(panel_path: str):
    """导入 trainer_app 并把历史面板指向合成 parquet，屏蔽实时板块探测。"""
    import stock_strategy.trainer_app as ta

    ta._history_panel_path = lambda: panel_path
    ta._load_history_panel(force_refresh=True)
    ta.get_sector_sync_status = lambda *a, **k: {
        "base_sector_count": 0,
        "dynamic_sector_available": False,
        "dynamic_sector_rows": 0,
        "dynamic_industry_count": 0,
        "error": "smoke-test: miniqmt unavailable",
    }
    ta.list_dynamic_industry_sectors = lambda *a, **k: []
    ta.get_symbol_name = lambda *a, **k: "合成测试股"
    return ta


def test_panel_backtest_rule_strategy():
    pyarrow_ok = True
    try:
        import pyarrow  # noqa: F401
    except Exception:
        pyarrow_ok = False
    if not pyarrow_ok:
        print("    (skip: 无 pyarrow，跳过面板回测用例)")
        return

    import stock_strategy.trainer_app as ta

    with tempfile.TemporaryDirectory() as tmpdir:
        ta = _patched_panel_app(_make_panel_parquet(tmpdir))
        client = ta.app.test_client()

        resp = client.get("/api/market/backtest").get_json()
        assert isinstance(resp, dict) and resp.get("error"), "空参应报手动股票池为空"

        resp = client.get(
            "/api/market/backtest"
            "?universe_mode=industry&industry=银行&strategy_category=rule"
            "&rule_preset=ma5_cross_ma20&data_source=panel"
            "&start=20240201&end=20241231&hold_days=3"
        )
        assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
        data = resp.get_json()
        assert data["mode"] == "portfolio_v2"
        assert data["universe"]["resolved_symbol_count"] == 2
        assert data["summary"]["trade_count"] >= 1, "随机游走面板应产生信号"
        assert len(data["equity_curve"]) >= 50
        assert {"strategy", "benchmark"} <= set(data["equity_curve"][0].keys())
        assert data["request"]["transaction_costs"]["preset"] in {"huatai_a_share_default", "custom"}

        # 成本参数应影响净值
        resp2 = client.get(
            "/api/market/backtest"
            "?universe_mode=industry&industry=银行&strategy_category=rule"
            "&rule_preset=ma5_cross_ma20&data_source=panel"
            "&start=20240201&end=20241231&hold_days=3"
            "&buy_commission_rate=0.01&sell_commission_rate=0.01"
        )
        d2 = resp2.get_json()
        assert d2["request"]["transaction_costs"]["preset"] == "custom"
        assert d2["summary"]["total_cost_amount"] > data["summary"]["total_cost_amount"]

        # 手动池 + 板块过滤（board_filter 不作用于 manual 模式，应保持全部 3 只）
        resp3 = client.get(
            "/api/market/backtest"
            "?universe_mode=manual&symbols=300750,600519,600276&strategy_category=rule"
            "&rule_preset=breakout_20d_high&data_source=panel"
            "&start=20240201&end=20241231&hold_days=2&board_filter=gem_only"
        )
        d3 = resp3.get_json()
        assert resp3.status_code == 200, str(d3)[:300]
        assert d3["universe"]["resolved_symbol_count"] == 3

        # 非法参数
        bad = client.get(
            "/api/market/backtest?universe_mode=nope&symbols=000001"
        )
        assert bad.status_code == 400


def test_panel_backtest_model_strategy():
    try:
        import pyarrow  # noqa: F401
    except Exception:
        print("    (skip: 无 pyarrow，跳过面板回测用例)")
        return

    import stock_strategy.trainer_app as ta

    with tempfile.TemporaryDirectory() as tmpdir:
        ta = _patched_panel_app(_make_panel_parquet(tmpdir))
        client = ta.app.test_client()
        resp = client.get(
            "/api/market/backtest"
            "?universe_mode=manual&symbols=000001,600519,300750,600276&strategy_category=model"
            "&model_preset=topk_prob_1d&model_backend=rule&data_source=panel"
            "&start=20240301&end=20241231&hold_days=5&top_k=2&min_history=60"
        )
        assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
        data = resp.get_json()
        assert data["strategy"]["category"] == "model"
        assert data["strategy"]["backend_requested"] == "rule"
        assert data["universe"]["resolved_symbol_count"] == 4
        assert data["summary"]["trade_count"] >= 1
        # rule 后端不回退
        assert data["strategy"]["backend_status"]["dl_available"] in {True, False}


def test_strategy_metrics_and_calibration():
    import stock_strategy.trainer_app as ta

    df = _synthetic_daily(120).set_index("date")
    prob = pd.Series(np.linspace(0.2, 0.9, len(df)), index=df.index)
    m = ta._strategy_metrics(df, prob, horizon=5, threshold=0.6)
    assert {"strategy_total_return", "buyhold_total_return", "trade_count", "points"} <= set(m)
    assert m["trade_count"] >= 1

    empty = ta._strategy_metrics(df.iloc[:5], prob.iloc[:5], horizon=5)
    assert empty["trade_count"] == 0

    y = np.array([1, 0, 1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.6, 0.4, 0.3])
    cm = ta._classification_metrics(y, p, threshold=0.5)
    assert cm["confusion"] == {"tp": 3, "tn": 3, "fp": 0, "fn": 0}
    bins = ta._calibration_bins(y, p, bins=10)
    assert bins and sum(b["count"] for b in bins) == len(y)


def test_legacy_backtest_branch():
    import stock_strategy.trainer_app as ta

    df = _synthetic_daily(300)
    ta._daily_from_fetcher = lambda symbol, start, end, **k: df

    args = {"symbol": "000001", "threshold": "0.5", "long_horizon": "40", "backend": ""}
    assert ta._is_legacy_backtest_request(args) is True
    assert ta._is_legacy_backtest_request({"symbol": "000001", "universe_mode": "manual"}) is False
    assert ta._is_legacy_backtest_request({}) is False

    with ta.app.test_request_context("/api/market/backtest?symbol=000001&start=20230101&end=20240101&threshold=0.5&long_horizon=40"):
        # 触发 legacy 分支的完整链路（分类指标 + 策略回测 + 校准分箱）
        ta._daily_from_fetcher = lambda symbol, start, end, **k: df
        try:
            resp = ta._market_backtest_legacy()
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"legacy 回测链路异常: {e}")
        payload = resp[0].get_json() if isinstance(resp, tuple) else resp.get_json()
    assert payload.get("legacy_mode") is True
    assert set(payload["classification"]) == {"d1", "d5", "long"}
    assert payload["window"]["samples"] > 0


def test_scan_core_offline():
    try:
        import pyarrow  # noqa: F401
    except Exception:
        print("    (skip: 无 pyarrow，跳过扫描用例)")
        return

    import stock_strategy.trainer_app as ta

    with tempfile.TemporaryDirectory() as tmpdir:
        ta = _patched_panel_app(_make_panel_parquet(tmpdir))
        # 无行情源：让逐股快照拉取直接失败，避免测试真实触达 MiniQMT
        ta._daily_from_fetcher = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("smoke-test: no market data"))
        result = ta._run_market_scan_core(
            mode="industry", industry="银行", sort_by="next_5d_up", board_filter="all",
            days=60, limit=5, backend="rule", prob_model=None,
            candidate_mode="full_order", random_seed=42,
        )
        # 无行情源：全部失败但流程完整（银行走本地兜底池）
        assert result["requested"] == 2 and result["success"] == 0 and result["failed"] == 2
        assert result["sector_source"] == "local_fallback"
        assert result["model_backend"]["requested"] == "rule"

        # 行业模式缺参数
        try:
            ta._run_market_scan_core(
                mode="industry", industry="", sort_by="next_5d_up", board_filter="all",
                days=60, limit=5, backend="rule", prob_model=None,
                candidate_mode="full_order", random_seed=42,
            )
            raise AssertionError("industry 为空应当抛错")
        except ValueError:
            pass


def test_whatif_calculator():
    import stock_strategy.portfolio_whatif_web as pw

    df = _synthetic_daily(60).set_index("date")

    pw.fetch_stock_data = lambda symbol, start_date, end_date, **k: df
    pw.get_symbol_name = lambda symbol: "合成测试股"

    payload_amount = {
        "position_mode": "amount", "time_mode": "day",
        "buy_time": "2024-03-04", "sell_time": "2024-04-01",
        "positions": [{"symbol": "600519", "value": "50000"}],
    }
    out = pw.calculate_portfolio(payload_amount)
    d = out["details"][0]
    assert d["lots"] >= 1 and d["shares"] == d["lots"] * 100
    assert out["summary"]["total_profit"] == d["profit"]

    payload_lots = dict(payload_amount, position_mode="lots", positions=[{"symbol": "000001", "value": "3"}])
    out2 = pw.calculate_portfolio(payload_lots)
    assert out2["details"][0]["shares"] == 300

    # 卖出早于买入
    try:
        pw.calculate_portfolio(dict(payload_amount, sell_time="2024-02-01"))
        raise AssertionError("卖出早于买入应当抛错")
    except ValueError:
        pass

    # 预算不足一手
    try:
        pw.calculate_portfolio({"position_mode": "amount", "time_mode": "day",
                                "buy_time": "2024-03-04", "sell_time": "2024-04-01",
                                "positions": [{"symbol": "600519", "value": "10"}]})
        raise AssertionError("预算不足应当抛错")
    except ValueError:
        pass


def test_feature_engineering():
    fe = _load_pipeline_module("fe_mod", "feature_engineering.py")

    panel = _synthetic_panel(PANEL_META[:4], days=140)
    feat = fe.add_features(panel, industry_feature_mode="use")
    assert all(c in feat.columns for c in fe.FEATURE_COLS)
    feat2, label_cols = fe.add_labels(feat, horizons=[5], cls_quantile=0.3)
    assert "label_fwd_ret_5d" in feat2.columns and len(label_cols) == 5
    feat3 = fe.apply_normalization(feat2, fe.FEATURE_COLS, mode="xsec_zscore")
    col = feat3[fe.FEATURE_COLS[0]].dropna()
    assert abs(col.mean()) < 1e-6, "横截面 z-score 后全表均值应接近 0"


def _load_pipeline_module(mod_name: str, filename: str):
    spec_path = os.path.join(PROJECT_ROOT, "stock_strategy", "research_pipeline", filename)
    spec = importlib.util.spec_from_file_location(mod_name, spec_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # dataclass 处理需要模块已注册
    spec.loader.exec_module(module)
    return module


def test_stage6_backtest():
    pb = _load_pipeline_module("pb_mod", "portfolio_backtest.py")

    panel = _synthetic_panel(PANEL_META, days=120)
    rng = np.random.default_rng(3)
    df = panel.copy()
    df["score"] = rng.normal(size=len(df))
    df["label_fwd_ret_5d"] = rng.normal(0.001, 0.02, len(df))
    df["label_bench_ret_5d"] = rng.normal(0.0005, 0.01, len(df))
    df["label_excess_ret_5d"] = df["label_fwd_ret_5d"] - df["label_bench_ret_5d"]

    nav, detail, group, metrics = pb.run_stage6_backtest(
        df, alias="t", label_col="label_fwd_ret_5d", bench_col="label_bench_ret_5d",
        rebalance_every=5, top_n=3, top_quantile=0.2, group_count=5, init_nav=1.0,
        buy_cost_rate=0.0008, sell_cost_rate=0.0018, min_candidates=5,
    )
    assert not nav.empty and metrics["periods"] == len(nav)
    assert metrics["net_total_return"] <= metrics["gross_total_return"], "扣费后收益不应高于毛收益"
    assert not detail.empty and not group.empty


def test_train_baseline_walkforward():
    tb = _load_pipeline_module("tb_mod", "train_baseline.py")

    rng = np.random.default_rng(5)
    days = pd.bdate_range("2024-01-02", periods=160)
    frames = []
    for code in ["000001", "600519", "300750", "600276"]:  # 每日 >=3 只才能算 RankIC
        f = pd.DataFrame({"date": days, "stock_code": code})
        for c in ["f1", "f2", "f3"]:
            f[c] = rng.normal(size=len(days))
        f["target"] = f["f1"] * 0.5 + rng.normal(0, 0.1, len(days))
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["split"] = np.where(df["date"] <= days[100], "train", np.where(df["date"] <= days[130], "val", "test"))

    pred = tb.walk_forward_predict(
        df, feature_cols=["f1", "f2", "f3"], target_col="target",
        train_end=days[100], retrain_every=10, train_window=0, min_train_days=20,
        model_factory=lambda: tb.RidgeClosedForm(alpha=1.0), pred_col_name="pred_ridge",
    )
    assert not pred.empty and "pred_ridge" in pred.columns
    assert (pred["date"] > days[100]).all(), "walk-forward 只应在 train_end 之后输出预测"

    m = tb.calc_metrics(pred, pred_col="pred_ridge", target_col="target", n_quantiles=3)
    assert m["rows"] == len(pred) and np.isfinite(m["rank_ic_mean"])

    tree = tb.SimpleTreeRegressor(max_depth=2, min_samples_split=20, min_samples_leaf=10)
    x = df[["f1", "f2", "f3"]].to_numpy()
    tree.fit(x, df["target"].to_numpy())
    assert tree.predict(x).shape == (len(df),)


def main() -> int:
    check("rule_backend_predict", test_rule_backend_predict)
    check("backend_switch_and_meta", test_backend_switch_and_meta)
    check("symbol_helpers", test_symbol_helpers)
    check("scan_anchor_and_cost_config", test_scan_anchor_and_cost_config)
    check("rule_signals_on_history", test_rule_signals_on_history)
    check("strategy_metrics_and_calibration", test_strategy_metrics_and_calibration)
    check("panel_backtest_rule_strategy", test_panel_backtest_rule_strategy)
    check("panel_backtest_model_strategy", test_panel_backtest_model_strategy)
    check("legacy_backtest_branch", test_legacy_backtest_branch)
    check("scan_core_offline", test_scan_core_offline)
    check("whatif_calculator", test_whatif_calculator)
    check("feature_engineering", test_feature_engineering)
    check("stage6_backtest", test_stage6_backtest)
    check("train_baseline_walkforward", test_train_baseline_walkforward)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n==== {len(RESULTS) - len(failed)}/{len(RESULTS)} passed ====")
    if failed:
        for name, _, err in failed:
            print(f"  FAILED: {name} -> {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
