"""Tests for the trading simulation and the feature pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.pmcal.backtest import CostModel, divergence_signal, run_backtest
from src.pmcal.features import clean, engineer, feature_matrix, prepare


def _frame(n=300, seed=4):
    rng = np.random.default_rng(seed)
    price = rng.uniform(0.05, 0.9, n)
    return pd.DataFrame({
        "market_id": np.arange(n),
        "price": price,
        "outcome": rng.binomial(1, price),
        "days_to_resolution": rng.choice([1, 3, 7, 30], n),
        "close_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "volume": rng.lognormal(10, 2, n),
        "p_recal": np.clip(price + rng.normal(0, 0.05, n), 0.01, 0.99),
    })


def test_costs_total_adds_up():
    c = CostModel(half_spread=0.02, fee=0.01, slippage=0.005)
    assert c.total == pytest.approx(0.035)


def test_signal_only_fires_above_threshold():
    df = _frame()
    sig = divergence_signal(df, threshold=0.10)
    assert (sig.loc[sig["trade"], "edge"].abs() >= 0.10).all()
    assert (sig.loc[~sig["trade"], "edge"].abs() < 0.10).all()


def test_signal_side_matches_edge_direction():
    df = _frame()
    sig = divergence_signal(df, threshold=0.0)
    assert (sig.loc[sig["edge"] > 0, "side"] == "YES").all()
    assert (sig.loc[sig["edge"] <= 0, "side"] == "NO").all()


def test_higher_threshold_never_increases_trade_count():
    df = _frame()
    counts = [run_backtest(df, CostModel(), threshold=t, n_boot=50).n_trades
              for t in (0.01, 0.05, 0.10, 0.25)]
    assert counts == sorted(counts, reverse=True)


def test_costs_reduce_return():
    df = _frame()
    free = run_backtest(df, CostModel(0.0, 0.0, 0.0), threshold=0.03, n_boot=200)
    paid = run_backtest(df, CostModel(0.03, 0.01, 0.01), threshold=0.03, n_boot=200)
    assert paid.net_roi < free.net_roi


def test_no_trades_returns_safely():
    df = _frame()
    r = run_backtest(df, CostModel(), threshold=0.99, n_boot=50)
    assert r.n_trades == 0
    assert not r.significant


def test_backtest_interval_brackets_point_estimate():
    df = _frame()
    r = run_backtest(df, CostModel(), threshold=0.05, n_boot=500)
    if r.n_trades > 10:
        assert r.net_roi_lo <= r.net_roi <= r.net_roi_hi


def test_significance_flag_requires_positive_lower_bound():
    df = _frame()
    r = run_backtest(df, CostModel(), threshold=0.05, n_boot=500)
    assert r.significant == (r.net_roi_lo > 0)


# ----------------------------------------------------------- features

def test_clean_drops_degenerate_prices():
    df = pd.DataFrame({
        "market_id": [1, 2, 3, 4],
        "price": [0.0, 1.0, 0.5, 0.25],
        "outcome": [0, 1, 1, 0],
        "days_to_resolution": [7, 7, 7, 7],
        "close_time": pd.date_range("2024-01-01", periods=4, tz="UTC"),
    })
    out = clean(df)
    assert len(out) == 2
    assert out.attrs["cleaning_report"]["dropped_degenerate_price"] == 2


def test_clean_drops_duplicate_market_horizon():
    df = pd.DataFrame({
        "market_id": [1, 1, 2],
        "price": [0.4, 0.4, 0.6],
        "outcome": [1, 1, 0],
        "days_to_resolution": [7, 7, 7],
        "close_time": pd.date_range("2024-01-01", periods=3, tz="UTC"),
    })
    out = clean(df)
    assert len(out) == 2
    assert out.attrs["cleaning_report"]["dropped_duplicate_market_horizon"] == 1


def test_engineer_creates_expected_columns():
    out = engineer(clean(_frame().drop(columns=["p_recal"])))
    for col in ("logit_price", "log_volume", "is_longshot",
                "is_favourite", "month_index", "price_x_horizon"):
        assert col in out.columns
    assert out["logit_price"].notna().all()


def test_logit_price_is_monotone_in_price():
    out = engineer(clean(_frame().drop(columns=["p_recal"])))
    s = out.sort_values("price")
    assert s["logit_price"].is_monotonic_increasing


def test_feature_matrix_groups_are_market_ids():
    df = prepare(_frame().drop(columns=["p_recal"]))
    X, y, groups = feature_matrix(df)
    assert X.shape[0] == len(y) == len(groups)
    np.testing.assert_array_equal(groups, df["market_id"].to_numpy())


def test_feature_matrix_rejects_unprepared_frame():
    with pytest.raises(ValueError):
        feature_matrix(_frame().drop(columns=["p_recal"]))
