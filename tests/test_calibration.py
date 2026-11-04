"""Tests for the reliability curve, the slope test, and the walk-forward fit."""

import numpy as np
import pandas as pd
import pytest

from src.pmcal.calibration import (
    favourite_longshot_test, inv_logit, logit, reliability_bins,
    walk_forward_recalibration,
)


def test_logit_roundtrip():
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    assert inv_logit(logit(p)) == pytest.approx(p, abs=1e-9)


def test_reliability_bins_recover_known_frequencies():
    p = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)])
    y = np.concatenate([
        np.array([1] * 50 + [0] * 450),
        np.array([1] * 450 + [0] * 50),
    ])
    b = reliability_bins(p, y, n_bins=2)
    assert len(b) == 2
    assert b["realised_freq"].iloc[0] == pytest.approx(0.1)
    assert b["realised_freq"].iloc[1] == pytest.approx(0.9)
    assert b["gap"].abs().max() < 1e-9


def test_wilson_interval_stays_in_unit_range():
    p = np.full(200, 0.01)
    y = np.zeros(200, dtype=int)
    b = reliability_bins(p, y, n_bins=2)
    assert (b["wilson_lo"] >= 0).all()
    assert (b["wilson_hi"] <= 1).all()


def test_flb_slope_is_one_for_calibrated_market():
    """y ~ Bernoulli(p) is calibrated, so the logit slope must be 1."""
    rng = np.random.default_rng(42)
    p = inv_logit(rng.normal(0, 2.0, 20_000))
    y = rng.binomial(1, p)
    r = favourite_longshot_test(p, y)
    assert r.slope == pytest.approx(1.0, abs=0.08)
    assert not r.slope_differs_from_one


def test_flb_detects_injected_longshot_bias():
    """Flatten the true log-odds and the fitted slope must exceed 1."""
    rng = np.random.default_rng(43)
    z_true = rng.normal(0, 2.0, 20_000)
    y = rng.binomial(1, inv_logit(z_true))
    p_quoted = inv_logit(z_true * 0.5)  # quoted odds too close to even
    r = favourite_longshot_test(p_quoted, y)
    assert r.slope > 1.3
    assert r.slope_differs_from_one


def test_walk_forward_has_no_lookahead():
    """The recalibrated value at row i must not change when later rows change.

    This is the test that actually matters. If the fit peeked at the future,
    mutating a later outcome would move an earlier prediction.
    """
    rng = np.random.default_rng(99)
    n = 400
    df = pd.DataFrame({
        "market_id": np.arange(n),
        "price": rng.uniform(0.02, 0.95, n),
        "close_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
    })
    df["outcome"] = rng.binomial(1, df["price"])

    a = walk_forward_recalibration(df, min_train=100, refit_every=10)

    tampered = df.copy()
    tail = tampered.index[-50:]
    tampered.loc[tail, "outcome"] = 1 - tampered.loc[tail, "outcome"]
    b = walk_forward_recalibration(tampered, min_train=100, refit_every=10)

    head = slice(0, 300)
    np.testing.assert_allclose(
        a["p_recal"].to_numpy()[head],
        b["p_recal"].to_numpy()[head],
        equal_nan=True, rtol=1e-12,
    )


def test_walk_forward_ignores_outcomes_unknown_at_forecast_time():
    """A market that closed before row i can still be unresolved when i's price
    was observed `days_to_resolution` days earlier, so it must not train row i.
    """
    rng = np.random.default_rng(101)
    n, i, horizon = 200, 150, 7
    df = pd.DataFrame({
        "market_id": np.arange(n),
        "price": rng.uniform(0.02, 0.95, n),
        "close_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "days_to_resolution": horizon,
    })
    df["outcome"] = rng.binomial(1, df["price"])

    def recal_after_flipping(rows):
        t = df.copy()
        t.loc[rows, "outcome"] = 1 - t.loc[rows, "outcome"]
        return walk_forward_recalibration(t, min_train=50, refit_every=1)["p_recal"][i]

    base = recal_after_flipping([])
    assert recal_after_flipping(range(i - horizon, i)) == base
    assert recal_after_flipping(range(i - horizon - 20, i - horizon)) != base


def test_walk_forward_burn_in_is_nan():
    rng = np.random.default_rng(100)
    n = 300
    df = pd.DataFrame({
        "market_id": np.arange(n),
        "price": rng.uniform(0.05, 0.9, n),
        "close_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
    })
    df["outcome"] = rng.binomial(1, df["price"])
    out = walk_forward_recalibration(df, min_train=120, refit_every=20)
    assert out["p_recal"].iloc[:120].isna().all()
    assert out["p_recal"].iloc[150:].notna().any()


def test_walk_forward_is_ordered_by_time():
    rng = np.random.default_rng(101)
    n = 250
    times = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({
        "market_id": np.arange(n),
        "price": rng.uniform(0.05, 0.9, n),
        "close_time": rng.permutation(times),
    })
    df["outcome"] = rng.binomial(1, df["price"])
    out = walk_forward_recalibration(df, min_train=80)
    assert out["close_time"].is_monotonic_increasing
