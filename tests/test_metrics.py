"""Known-answer tests for the scoring rules.

Each test here has an answer that can be worked out on paper or that follows
from a generator with a known truth. That is the point: if the statistics are
wrong, these fail, and no amount of plausible-looking output on real data
would have told us.
"""

import numpy as np
import pytest

from src.pmcal.metrics import (
    brier_score, brier_skill_score, cluster_bootstrap,
    expected_calibration_error, log_loss, murphy_decomposition,
)


def test_brier_perfect_forecast_is_zero():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1, 0, 1, 0])
    assert brier_score(p, y) == pytest.approx(0.0)


def test_brier_coin_flip_is_quarter():
    p = np.full(100, 0.5)
    y = np.array([0, 1] * 50)
    assert brier_score(p, y) == pytest.approx(0.25)


def test_brier_worst_case_is_one():
    p = np.array([0.0, 1.0])
    y = np.array([1, 0])
    assert brier_score(p, y) == pytest.approx(1.0)


def test_brier_hand_computed():
    # ((0.8-1)^2 + (0.3-0)^2 + (0.6-1)^2) / 3 = (0.04 + 0.09 + 0.16)/3
    p = np.array([0.8, 0.3, 0.6])
    y = np.array([1, 0, 1])
    assert brier_score(p, y) == pytest.approx((0.04 + 0.09 + 0.16) / 3)


def test_log_loss_matches_hand_computation():
    p = np.array([0.9, 0.2])
    y = np.array([1, 0])
    expected = -(np.log(0.9) + np.log(0.8)) / 2
    assert log_loss(p, y) == pytest.approx(expected, rel=1e-6)


def test_bss_is_zero_for_base_rate_forecast():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=5000)
    p = np.full(len(y), y.mean())
    assert brier_skill_score(p, y) == pytest.approx(0.0, abs=1e-12)


def test_bss_is_one_for_perfect_forecast():
    y = np.array([1, 0, 1, 1, 0])
    p = y.astype(float)
    assert brier_skill_score(p, y) == pytest.approx(1.0)


def test_bss_negative_for_anti_correlated_forecast():
    y = np.array([1, 0, 1, 0] * 25)
    p = 1.0 - y
    assert brier_skill_score(p, y) < 0


def test_murphy_identity_holds_on_random_data():
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1, 4000)
    y = rng.binomial(1, p)
    d = murphy_decomposition(p, y, n_bins=10)
    # The identity is exact for the BINNED forecast, not the raw one.
    assert d.residual == pytest.approx(0.0, abs=1e-9)
    assert d.brier_binned == pytest.approx(
        d.reliability - d.resolution + d.uncertainty, abs=1e-9)
    # Binning discards within-bin information, so the BINNED score is the
    # worse of the two and binning_error (raw - binned) is negative.
    assert d.binning_error < 0
    assert d.brier_binned > d.brier
    assert d.brier == pytest.approx(d.brier_binned + d.binning_error, abs=1e-12)


def test_murphy_reliability_near_zero_for_calibrated_forecaster():
    """A forecaster drawing y ~ Bernoulli(p) is calibrated by construction."""
    rng = np.random.default_rng(11)
    p = rng.uniform(0.05, 0.95, 40_000)
    y = rng.binomial(1, p)
    d = murphy_decomposition(p, y, n_bins=10)
    assert d.reliability < 0.002


def test_murphy_reliability_detects_injected_miscalibration():
    """Shrink every forecast toward 0 and reliability must rise."""
    rng = np.random.default_rng(13)
    p = rng.uniform(0.05, 0.95, 40_000)
    y = rng.binomial(1, p)
    honest = murphy_decomposition(p, y, n_bins=10)
    biased = murphy_decomposition(p * 0.6, y, n_bins=10)
    assert biased.reliability > honest.reliability * 10


def test_murphy_resolution_zero_for_constant_forecast():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.4, 2000)
    p = np.full(len(y), 0.4)
    d = murphy_decomposition(p, y, n_bins=10)
    assert d.resolution == pytest.approx(0.0, abs=1e-12)


def test_ece_zero_for_perfectly_calibrated_bins():
    p = np.concatenate([np.full(1000, 0.2), np.full(1000, 0.8)])
    y = np.concatenate([
        np.array([1] * 200 + [0] * 800),
        np.array([1] * 800 + [0] * 200),
    ])
    ece, mce = expected_calibration_error(p, y, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert mce == pytest.approx(0.0, abs=1e-9)


def test_mce_catches_single_bad_bin_that_ece_hides():
    # 99% of mass perfectly calibrated, 1% wildly wrong.
    p = np.concatenate([np.full(9900, 0.5), np.full(100, 0.95)])
    y = np.concatenate([
        np.array([1] * 4950 + [0] * 4950),
        np.zeros(100, dtype=int),
    ])
    ece, mce = expected_calibration_error(p, y, n_bins=2)
    assert ece < 0.02
    assert mce > 0.9


def test_cluster_bootstrap_interval_contains_truth():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.1, 0.9, 3000)
    y = rng.binomial(1, p)
    point, lo, hi, kept = cluster_bootstrap(brier_score, p, y, n_boot=500)
    assert lo < point < hi
    assert kept == 500


def test_clustered_intervals_are_wider_than_naive_ones():
    """The whole reason the project clusters: ignoring it understates width.

    Build 200 markets, each contributing 4 identical-outcome rows. Resampling
    rows pretends there are 800 independent observations; resampling markets
    knows there are 200.
    """
    rng = np.random.default_rng(17)
    n_markets = 200
    market_p = rng.uniform(0.1, 0.9, n_markets)
    market_y = rng.binomial(1, market_p)

    p = np.repeat(market_p, 4) + rng.normal(0, 0.01, n_markets * 4)
    p = np.clip(p, 0.01, 0.99)
    y = np.repeat(market_y, 4)
    clusters = np.repeat(np.arange(n_markets), 4)

    _, nlo, nhi, _ = cluster_bootstrap(brier_score, p, y, clusters=None,
                                       n_boot=800, seed=1)
    _, clo, chi, _ = cluster_bootstrap(brier_score, p, y, clusters=clusters,
                                       n_boot=800, seed=1)
    assert (chi - clo) > (nhi - nlo) * 1.5


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        brier_score([0.5, 1.2], [0, 1])
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [0, 2])
    with pytest.raises(ValueError):
        brier_score([0.5], [0, 1])
    with pytest.raises(ValueError):
        brier_score([], [])
