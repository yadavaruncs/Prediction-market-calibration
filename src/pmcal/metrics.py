"""Proper scoring rules and their decompositions.

The scores here are all *proper*: they are minimised in expectation only by
reporting your true belief. That matters for this project, because the whole
question is whether market prices are honest probabilities, and a scoring rule
that could be gamed would not answer it.

Everything takes (p, y) where p is a forecast probability in [0, 1] and y is a
binary outcome in {0, 1}.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# log loss blows up at exactly 0 or 1, and market prices do touch both ends
# of the book. 1e-6 is well below the 0.001 tick we actually observe, so it
# clips nothing real.
_EPS = 1e-6


def _validate(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: p{p.shape} vs y{y.shape}")
    if p.size == 0:
        raise ValueError("empty input")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities outside [0, 1]")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("outcomes must be 0 or 1")
    return p, y


def brier_score(p, y) -> float:
    """Mean squared error of the forecast. Lower is better; 0.25 is a coin flip."""
    p, y = _validate(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    """Negative log likelihood per observation. Punishes confident mistakes hard."""
    p, y = _validate(p, y)
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def base_rate(y) -> float:
    return float(np.mean(np.asarray(y, dtype=float)))


def brier_skill_score(p, y, reference: float | None = None) -> float:
    """Brier relative to a constant forecast at the base rate.

    0 means 'no better than always predicting the base rate'. 1 is perfection.
    Negative means the forecasts are actively worse than knowing nothing but
    how often the event happens.

    Note the reference is the *in-sample* base rate by default, which is a
    generous benchmark to beat -- it already knows something a real forecaster
    would not. Pass `reference` explicitly to score against a prior base rate.
    """
    p, y = _validate(p, y)
    ref = base_rate(y) if reference is None else float(reference)
    bs_ref = float(np.mean((ref - y) ** 2))
    if bs_ref == 0:
        return float("nan")
    return float(1 - brier_score(p, y) / bs_ref)


@dataclass(frozen=True)
class MurphyDecomposition:
    """Brier = reliability - resolution + uncertainty.

    reliability  how far each bin's realised frequency sits from its forecast.
                 Lower is better; 0 is perfect calibration.
    resolution   how far the bins' realised frequencies spread away from the
                 base rate. Higher is better; it is the discrimination the
                 forecast actually adds.
    uncertainty  variance of the outcome itself, p(1-p) at the base rate.
                 Nothing the forecaster does can change it.

    Calibration and discrimination are different virtues, and a single Brier
    number hides which one a forecaster has. A broken clock that always says
    'base rate' is perfectly calibrated and has zero resolution.
    """

    brier: float          # on the raw forecasts
    brier_binned: float   # on bin-mean forecasts; the identity applies to this
    # brier - brier_binned. Normally NEGATIVE: replacing each forecast with
    # its bin mean discards within-bin information, so the binned score is
    # worse than the raw one. A positive value would mean the raw forecasts
    # were so badly ordered that coarsening them helped.
    binning_error: float
    reliability: float
    resolution: float
    uncertainty: float
    n_bins_used: int
    residual: float  # brier_binned - (rel - res + unc); must be ~0

    def as_dict(self) -> dict:
        return asdict(self)


def murphy_decomposition(p, y, bins: np.ndarray | None = None,
                         n_bins: int = 10, strategy: str = "quantile") -> MurphyDecomposition:
    """Decompose the Brier score into reliability, resolution and uncertainty.

    A caveat worth understanding rather than hiding. The identity

        BS = reliability - resolution + uncertainty

    is exact only for forecasts taking finitely many values. Market prices
    are effectively continuous, so they must be binned, and the identity
    then holds for the *binned* forecast (each price replaced by its bin
    mean) rather than the raw one. The difference between the two, reported
    here as `binning_error`, is the within-bin variance that binning throws
    away. It is small but it is not zero, and a decomposition that quietly
    reported `residual = 0` against the raw Brier would be wrong.

    See `_bin_edges` for why this project uses quantile bins by default.
    """
    p, y = _validate(p, y)
    if bins is None:
        bins = _bin_edges(p, n_bins=n_bins, strategy=strategy)

    idx = np.clip(np.digitize(p, bins[1:-1], right=True), 0, len(bins) - 2)
    n = len(p)
    ybar = float(np.mean(y))

    reliability = 0.0
    resolution = 0.0
    used = 0
    p_binned = np.empty(n, dtype=float)
    for k in range(len(bins) - 1):
        mask = idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        used += 1
        pk = float(np.mean(p[mask]))   # representative forecast for the bin
        ok = float(np.mean(y[mask]))   # realised frequency in the bin
        p_binned[mask] = pk
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - ybar) ** 2

    reliability /= n
    resolution /= n
    uncertainty = ybar * (1 - ybar)

    bs = brier_score(p, y)
    bs_binned = float(np.mean((p_binned - y) ** 2))
    residual = bs_binned - (reliability - resolution + uncertainty)
    return MurphyDecomposition(
        brier=bs,
        brier_binned=bs_binned,
        binning_error=float(bs - bs_binned),
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        n_bins_used=used,
        residual=float(residual),
    )


def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        # Equal-count bins. On this dataset the median price is under 0.01, so
        # equal-width bins drop three quarters of the sample into [0, 0.1) and
        # leave the rest nearly empty -- the reliability diagram then says
        # almost nothing about where the mass actually is.
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
        edges[0], edges[-1] = 0.0, 1.0
        if len(edges) < 3:
            return np.linspace(0.0, 1.0, n_bins + 1)
        return edges
    raise ValueError(f"unknown binning strategy: {strategy!r}")


def expected_calibration_error(p, y, n_bins: int = 10,
                               strategy: str = "quantile") -> tuple[float, float]:
    """Return (ECE, MCE).

    ECE is the sample-weighted mean gap between forecast and realised
    frequency; MCE is the worst single bin. ECE can look excellent while one
    region of the probability scale is badly wrong, which is what MCE catches.
    """
    p, y = _validate(p, y)
    edges = _bin_edges(p, n_bins=n_bins, strategy=strategy)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    total = 0.0
    worst = 0.0
    n = len(p)
    for k in range(len(edges) - 1):
        mask = idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        gap = abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
        total += (nk / n) * gap
        worst = max(worst, gap)
    return float(total), float(worst)


def cluster_bootstrap(
    statistic,
    *arrays,
    clusters=None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260301,
):
    """Percentile bootstrap CI, resampling whole clusters rather than rows.

    Each market contributes up to four rows to this dataset (one per horizon),
    and those rows share an outcome, so they are emphatically not independent.
    Resampling rows would understate the width of every interval. Resampling
    markets does not.

    Pass clusters=None to get the ordinary i.i.d. bootstrap.

    Returns (point_estimate, lo, hi, n_effective_resamples).
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    point = float(statistic(*arrays))

    if clusters is None:
        groups = [np.array([i]) for i in range(n)]
    else:
        clusters = np.asarray(clusters)
        _, inverse = np.unique(clusters, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(inverse.max() + 2))
        groups = [order[bounds[g]:bounds[g + 1]] for g in range(len(bounds) - 1)]

    n_groups = len(groups)
    stats = np.empty(n_boot, dtype=float)
    kept = 0
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([groups[i] for i in pick])
        try:
            stats[kept] = statistic(*[a[rows] for a in arrays])
        except (ValueError, ZeroDivisionError):
            # A resample can be degenerate (e.g. all outcomes identical, so a
            # skill score has no reference variance). Dropping those is
            # standard; counting them as zero would bias the interval.
            continue
        kept += 1

    if kept == 0:
        return point, float("nan"), float("nan"), 0
    lo, hi = np.quantile(stats[:kept], [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi), kept
