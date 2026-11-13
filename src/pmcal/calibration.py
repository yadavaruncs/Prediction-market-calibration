"""Reliability curves, the favourite-longshot test, and recalibration.

The question this module answers: when Polymarket says 20%, does it happen
20% of the time? And if it does not, is the gap big enough to be real?
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .metrics import _bin_edges, _validate

_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def reliability_bins(p, y, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Binned forecast vs realised frequency, with a per-bin Wilson interval.

    The Wilson interval is used rather than the normal approximation because
    several bins here sit at realised frequencies very close to 0, where the
    normal interval runs below zero and stops meaning anything.
    """
    p, y = _validate(p, y)
    edges = _bin_edges(p, n_bins=n_bins, strategy=strategy)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    rows = []
    for k in range(len(edges) - 1):
        mask = idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        mean_p = float(np.mean(p[mask]))
        hits = int(np.sum(y[mask]))
        freq = hits / nk
        lo, hi = _wilson(hits, nk)
        rows.append(
            dict(
                bin=k,
                lo_edge=float(edges[k]),
                hi_edge=float(edges[k + 1]),
                n=nk,
                mean_price=mean_p,
                realised_freq=freq,
                wilson_lo=lo,
                wilson_hi=hi,
                gap=mean_p - freq,
            )
        )
    return pd.DataFrame(rows)


def _wilson(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


@dataclass
class FavouriteLongshot:
    """Regression of outcome on logit(price).

    slope = 1 and intercept = 0 is a perfectly calibrated market.

    slope > 1 is the classic favourite-longshot bias: the realised curve is
    steeper than the diagonal, so longshots are priced too high and
    favourites too low. slope < 1 is the reverse: a flatter curve, with
    longshots underpriced and favourites overpriced.

    The 7-day result on this data (slope 0.867, CI excluding 1) is the
    reverse bias. The bins show it: contracts near 5c resolve at 15.7%,
    contracts near 83.5c at 81.9%.

    The test that matters is whether the CI for the slope excludes 1 -- not
    whether the point estimate happens to differ from it.

    `ols_calibration_slope` returns the same fields for the identity-scale
    fit (outcome on raw price); its slope is a different estimand.
    """

    slope: float
    slope_lo: float
    slope_hi: float
    intercept: float
    intercept_lo: float
    intercept_hi: float
    n: int
    slope_differs_from_one: bool
    cluster_robust: bool
    p_value_slope_eq_1: float = field(default=float("nan"))

    def as_dict(self) -> dict:
        return asdict(self)


def favourite_longshot_test(p, y, clusters=None, alpha: float = 0.05) -> FavouriteLongshot:
    """Fit outcome ~ logit(price) by logistic regression and test slope == 1.

    Standard errors are cluster-robust by market when `clusters` is supplied,
    for the same reason the bootstrap clusters: repeated horizons on one
    market are not independent observations.
    """
    p, y = _validate(p, y)
    X = sm.add_constant(logit(p))
    model = sm.Logit(y, X)

    if clusters is not None:
        res = model.fit(disp=0, cov_type="cluster",
                        cov_kwds={"groups": np.asarray(clusters)})
    else:
        res = model.fit(disp=0)

    ci = res.conf_int(alpha=alpha)
    intercept, slope = float(res.params[0]), float(res.params[1])
    slope_lo, slope_hi = float(ci[1][0]), float(ci[1][1])

    # Wald test of H0: slope == 1, written out rather than delegated.
    # statsmodels has changed the signature of wald_test across versions and
    # the statistic is two lines: ((b - 1) / se)^2 is chi-square with 1 df.
    se_slope = float(res.bse[1])
    if se_slope > 0:
        wald_stat = ((slope - 1.0) / se_slope) ** 2
        pval = float(stats.chi2.sf(wald_stat, df=1))
    else:
        pval = float("nan")

    return FavouriteLongshot(
        slope=slope,
        slope_lo=slope_lo,
        slope_hi=slope_hi,
        intercept=intercept,
        intercept_lo=float(ci[0][0]),
        intercept_hi=float(ci[0][1]),
        n=int(len(p)),
        slope_differs_from_one=not (slope_lo <= 1.0 <= slope_hi),
        cluster_robust=clusters is not None,
        p_value_slope_eq_1=pval,
    )


def ols_calibration_slope(p, y, clusters=None, alpha: float = 0.05) -> FavouriteLongshot:
    """Fit outcome ~ price by OLS (identity scale) and test slope == 1.

    This is the specification of the study the dataset came from. A
    calibrated market gives slope 1 here too, but deviations are measured in
    probability rather than log-odds, so most of this sample (priced under
    5c, where squared errors are tiny) has little leverage. The slope is not
    comparable to `favourite_longshot_test`'s.
    """
    p, y = _validate(p, y)
    model = sm.OLS(y, sm.add_constant(p))
    if clusters is not None:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(clusters)})
    else:
        res = model.fit(cov_type="HC1")

    ci = res.conf_int(alpha=alpha)
    intercept, slope = float(res.params[0]), float(res.params[1])
    slope_lo, slope_hi = float(ci[1][0]), float(ci[1][1])
    se_slope = float(res.bse[1])
    pval = float(stats.chi2.sf(((slope - 1.0) / se_slope) ** 2, df=1)) if se_slope > 0 else float("nan")

    return FavouriteLongshot(
        slope=slope,
        slope_lo=slope_lo,
        slope_hi=slope_hi,
        intercept=intercept,
        intercept_lo=float(ci[0][0]),
        intercept_hi=float(ci[0][1]),
        n=int(len(p)),
        slope_differs_from_one=not (slope_lo <= 1.0 <= slope_hi),
        cluster_robust=clusters is not None,
        p_value_slope_eq_1=pval,
    )


def walk_forward_recalibration(
    df: pd.DataFrame,
    price_col: str = "price",
    outcome_col: str = "outcome",
    time_col: str = "close_time",
    horizon_col: str = "days_to_resolution",
    min_train: int = 150,
    refit_every: int = 25,
) -> pd.DataFrame:
    """Platt scaling fit only on the past, applied to the next market.

    This is the part of a calibration study that is easiest to get wrong. If
    you fit the recalibration on the whole sample and then score it on the
    whole sample, you will always find an edge, because the model has already
    seen the answers. Every fit here uses an expanding window of markets that
    had already resolved when the price being recalibrated was observed, which
    is `horizon_col` days before that market's own close. Expects one horizon
    per call, so that this window only ever grows.

    Refitting every `refit_every` new outcomes rather than every row is a speed
    tradeoff; with ~900 markets the difference in output is negligible and it
    keeps the whole pipeline under a second.

    Returns the input frame, ordered by time, with a `p_recal` column that is
    NaN for the burn-in rows with fewer than `min_train` known outcomes.
    """
    out = df.sort_values(time_col, kind="stable").reset_index(drop=True).copy()
    p = out[price_col].to_numpy(float)
    y = out[outcome_col].to_numpy(float)
    z = logit(p)

    observed_at = out[time_col]
    if horizon_col in out:
        observed_at = observed_at - pd.to_timedelta(out[horizon_col], unit="D")
    # side="left": a market closing at the observation instant is not yet known.
    n_known = out[time_col].searchsorted(observed_at, side="left")

    recal = np.full(len(out), np.nan)
    coef = None
    fitted_on = 0
    for i in range(len(out)):
        k = int(n_known[i])
        if k < min_train:
            continue
        if coef is None or k - fitted_on >= refit_every:
            Xtr = sm.add_constant(z[:k])
            ytr = y[:k]
            if len(np.unique(ytr)) < 2:
                continue
            try:
                fit = sm.Logit(ytr, Xtr).fit(disp=0)
                coef = (float(fit.params[0]), float(fit.params[1]))
                fitted_on = k
            except Exception:
                # Separation or a singular design in an early window: keep the
                # previous coefficients rather than inventing a forecast.
                pass
        if coef is not None:
            recal[i] = inv_logit(coef[0] + coef[1] * z[i])

    out["p_recal"] = recal
    return out
