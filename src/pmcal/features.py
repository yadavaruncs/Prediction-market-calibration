"""Cleaning, validation and feature engineering.

Design rule for this module: every feature must be computable from
information available *before* the market resolved. A feature that leaks the
outcome will produce a beautiful model and a worthless one, and on a dataset
this small the leak is easy to miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6

# Features the model is allowed to see. `logit_price` is the market's own
# forecast and acts as the baseline; everything else is a candidate for
# information the market may have missed.
FEATURE_COLUMNS = [
    "logit_price",
    "days_to_resolution",
    "log_volume",
    "is_longshot",
    "is_favourite",
    "month_index",
    "price_x_horizon",
]

BASELINE_COLUMNS = ["logit_price"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and record why.

    Prices at exactly 0 or 1 are dropped rather than clipped: a contract
    quoted at 0.000 is either already settled or has no book, and in both
    cases it is not a forecast anyone could have traded against.
    """
    out = df.copy()
    n0 = len(out)

    out = out.dropna(subset=["price", "outcome", "close_time", "days_to_resolution"])
    n_na = n0 - len(out)

    out = out[(out["price"] > 0.0) & (out["price"] < 1.0)]
    n_degenerate = n0 - n_na - len(out)

    before_dupes = len(out)
    out = out.drop_duplicates(subset=["market_id", "days_to_resolution"], keep="first")
    n_dupes = before_dupes - len(out)

    out.attrs["cleaning_report"] = {
        "rows_in": n0,
        "dropped_missing": int(n_na),
        "dropped_degenerate_price": int(n_degenerate),
        "dropped_duplicate_market_horizon": int(n_dupes),
        "rows_out": int(len(out)),
    }
    return out.reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features.

    logit_price       the market forecast on the log-odds scale. Linear models
                      on raw probability are badly misspecified near 0 and 1,
                      and three quarters of this sample sits below 0.3.
    log_volume        volume is heavy-tailed across several orders of
                      magnitude; the log makes it usable and interpretable as
                      'per decade of volume'. Known exception to the module's
                      design rule: the dataset only has each market's final
                      volume, identical at every horizon. Kept because the
                      leak can only flatter the models, and they still lose.
    is_longshot       below 5c. These dominate the sample by count and behave
                      differently enough to deserve their own indicator.
    is_favourite      above 50c.
    month_index       months since the first close in the sample. A crude
                      drift term -- if the market got better calibrated over
                      time, this picks some of it up.
    price_x_horizon   lets the weight on the market's forecast vary with how
                      far out it was made.
    """
    out = df.copy()
    p = out["price"].to_numpy(float)

    out["logit_price"] = np.log(np.clip(p, _EPS, 1 - _EPS) / np.clip(1 - p, _EPS, 1 - _EPS))

    vol = out["volume"].to_numpy(float) if "volume" in out else np.zeros(len(out))
    out["log_volume"] = np.log1p(np.clip(vol, 0, None))

    out["is_longshot"] = (p < 0.05).astype(int)
    out["is_favourite"] = (p > 0.50).astype(int)

    t = pd.to_datetime(out["close_time"], utc=True)
    out["month_index"] = ((t - t.min()).dt.days / 30.44).astype(float)

    out["price_x_horizon"] = out["logit_price"] * np.log1p(out["days_to_resolution"].astype(float))

    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """clean -> engineer, carrying the cleaning report through."""
    cleaned = clean(df)
    report = cleaned.attrs.get("cleaning_report", {})
    out = engineer(cleaned)
    out.attrs["cleaning_report"] = report
    return out


def feature_matrix(df: pd.DataFrame, columns=None):
    """Return (X, y, groups) ready for sklearn.

    groups is market_id, so that cross-validation can keep all horizons of
    one market inside the same fold. Splitting them across folds would let
    the model see a market's outcome in training and predict the same
    market's outcome in validation -- a leak that looks like skill.
    """
    cols = list(columns) if columns is not None else list(FEATURE_COLUMNS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing engineered columns: {missing} -- call prepare() first")
    X = df[cols].to_numpy(float)
    y = df["outcome"].to_numpy(int)
    groups = df["market_id"].to_numpy()
    return X, y, groups
