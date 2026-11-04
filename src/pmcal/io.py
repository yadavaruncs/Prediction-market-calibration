"""Loading and validating the market dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "market_id", "price", "outcome", "days_to_resolution", "close_time",
}

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "markets.csv"


def load_markets(path: str | Path = DEFAULT_PATH, horizon: int | None = None) -> pd.DataFrame:
    """Load the observation table.

    One row is one (market, horizon) pair: the mid price observed N days
    before the market resolved, plus the outcome it eventually settled to.
    A single market therefore appears up to four times, which is why every
    interval in this project clusters by market_id.

    Pass `horizon` to slice to a single days_to_resolution value.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset at {path}. Run `make fetch` to download it, or see data/README.md."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")

    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, format="mixed")
    df = df.dropna(subset=["price", "outcome", "close_time"])
    df = df[(df["price"] > 0) & (df["price"] < 1)]
    df["outcome"] = df["outcome"].astype(int)

    if horizon is not None:
        df = df[df["days_to_resolution"] == horizon]
        if df.empty:
            raise ValueError(f"no rows at horizon={horizon}")

    return df.sort_values("close_time", kind="stable").reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict:
    return {
        "n_observations": int(len(df)),
        "n_markets": int(df["market_id"].nunique()),
        "horizons": sorted(int(h) for h in df["days_to_resolution"].unique()),
        "base_rate": float(df["outcome"].mean()),
        "median_price": float(df["price"].median()),
        "date_from": str(df["close_time"].min().date()),
        "date_to": str(df["close_time"].max().date()),
    }
