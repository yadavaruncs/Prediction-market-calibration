"""Exploratory analysis: what is actually in this dataset before we model it."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import INK, ACCENT, WARN, GRID, _style


def summary(df: pd.DataFrame) -> dict:
    """Headline facts a reader needs before trusting anything downstream."""
    p = df["price"]
    out = {
        "n_observations": int(len(df)),
        "n_markets": int(df["market_id"].nunique()),
        "observations_per_market": float(len(df) / df["market_id"].nunique()),
        "horizons": sorted(int(h) for h in df["days_to_resolution"].unique()),
        "base_rate": float(df["outcome"].mean()),
        "class_balance": {
            "yes": int((df["outcome"] == 1).sum()),
            "no": int((df["outcome"] == 0).sum()),
        },
        "price": {
            "mean": float(p.mean()),
            "median": float(p.median()),
            "q25": float(p.quantile(0.25)),
            "q75": float(p.quantile(0.75)),
            "min": float(p.min()),
            "max": float(p.max()),
            "share_under_5c": float((p < 0.05).mean()),
            "share_over_50c": float((p > 0.50).mean()),
        },
        "date_from": str(pd.to_datetime(df["close_time"]).min().date()),
        "date_to": str(pd.to_datetime(df["close_time"]).max().date()),
    }
    if "category" in df:
        cats = df["category"].value_counts()
        out["categories"] = {str(k): int(v) for k, v in cats.items()}
        out["category_usable"] = bool(cats.index.nunique() > 1)
    return out


def base_rate_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("days_to_resolution").agg(
        n=("outcome", "size"),
        base_rate=("outcome", "mean"),
        mean_price=("price", "mean"),
        median_price=("price", "median"),
    ).reset_index()
    g["price_minus_rate"] = g["mean_price"] - g["base_rate"]
    return g


def drift_by_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Has the market's accuracy moved over the sample period?"""
    d = df.copy()
    d["quarter"] = pd.to_datetime(d["close_time"], utc=True).dt.to_period("Q").astype(str)
    g = d.groupby("quarter").agg(
        n=("outcome", "size"),
        base_rate=("outcome", "mean"),
        mean_price=("price", "mean"),
        brier=("outcome", lambda s: float(np.mean(
            (d.loc[s.index, "price"].to_numpy(float) - s.to_numpy(float)) ** 2))),
    ).reset_index()
    return g


def correlations(df: pd.DataFrame, columns) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns] + ["outcome"]
    return df[cols].corr(method="spearman")


# ---------------------------------------------------------------- figures

def plot_price_distribution(df: pd.DataFrame, out: Path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    _style(a); _style(b)

    a.hist(df["price"], bins=50, color=ACCENT, alpha=0.8)
    a.set_title("Price on a linear scale", fontsize=11, loc="left")
    a.set_xlabel("market price", fontsize=10)
    a.set_ylabel("count", fontsize=10)
    a.annotate("almost everything\nis a longshot", xy=(0.05, a.get_ylim()[1] * 0.6),
               fontsize=9, color=WARN)

    bins = np.logspace(np.log10(max(df["price"].min(), 1e-4)), 0, 50)
    b.hist(df["price"], bins=bins, color=ACCENT, alpha=0.8)
    b.set_xscale("log")
    b.set_title("Same data, log scale -- the structure appears", fontsize=11, loc="left")
    b.set_xlabel("market price (log)", fontsize=10)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_outcome_by_price_decile(df: pd.DataFrame, out: Path) -> Path:
    d = df.copy()
    d["decile"] = pd.qcut(d["price"], 10, labels=False, duplicates="drop")
    g = d.groupby("decile").agg(mean_price=("price", "mean"),
                                realised=("outcome", "mean"),
                                n=("outcome", "size")).reset_index()

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _style(ax)
    x = np.arange(len(g))
    w = 0.38
    ax.bar(x - w / 2, g["mean_price"], width=w, color=ACCENT, label="mean price")
    ax.bar(x + w / 2, g["realised"], width=w, color=WARN, label="realised frequency")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i+1}" for i in range(len(g))], fontsize=9)
    ax.set_xlabel("price decile", fontsize=10)
    ax.set_ylabel("probability", fontsize=10)
    ax.set_title("Priced vs realised, by decile", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_correlation_heatmap(corr: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else INK)
    ax.set_title("Spearman correlation", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_drift(drift: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    _style(ax)
    x = np.arange(len(drift))
    ax.plot(x, drift["brier"], "o-", color=ACCENT, label="Brier (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(drift["quarter"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Brier score", fontsize=10)
    ax.set_title("Market accuracy over time", fontsize=11, loc="left")
    for i, n in enumerate(drift["n"]):
        ax.annotate(f"n={int(n)}", (i, drift["brier"].iloc[i]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha="center", color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out
