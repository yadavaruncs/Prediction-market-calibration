"""Figures. Two rules: show the sample size, and never hide an interval."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1a1a1a"
ACCENT = "#2b6cb0"
WARN = "#c05621"
GRID = "#d8d8d8"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)
    return ax


def reliability_diagram(bins: pd.DataFrame, title: str, out: Path,
                        logit_scale: bool = False) -> Path:
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6.2, 6.4), height_ratios=[3, 1], sharex=True
    )
    _style(ax)
    _style(axh)

    # Drawn as a dense grid, not two endpoints: on a logit x-axis the
    # identity y = x is a curve, and a two-point line would render as a
    # chord that quietly misrepresents where perfect calibration sits.
    grid = np.linspace(1e-4, 1 - 1e-4, 500)
    ax.plot(grid, grid, color=INK, linewidth=1, linestyle="--",
            label="perfect calibration")

    yerr = np.vstack([
        bins["realised_freq"] - bins["wilson_lo"],
        bins["wilson_hi"] - bins["realised_freq"],
    ])
    ax.errorbar(bins["mean_price"], bins["realised_freq"], yerr=yerr,
                fmt="o", color=ACCENT, ecolor=ACCENT, elinewidth=1.2,
                capsize=3, markersize=5, label="observed (95% Wilson)")

    ax.set_ylabel("realised frequency", color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    if logit_scale:
        ax.set_xscale("logit")
        axh.set_xscale("logit")

    # Quantile bins can be narrower than a pixel at the longshot end -- this
    # dataset's first bin spans 0.0015 and holds 297 of 825 observations. A
    # bar drawn to true width would be invisible, which would hide exactly
    # where the sample mass is, so narrow bars get a floor.
    true_w = (bins["hi_edge"] - bins["lo_edge"]) * 0.8
    if logit_scale:
        # In log space a constant floor is meaningless; scale each bar to a
        # fixed fraction of its own position instead.
        widths = np.maximum(true_w, bins["mean_price"] * 0.6)
    else:
        widths = np.maximum(true_w, 0.012)
    axh.bar(bins["mean_price"], bins["n"], width=widths,
            color=ACCENT, alpha=0.45)
    for _, r in bins.iterrows():
        axh.annotate(f"{int(r['n'])}", (r["mean_price"], r["n"]),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=7, color=INK)
    axh.set_ylabel("n", color=INK, fontsize=10)
    axh.set_xlabel("market price (implied probability)", color=INK, fontsize=10)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def horizon_panel(rows: pd.DataFrame, out: Path) -> Path:
    """Brier skill score by forecast horizon, with intervals."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    _style(ax)
    x = np.arange(len(rows))
    yerr = np.vstack([
        rows["bss"] - rows["bss_lo"],
        rows["bss_hi"] - rows["bss"],
    ])
    ax.errorbar(x, rows["bss"], yerr=yerr, fmt="o", color=ACCENT,
                capsize=4, markersize=6, elinewidth=1.3)
    ax.axhline(0, color=INK, linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}d\nn={int(n)}" for h, n in
                        zip(rows["horizon"], rows["n"])], fontsize=9)
    ax.set_ylabel("Brier skill score", fontsize=10)
    ax.set_title("Forecast skill against the base rate, by horizon",
                 fontsize=11, loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def threshold_curve(sweep: pd.DataFrame, out: Path) -> Path:
    """Net ROI across signal thresholds, with the zero line visible."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _style(ax)
    yerr = np.vstack([
        sweep["net_roi"] - sweep["net_roi_lo"],
        sweep["net_roi_hi"] - sweep["net_roi"],
    ])
    ax.errorbar(sweep["threshold"], sweep["net_roi"], yerr=yerr, fmt="o-",
                color=WARN, capsize=4, markersize=5, elinewidth=1.2,
                label="net ROI (95% CI)")
    ax.axhline(0, color=INK, linewidth=1, linestyle="--")
    for _, r in sweep.iterrows():
        ax.annotate(f"n={int(r['n_trades'])}",
                    (r["threshold"], r["net_roi"]),
                    textcoords="offset points", xytext=(0, 9),
                    fontsize=8, color=INK, ha="center")
    ax.set_xlabel("signal threshold (|recalibrated - price|)", fontsize=10)
    ax.set_ylabel("net return on notional", fontsize=10)
    ax.set_title("Every interval crosses zero", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out
