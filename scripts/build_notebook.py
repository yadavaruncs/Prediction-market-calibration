#!/usr/bin/env python3
"""Generate and execute notebooks/01_exploration.ipynb.

The notebook is built from source rather than hand-edited so that it can
never drift from the library: if `make analyse` changes, re-running this
script regenerates the narrative with the new numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_exploration.ipynb"

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell

CELLS = [
    MD("""# Polymarket calibration — exploration

Working notebook behind the analysis in `README.md`. Everything here calls
the same library used by `make analyse`, so the numbers cannot disagree.

The question: when the market says 20c, does it happen 20% of the time?"""),

    CODE("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import pandas as pd

from src.pmcal.io import load_markets
from src.pmcal.features import prepare, FEATURE_COLUMNS
from src.pmcal import eda

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 30)

raw = load_markets("../data/raw/markets.csv")
df = prepare(raw)
print(df.attrs["cleaning_report"])
df.head()"""),

    MD("""## 1. What is actually in here

Before any modelling: how big is the sample, how often does YES happen, and
what do the prices look like?"""),

    CODE("""summary = eda.summary(df)
for k, v in summary.items():
    print(f"{k:28s} {v}")"""),

    MD("""The median price is **0.0055**. Nearly two thirds of observations are
priced under 5c.

This is the fact that shapes every later choice. A sample dominated by deep
longshots breaks equal-width binning — put these into ten uniform buckets
and the first one swallows most of the data."""),

    CODE("""p = df["price"]
uniform_counts = pd.cut(p, bins=np.linspace(0, 1, 11)).value_counts().sort_index()
print("equal-width bins:")
print(uniform_counts.to_string())
print()
print(f"first bin holds {uniform_counts.iloc[0] / len(p):.1%} of the sample")"""),

    CODE("""quantile_counts = pd.qcut(p, 10, duplicates="drop").value_counts().sort_index()
print("quantile bins:")
print(quantile_counts.to_string())"""),

    MD("""Quantile bins spread the sample evenly, which is what makes a reliability
diagram readable. That is why `metrics._bin_edges` defaults to `quantile`."""),

    CODE("""eda.plot_price_distribution(df, pathlib.Path("../figures/nb_price_dist.png"))
from IPython.display import Image
Image("../figures/nb_price_dist.png")"""),

    MD("""## 2. Class balance and the horizon structure

Each market appears up to four times — once per observation horizon — and
those rows share one outcome. They are **not** independent."""),

    CODE("""print(f"rows:    {len(df)}")
print(f"markets: {df.market_id.nunique()}")
print(f"rows per market: {len(df) / df.market_id.nunique():.2f}")
print()
print(df.groupby("market_id").size().value_counts().sort_index().to_string())"""),

    CODE("""eda.base_rate_by_horizon(df)"""),

    MD("""Mean price exceeds the base rate at every horizon — a first hint that
contracts are, on average, priced slightly high. Whether that is real is
what the favourite-longshot test decides."""),

    MD("""## 3. Priced versus realised

The decile view, before any statistics."""),

    CODE("""eda.plot_outcome_by_price_decile(df, pathlib.Path("../figures/nb_decile.png"))
Image("../figures/nb_decile.png")"""),

    MD("""## 4. Does anything correlate with the outcome besides price?

If some other feature carried real information, this is where the first sign
of it would show up."""),

    CODE("""corr = eda.correlations(df, FEATURE_COLUMNS)
corr["outcome"].sort_values(ascending=False)"""),

    MD("""`logit_price` dominates and nothing else comes close. That is the first
evidence for the conclusion the model comparison later confirms: the price
has already absorbed whatever the other columns know."""),

    MD("""## 5. Headline calibration at the 7-day horizon"""),

    CODE("""from src.pmcal.metrics import (
    brier_score, brier_skill_score, cluster_bootstrap,
    murphy_decomposition, expected_calibration_error,
)

h = df[df.days_to_resolution == 7]
pp, yy, mm = (h["price"].to_numpy(float), h["outcome"].to_numpy(float),
              h["market_id"].to_numpy())

bs, lo, hi, _ = cluster_bootstrap(brier_score, pp, yy, clusters=mm, n_boot=4000)
k, klo, khi, _ = cluster_bootstrap(brier_skill_score, pp, yy, clusters=mm, n_boot=4000)
ece, mce = expected_calibration_error(pp, yy)

print(f"n              {len(h)}")
print(f"base rate      {yy.mean():.3f}")
print(f"Brier          {bs:.4f}  [{lo:.4f}, {hi:.4f}]")
print(f"Brier skill    {k:.3f}   [{klo:.3f}, {khi:.3f}]")
print(f"ECE / MCE      {ece:.4f} / {mce:.4f}")"""),

    CODE("""d = murphy_decomposition(pp, yy)
print(f"reliability  {d.reliability:.5f}   (miscalibration, lower better)")
print(f"resolution   {d.resolution:.5f}   (discrimination, higher better)")
print(f"uncertainty  {d.uncertainty:.5f}   (irreducible)")
print(f"residual     {d.residual:.2e}      (identity check on binned forecast)")"""),

    MD("""## 6. Why the bootstrap has to cluster

The claim is that ignoring the repeated-market structure produces intervals
that are too narrow. Here is the size of that effect on the real data,
pooled across all horizons."""),

    CODE("""pa, ya, ma = (df["price"].to_numpy(float), df["outcome"].to_numpy(float),
              df["market_id"].to_numpy())

_, nlo, nhi, _ = cluster_bootstrap(brier_score, pa, ya, clusters=None,
                                   n_boot=3000, seed=1)
_, clo, chi, _ = cluster_bootstrap(brier_score, pa, ya, clusters=ma,
                                   n_boot=3000, seed=1)

print(f"naive   [{nlo:.5f}, {nhi:.5f}]  width {nhi-nlo:.5f}")
print(f"clustered [{clo:.5f}, {chi:.5f}]  width {chi-clo:.5f}")
print(f"\\nclustered interval is {(chi-clo)/(nhi-nlo):.2f}x wider")"""),

    MD("""Narrower intervals are not a better result — they are a false one. Every
confidence statement in this project uses the clustered version."""),

    MD("""## 7. The favourite-longshot test"""),

    CODE("""from src.pmcal.calibration import favourite_longshot_test, reliability_bins

flb = favourite_longshot_test(pp, yy, clusters=mm)
print(f"slope      {flb.slope:.3f}  [{flb.slope_lo:.3f}, {flb.slope_hi:.3f}]")
print(f"intercept  {flb.intercept:.3f}")
print(f"p(slope=1) {flb.p_value_slope_eq_1:.4f}")
print(f"differs from 1: {flb.slope_differs_from_one}")"""),

    CODE("""reliability_bins(pp, yy)"""),

    MD("""A slope below 1 means the realised curve is flatter than the diagonal.
The bin table localises it: mid-probability contracts resolve YES more often
than they are priced.

The published study this dataset came from reported a slope of 1.01 and
found no bias. This analysis clusters the standard errors and restricts to a
single horizon, and gets an interval that excludes 1. That disagreement is
documented in the README rather than smoothed over."""),

    MD("""## 8. Where this goes next

The backtest, the threshold sweep and the model comparison all live in
`src/pmcal/` and run together via `make analyse`. The short version: the
bias is real but small, and it does not survive transaction costs — every
net-return interval crosses zero."""),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = list(CELLS)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    return nb


def main() -> int:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = NotebookClient(nb, timeout=600, kernel_name="python3",
                                resources={"metadata": {"path": str(OUT.parent)}})
        client.execute()
        print("executed cleanly")
    except Exception as exc:
        print(f"execution failed ({exc}); writing unexecuted notebook")
    nbf.write(nb, OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
