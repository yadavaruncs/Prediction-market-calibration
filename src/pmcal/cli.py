"""Run the whole analysis end to end and write results + figures.

    python -m src.pmcal.cli --data data/raw/markets.csv --out results --figures figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import eda, models, plots
from .backtest import CostModel, run_backtest, threshold_sweep
from .calibration import (
    favourite_longshot_test, reliability_bins, walk_forward_recalibration,
)
from .features import FEATURE_COLUMNS, prepare
from .io import load_markets
from .metrics import (
    brier_score, brier_skill_score, cluster_bootstrap,
    expected_calibration_error, log_loss, murphy_decomposition,
)

HEADLINE_HORIZON = 7


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, pd.Timestamp):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prediction market calibration analysis")
    ap.add_argument("--data", default="data/raw/markets.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--horizon", type=int, default=HEADLINE_HORIZON)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args(argv)

    out_dir, fig_dir = Path(args.out), Path(args.figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    R: dict = {}

    # ---------------------------------------------------------------- 1. EDA
    print("[1/6] loading and exploring")
    raw = load_markets(args.data)
    df = prepare(raw)
    R["cleaning"] = df.attrs.get("cleaning_report", {})
    R["eda"] = eda.summary(df)
    R["base_rate_by_horizon"] = eda.base_rate_by_horizon(df)
    drift = eda.drift_by_quarter(df)
    R["drift_by_quarter"] = drift

    corr = eda.correlations(df, FEATURE_COLUMNS)
    eda.plot_price_distribution(df, fig_dir / "01_price_distribution.png")
    eda.plot_outcome_by_price_decile(df, fig_dir / "02_price_vs_realised.png")
    eda.plot_correlation_heatmap(corr, fig_dir / "03_correlation.png")
    eda.plot_drift(drift, fig_dir / "04_drift.png")

    # ------------------------------------------------- 2. headline calibration
    print(f"[2/6] calibration at the {args.horizon}-day horizon")
    h = df[df["days_to_resolution"] == args.horizon].reset_index(drop=True)
    p, y = h["price"].to_numpy(float), h["outcome"].to_numpy(float)
    mk = h["market_id"].to_numpy()

    bs, bs_lo, bs_hi, _ = cluster_bootstrap(brier_score, p, y, clusters=mk,
                                            n_boot=args.n_boot)
    bss, bss_lo, bss_hi, _ = cluster_bootstrap(brier_skill_score, p, y,
                                               clusters=mk, n_boot=args.n_boot)
    ece, mce = expected_calibration_error(p, y, n_bins=args.bins)
    murphy = murphy_decomposition(p, y, n_bins=args.bins)

    R["headline"] = {
        "horizon_days": args.horizon,
        "n_markets": int(len(h)),
        "base_rate": float(y.mean()),
        "brier": bs, "brier_lo": bs_lo, "brier_hi": bs_hi,
        "brier_skill_score": bss, "bss_lo": bss_lo, "bss_hi": bss_hi,
        "log_loss": log_loss(p, y),
        "ece": ece, "mce": mce,
        "murphy": murphy.as_dict(),
    }

    bins = reliability_bins(p, y, n_bins=args.bins)
    R["reliability_bins"] = bins
    plots.reliability_diagram(
        bins, f"Polymarket calibration, {args.horizon} days before resolution "
              f"(n={len(h)})", fig_dir / "05_reliability.png")
    # Most of the sample lives below 5c, where a linear axis compresses every
    # longshot bin into the left edge. The logit axis is where that region is
    # actually readable.
    plots.reliability_diagram(
        bins, f"Same data on a logit axis -- the longshot end, legible "
              f"(n={len(h)})", fig_dir / "05b_reliability_logit.png",
        logit_scale=True)

    # ---------------------------------------------- 3. favourite-longshot test
    print("[3/6] favourite-longshot test")
    flb = favourite_longshot_test(p, y, clusters=mk)
    R["favourite_longshot"] = flb.as_dict()

    # ------------------------------------------------------ 4. across horizons
    print("[4/6] all horizons")
    rows = []
    for hz in sorted(df["days_to_resolution"].unique()):
        s = df[df["days_to_resolution"] == hz]
        ps, ys, ms = (s["price"].to_numpy(float), s["outcome"].to_numpy(float),
                      s["market_id"].to_numpy())
        b, blo, bhi, _ = cluster_bootstrap(brier_score, ps, ys, clusters=ms,
                                           n_boot=max(2000, args.n_boot // 5))
        k, klo, khi, _ = cluster_bootstrap(brier_skill_score, ps, ys, clusters=ms,
                                           n_boot=max(2000, args.n_boot // 5))
        rows.append(dict(horizon=int(hz), n=int(len(s)), base_rate=float(ys.mean()),
                         brier=b, brier_lo=blo, brier_hi=bhi,
                         bss=k, bss_lo=klo, bss_hi=khi))
    horizons = pd.DataFrame(rows)
    R["by_horizon"] = horizons
    plots.horizon_panel(horizons, fig_dir / "06_horizons.png")

    # ------------------------------------- 5. walk-forward recal + backtest
    print("[5/6] walk-forward recalibration and backtest")
    wf = walk_forward_recalibration(h)
    scored = wf.dropna(subset=["p_recal"])
    if len(scored) > 0:
        R["walk_forward"] = {
            "n_scored": int(len(scored)),
            "burn_in": int(len(wf) - len(scored)),
            "brier_market": brier_score(scored["price"], scored["outcome"]),
            "brier_recalibrated": brier_score(scored["p_recal"], scored["outcome"]),
        }
        R["walk_forward"]["improvement"] = (
            R["walk_forward"]["brier_market"] - R["walk_forward"]["brier_recalibrated"]
        )

    costs = CostModel()
    bt = run_backtest(wf, costs, threshold=0.05, n_boot=args.n_boot)
    R["backtest"] = bt.as_dict()
    R["cost_model"] = {"half_spread": costs.half_spread, "fee": costs.fee,
                       "slippage": costs.slippage, "total": costs.total}

    sweep = threshold_sweep(wf, costs)
    R["threshold_sweep"] = sweep
    plots.threshold_curve(sweep, fig_dir / "07_threshold_sweep.png")

    # ----------------------------------------------------- 6. model comparison
    if not args.skip_models:
        print("[6/6] model comparison (beating the market price is the bar)")
        table, artefacts = models.compare_models(df)
        R["model_comparison"] = table
        try:
            imp = models.feature_importance(df, artefacts)
            R["feature_importance"] = imp
        except Exception as exc:  # pragma: no cover
            R["feature_importance_error"] = str(exc)
    else:
        print("[6/6] skipped")

    # ------------------------------------------------------------- write out
    (out_dir / "results.json").write_text(
        json.dumps(R, indent=2, default=_jsonable))
    _write_report(R, out_dir / "report.md")
    print(f"\nwrote {out_dir/'results.json'} and {out_dir/'report.md'}")
    print(f"figures in {fig_dir}/")
    return 0


def _write_report(R: dict, path: Path) -> None:
    h = R.get("headline", {})
    m = h.get("murphy", {})
    flb = R.get("favourite_longshot", {})
    bt = R.get("backtest", {})
    e = R.get("eda", {})

    L = []
    L.append("# Results\n")
    L.append("Generated by `python -m src.pmcal.cli`. Every number here comes "
             "from that run; nothing is copied in by hand.\n")

    L.append("\n## Dataset\n")
    L.append(f"- {e.get('n_observations')} observations across "
             f"{e.get('n_markets')} resolved binary markets")
    L.append(f"- {e.get('date_from')} to {e.get('date_to')}")
    L.append(f"- base rate {e.get('base_rate', float('nan')):.3f}; "
             f"median price {e.get('price', {}).get('median', float('nan')):.4f}")
    L.append(f"- {e.get('price', {}).get('share_under_5c', float('nan')):.1%} "
             "of observations priced under 5c\n")

    L.append("\n## Headline calibration "
             f"({h.get('horizon_days')} days before resolution, n={h.get('n_markets')})\n")
    L.append("| metric | value | 95% CI |")
    L.append("| --- | --- | --- |")
    L.append(f"| Brier score | {h.get('brier', float('nan')):.4f} | "
             f"[{h.get('brier_lo', float('nan')):.4f}, {h.get('brier_hi', float('nan')):.4f}] |")
    L.append(f"| Brier skill score | {h.get('brier_skill_score', float('nan')):.3f} | "
             f"[{h.get('bss_lo', float('nan')):.3f}, {h.get('bss_hi', float('nan')):.3f}] |")
    L.append(f"| Log loss | {h.get('log_loss', float('nan')):.4f} | - |")
    L.append(f"| ECE | {h.get('ece', float('nan')):.4f} | - |")
    L.append(f"| MCE | {h.get('mce', float('nan')):.4f} | - |\n")

    L.append("\n### Murphy decomposition\n")
    L.append("Brier = reliability - resolution + uncertainty\n")
    L.append(f"- reliability (miscalibration, lower better): {m.get('reliability', float('nan')):.5f}")
    L.append(f"- resolution (discrimination, higher better): {m.get('resolution', float('nan')):.5f}")
    L.append(f"- uncertainty (irreducible): {m.get('uncertainty', float('nan')):.5f}")
    L.append(f"- identity residual: {m.get('residual', float('nan')):.2e}\n")

    L.append("\n## Favourite-longshot test\n")
    L.append(f"- slope {flb.get('slope', float('nan')):.3f} "
             f"[{flb.get('slope_lo', float('nan')):.3f}, {flb.get('slope_hi', float('nan')):.3f}]")
    L.append(f"- differs from 1: **{flb.get('slope_differs_from_one')}** "
             f"(p = {flb.get('p_value_slope_eq_1', float('nan')):.3f})\n")

    L.append("\n## Backtest\n")
    L.append(f"- trades: {bt.get('n_trades')} of {bt.get('n_candidates')} candidates")
    L.append(f"- gross ROI: {bt.get('gross_roi', float('nan')):.1%}")
    L.append(f"- net ROI: {bt.get('net_roi', float('nan')):.1%} "
             f"[{bt.get('net_roi_lo', float('nan')):.1%}, {bt.get('net_roi_hi', float('nan')):.1%}]")
    L.append(f"- statistically significant: **{bt.get('significant')}**\n")

    if "model_comparison" in R:
        L.append("\n## Model comparison\n")
        L.append("Bar to clear: the `market_price` row. A positive "
                 "`skill_vs_market` means the model found information the "
                 "price did not already contain.\n")
        t = pd.DataFrame(R["model_comparison"]) if not isinstance(
            R["model_comparison"], pd.DataFrame) else R["model_comparison"]
        cols = ["name", "brier", "roc_auc", "macro_f1", "skill_vs_market"]
        L.append(t[cols].to_markdown(index=False, floatfmt=".4f"))
        L.append("")

    if "feature_importance" in R:
        L.append("\n## Permutation importance\n")
        t = pd.DataFrame(R["feature_importance"])
        L.append(t.to_markdown(index=False, floatfmt=".5f"))
        L.append("")

    path.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
