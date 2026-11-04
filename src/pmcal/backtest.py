"""Does the calibration gap survive transaction costs?

Almost always the answer is no, and the interesting work is showing *why*
the headline return number should not be believed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .metrics import cluster_bootstrap


@dataclass
class CostModel:
    """Costs charged on entry, as a fraction of notional.

    half_spread   you cross the book to get filled, so you pay half the
                  quoted spread. Polymarket books on thin markets are wide;
                  2c is optimistic for a longshot and pessimistic for a
                  liquid one.
    fee           platform/settlement fee.
    slippage      the extra you pay for not being the only person who
                  noticed. Set to zero to see the frictionless case.
    """

    half_spread: float = 0.02
    fee: float = 0.00
    slippage: float = 0.005

    @property
    def total(self) -> float:
        return self.half_spread + self.fee + self.slippage


@dataclass
class BacktestResult:
    n_trades: int
    n_candidates: int
    threshold: float
    gross_roi: float
    net_roi: float
    net_roi_lo: float
    net_roi_hi: float
    hit_rate: float
    mean_edge: float
    cost_per_trade: float
    significant: bool

    def as_dict(self) -> dict:
        return asdict(self)


def divergence_signal(df: pd.DataFrame, threshold: float = 0.05,
                      price_col: str = "price",
                      recal_col: str = "p_recal") -> pd.DataFrame:
    """Trade where the recalibrated probability disagrees with the market.

    Buy YES when the recalibrated model says the event is more likely than
    the price implies, buy NO in the reverse case. One contract per
    trade -- no sizing, no Kelly, because position sizing on an edge you have
    not established yet is how you turn a statistical artefact into a loss.
    """
    out = df.dropna(subset=[recal_col]).copy()
    out["edge"] = out[recal_col] - out[price_col]
    out["side"] = np.where(out["edge"] > 0, "YES", "NO")
    out["trade"] = out["edge"].abs() >= threshold
    return out


def run_backtest(df: pd.DataFrame, costs: CostModel, threshold: float = 0.05,
                 price_col: str = "price", outcome_col: str = "outcome",
                 cluster_col: str | None = "market_id",
                 n_boot: int = 10_000, seed: int = 20260301) -> BacktestResult:
    """Return per-trade ROI with a bootstrap interval around it.

    Payoff convention: a YES contract bought at price q pays 1 if the event
    happens and 0 otherwise, so a single trade returns (y - q) / q. For a
    longshot at q = 0.005 that is a 200x multiple, which would swamp a simple
    average of per-trade returns.

    So returns are pooled instead: total profit over total capital staked,
    buying one contract per trade. A longshot stakes little and so weighs
    little, which is the only way the figure is readable.
    """
    sig = divergence_signal(df, threshold=threshold, price_col=price_col)
    trades = sig[sig["trade"]].copy()

    if len(trades) == 0:
        return BacktestResult(0, len(sig), threshold, float("nan"), float("nan"),
                              float("nan"), float("nan"), float("nan"),
                              float("nan"), costs.total, False)

    q = trades[price_col].to_numpy(float)
    y = trades[outcome_col].to_numpy(float)
    is_yes = (trades["side"] == "YES").to_numpy()

    # Effective entry price after crossing the spread and paying costs.
    q_eff = np.where(is_yes, q + costs.total, (1 - q) + costs.total)
    q_eff = np.clip(q_eff, 1e-4, 1 - 1e-4)

    win = np.where(is_yes, y == 1, y == 0)

    gross_entry = np.where(is_yes, q, 1 - q)
    gross_pnl = np.where(win, 1.0 - gross_entry, -gross_entry)
    net_pnl = np.where(win, 1.0 - q_eff, -q_eff)

    def _roi(pnl, stake):
        s = float(np.sum(stake))
        return float(np.sum(pnl) / s) if s > 0 else float("nan")

    gross_roi = _roi(gross_pnl, gross_entry)
    net_roi = _roi(net_pnl, q_eff)

    clusters = trades[cluster_col].to_numpy() if cluster_col in trades else None
    _, lo, hi, _ = cluster_bootstrap(
        lambda a, b: _roi(a, b),
        net_pnl, q_eff,
        clusters=clusters, n_boot=n_boot, seed=seed,
    )

    return BacktestResult(
        n_trades=int(len(trades)),
        n_candidates=int(len(sig)),
        threshold=threshold,
        gross_roi=gross_roi,
        net_roi=net_roi,
        net_roi_lo=lo,
        net_roi_hi=hi,
        hit_rate=float(np.mean(win)),
        mean_edge=float(np.mean(np.abs(trades["edge"]))),
        cost_per_trade=costs.total,
        significant=bool(lo > 0),
    )


def threshold_sweep(df: pd.DataFrame, costs: CostModel,
                    thresholds=(0.02, 0.03, 0.05, 0.08, 0.10, 0.15),
                    n_boot: int = 2_000) -> pd.DataFrame:
    """Run the backtest across thresholds.

    This exists to make the garden of forking paths visible rather than to
    find the best setting. If the return is only positive at one threshold,
    that is a sign you have found noise, and the reader should be able to see
    that for themselves.
    """
    rows = []
    for t in thresholds:
        r = run_backtest(df, costs, threshold=t, n_boot=n_boot)
        rows.append(r.as_dict())
    return pd.DataFrame(rows)
