# Are Polymarket prices honest probabilities?

When a prediction market quotes 20c on a contract, does that event happen
20% of the time? This repository answers that for 897 resolved binary
Polymarket markets between January 2024 and August 2026, and then asks the
follow-up that actually matters: if the prices are *not* perfectly
calibrated, is the gap big enough to trade?

**Short answer.** The market is well calibrated overall (Brier 0.077, skill
score 0.51 against the base rate). Seven days out, the slope of realised
outcomes against quoted log-odds is 0.867, and its 95% confidence interval
[0.754, 0.980] excludes 1, with mid-probability contracts underpriced. That
deviation is suggestive rather than established: it appears at one of four
horizons and vanishes when they are pooled. Trading it returns 9.2% after
costs — with a 95% interval of [−7.3%, +26.1%], which is to say: no
demonstrated edge.

The headline number is the last one. An apparent 9.2% return that cannot be
distinguished from zero is the whole point of the exercise.

---

## Running it

```bash
pip install -r requirements.txt
make test          # 40 tests, ~2s
make analyse       # full pipeline, ~90s
```

`make analyse` writes `results/results.json`, `results/report.md` and eight
figures. Every number in this README comes from that run. Python 3.11+ is
required, and scikit-learn is pinned to 1.9.1 because earlier versions split
the cross-validation folds differently, which changes the model comparison.

The dataset ships with the repo (`data/raw/markets.csv`, 3,330 rows, one of
which cleaning drops for an already-settled price of 1.0), so the analysis
reproduces offline. `make fetch` re-downloads from Polymarket's
public APIs if you want a fresher sample; it needs network access and is
rate-limited.

---

## What's in the data

| | |
| --- | --- |
| Observations | 3,329 |
| Markets | 897 resolved binary markets |
| Horizons | price observed 1, 3, 7 and 30 days before resolution |
| Period | 2024-01-10 to 2026-08-04 |
| Base rate | 20.1% of contracts resolved YES |
| Median price | 0.0055 |

That median is the single most important fact about this dataset, and it
drives two design decisions downstream. **64.3% of observations are priced
below 5c.** This is a sample dominated by deep longshots, so equal-width
probability bins would put two-thirds of the data in one bucket and leave
the reliability diagram saying nothing. Every binned analysis here uses
quantile bins instead, and the reliability diagram is rendered on both a
linear and a logit axis (`figures/05_` and `05b_`) because the longshot
region is only legible on the second.

Each market contributes up to four rows, one per horizon, and those rows
share a single outcome. They are therefore **not independent**, which is
the other decision that shapes the whole analysis — see Methods.

---

## Results

### Calibration at the 7-day horizon (n = 825)

| metric | value | 95% CI |
| --- | --- | --- |
| Brier score | 0.0769 | [0.0648, 0.0896] |
| Brier skill score | 0.508 | [0.437, 0.574] |
| Log loss | 0.2309 | — |
| Expected calibration error | 0.0321 | — |
| Maximum calibration error | 0.1223 | — |

A Brier skill score of 0.51 means the market halves the squared error of
someone who knows only the base rate. That is genuinely good.

### Murphy decomposition

Brier decomposes into three parts, and the single number hides which one is
doing the work:

| term | value | reading |
| --- | --- | --- |
| reliability | 0.0031 | miscalibration — near zero, so the prices are close to honest |
| resolution | 0.0803 | discrimination — the market separates winners from losers |
| uncertainty | 0.1563 | irreducible variance of the outcome itself |

Calibration and discrimination are different virtues. A forecaster who
always says "20%" is perfectly calibrated and completely useless; its
reliability is 0 and its resolution is also 0. Polymarket has both.

### The favourite-longshot test

Regressing outcome on logit(price) gives a slope of **0.867, 95% CI
[0.754, 0.980]**, p = 0.021 against the null that the slope is 1.

At this horizon the interval excludes 1. A slope below 1 means the realised curve is flatter than the diagonal:
mid-probability contracts resolve YES more often than their price implies.
The binned view shows where it lives — contracts priced around 5c resolved
at 15.7%, and contracts around 23c resolved at 35.4%.

**That result does not hold across the other horizons:**

| horizon | n | slope | 95% CI | p (slope = 1) |
| --- | --- | --- | --- | --- |
| 1 day | 885 | 1.178 | [0.943, 1.412] | 0.137 |
| 3 days | 887 | 1.030 | [0.861, 1.200] | 0.726 |
| 7 days | 825 | 0.867 | [0.754, 0.980] | 0.021 |
| 30 days | 732 | 0.905 | [0.774, 1.037] | 0.159 |
| all, pooled | 3,329 | 0.953 | [0.841, 1.064] | 0.406 |

Only the 7-day interval excludes 1, and a Bonferroni correction for four
horizons takes its p-value to 0.083. So the honest reading is a possible
7-day deviation, not a demonstrated bias.

**The study that produced the dataset** reported a slope of 1.01 [0.93, 1.09]
and concluded there was no measurable bias. It used the same data
(`data/raw/markets.csv` is byte-identical to its `data/sample/markets.csv`)
with a different specification: OLS of outcome on raw price, where this repo
fits a logistic regression on logit(price). Both are reproduced here at the
7-day horizon, with standard errors clustered by market:

| specification | slope | 95% CI |
| --- | --- | --- |
| OLS, outcome on price (source study) | 1.011 | [0.932, 1.090] |
| logistic, outcome on logit(price) (this repo) | 0.867 | [0.754, 0.980] |

The two slopes are different estimands, so neither contradicts the other.
Both equal 1 for a perfectly calibrated market, but they measure misses on
different scales. On the logit scale a miss is measured in log-odds, so the
~5c bin resolving at 15.7% is a large error (odds off by a factor of about
3.3), and 64% of the sample sits below 5c. On the identity scale the same bin
is off by 10 percentage points, the hundreds of sub-1c contracts contribute
almost nothing, and the fit is dominated by mid-range prices and favourites,
which sit close to the diagonal. "No slope deviation in probability" and "a
flatter-than-diagonal curve in log-odds" are both true of this data. The OLS
fit is written to `results.json` under `favourite_longshot_ols`.

### Skill decays with horizon, as it should

| horizon | n | Brier | skill score | 95% CI |
| --- | --- | --- | --- | --- |
| 1 day | 885 | 0.0497 | 0.703 | [0.652, 0.750] |
| 3 days | 887 | 0.0596 | 0.646 | [0.586, 0.700] |
| 7 days | 825 | 0.0769 | 0.508 | [0.437, 0.574] |
| 30 days | 732 | 0.0807 | 0.448 | [0.370, 0.520] |

Monotone, which is a useful sanity check: a pipeline that found 30-day
forecasts sharper than 1-day forecasts would have a bug in it.

### Can the bias be traded? No.

Walk-forward recalibration (Platt scaling fit only on outcomes already known
when each price was observed) improves the Brier score from 0.0778 to 0.0765.
Trading wherever
the recalibrated probability diverges from the price by at least 5
percentage points, paying 2c half-spread plus 0.5c slippage:

| | |
| --- | --- |
| Trades | 190 of 668 candidates |
| Gross ROI | 16.8% |
| **Net ROI** | **9.2%, 95% CI [−7.3%, +26.1%]** |
| Significant | **No** |

And across every threshold tested:

| threshold | trades | net ROI | 95% CI |
| --- | --- | --- | --- |
| 0.02 | 281 | +11.7% | [−1.6%, +25.7%] |
| 0.03 | 252 | +10.9% | [−3.9%, +25.1%] |
| 0.05 | 190 | +9.2% | [−7.8%, +25.7%] |
| 0.08 | 109 | +23.5% | [−1.0%, +47.3%] |
| 0.10 | 54 | +8.1% | [−25.5%, +43.3%] |
| 0.15 | 1 | −100.0% | single trade, no interval |

Every interval crosses zero. (The 0.15 threshold fires once, so it has no
interval worth the name; its bootstrap is the same lost trade resampled.) The
point estimates are positive down to 0.10, which is mildly suggestive, but
"positive point estimate" is not a finding — and the way the apparent return
jumps around as the sample shrinks, from +9.2% to +23.5% to +8.1%, is the
signature of noise, not edge. The sweep is in the repo precisely so this is
visible rather than buried.

---

## Can any model beat the price?

Predicting the outcome *from* the price is near-trivial, because the price
is already a forecast. The honest question is whether any feature beyond
the price adds information the market missed. So the market price is the
baseline, and a model only counts if it beats that.

Features offered to the models: `logit_price`, `days_to_resolution`,
`log_volume`, `is_longshot`, `is_favourite`, `month_index`,
`price_x_horizon`. Validation is 5-fold `StratifiedGroupKFold` grouped by
market, with `GridSearchCV` inside.

| model | Brier | ROC AUC | macro F1 | skill vs market |
| --- | --- | --- | --- | --- |
| market + Platt scaling | 0.0654 | 0.960 | 0.852 | **+0.007** |
| **market price (baseline)** | **0.0659** | **0.960** | **0.841** | **0.000** |
| logistic regression | 0.0662 | 0.958 | 0.849 | −0.004 |
| random forest | 0.0667 | 0.958 | 0.849 | −0.012 |
| gradient boosting | 0.0769 | 0.945 | 0.814 | −0.167 |
| base-rate dummy | 0.1604 | 0.499 | 0.444 | −1.434 |

**Nothing beats the price.** Every learned model is worse than simply
reading the quote, though the best get within about 1%, and the only thing
that edges ahead is a two-parameter recalibration of the price itself, by
0.7%.

That is the result, and it is not a disappointing one. It says this slice
of Polymarket is informationally efficient with respect to the features
available here: volume, timing and horizon carry nothing the price has not
already absorbed. Permutation importance agrees — `logit_price` scores
0.180 and no other feature scores meaningfully above zero.

A model that *had* beaten the price by a wide margin on 897 markets with
seven features would have been evidence of a leak, not a discovery.

---

## Methods, and the two decisions that matter

**Clustered inference.** Each market appears at up to four horizons, sharing
one outcome. Resampling rows would pretend there are 3,329 independent
observations when there are 897. Every bootstrap in this repo resamples
whole markets, and the favourite-longshot regression uses cluster-robust
standard errors. This matters for pooled statistics; at a single horizon each
market is one row, so clustering there reduces to ordinary robust errors. `test_clustered_intervals_are_wider_than_naive_ones`
demonstrates the difference on synthetic data where the truth is known.

**No lookahead.** The recalibration is fit on an expanding window of markets
that had already resolved when the price was observed, refit every 25 new
outcomes, with a 150-market burn-in. Closing earlier is not enough: a market
that closes two days before this one was still open when this one's 7-day
price was quoted. Fitting on the whole sample and scoring on the whole sample
would manufacture an edge out of nothing. `test_walk_forward_has_no_lookahead`
and `test_walk_forward_ignores_outcomes_unknown_at_forecast_time` enforce this
directly: they mutate outcomes that were unknown at forecast time and assert
that the prediction does not move.

Smaller choices, each with a reason in the code: Wilson intervals on the
per-bin frequencies rather than normal ones, because several bins sit at a
realised frequency of zero where the normal interval runs negative;
quantile bins rather than equal-width, for the skew described above;
pooled return (total profit over total capital staked) rather than an average
of per-trade returns, so that a 0.5c longshot's 200x multiple cannot swamp the
figure; unweighted classes in the models, because reweighting distorts the
probabilities that Brier scores; permutation importance
rather than a tree's built-in `feature_importances_`, which is biased
towards high-cardinality continuous features and would have handed the
crown to `logit_price` for the wrong reason.

### A note on the Murphy identity

`BS = reliability − resolution + uncertainty` is exact only for forecasts
taking finitely many values. Market prices are effectively continuous, so
the identity holds for the *binned* forecast, not the raw one. The code
reports both, plus `binning_error` — the within-bin information that
binning discards (−0.0022 here, negative because coarsening makes the score
worse). A decomposition that reported a zero residual against the raw Brier
would be quietly wrong, and the test suite checks the identity against the
right quantity.

---

## Limitations

- **The `category` field in this sample is entirely `unknown`**, so the
  analysis cannot slice by market type. Politics, sports and crypto almost
  certainly calibrate differently, and that is the first thing I would
  check with richer data.
- **The data is a derived sample** (price, outcome, horizon, volume) from a
  published open-source study, not a pull I made myself. Provenance is in
  `data/README.md`. The pipeline is my own; the data collection is not.
- **897 markets is not many** for subgroup analysis. The horizon table is
  about as fine as this sample can be cut before the intervals swallow the
  estimates.
- **`log_volume` is not point-in-time.** The dataset has each market's final
  volume, the same at every horizon, so this feature contains information from
  after the price was quoted. That can only flatter the models, which still
  lose to the price.
- **Hyperparameter search is not nested.** Parameters are chosen on the same
  folds that score them, another bias in the models' favour.
- **Survivorship is unexamined.** Which markets ended up in the sample, and
  whether unresolved or delisted markets differ systematically, is not
  something this data can answer.
- **The cost model is a constant.** Real half-spreads vary enormously with
  liquidity, and a 2c assumption is optimistic on a thin longshot book.

## Next

Pull the Gamma and CLOB APIs directly for category and order-book depth,
then test whether the slope deviation survives inside market types and at
realistic per-market spreads. If it does, it is interesting. If it does not,
the efficient-market reading stands.

---

## Layout

```
src/pmcal/
  io.py            loading, validation
  features.py      cleaning, feature engineering, leak-safe matrices
  eda.py           summaries and exploratory figures
  metrics.py       Brier, log loss, Murphy decomposition, clustered bootstrap
  calibration.py   reliability bins, favourite-longshot test, walk-forward Platt
  backtest.py      divergence signal, transaction costs, threshold sweep
  models.py        model zoo, grouped CV, permutation importance
  plots.py         figures
  cli.py           orchestration
tests/             40 known-answer tests
data/raw/          the dataset
results/           generated: results.json, report.md
figures/           generated: 8 PNGs
```

Arun Yadav — [github.com/yadavaruncs](https://github.com/yadavaruncs)
