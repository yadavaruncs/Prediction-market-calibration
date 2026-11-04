"""Can anything beat the market price?

The modelling question here is deliberately narrow. Predicting a binary
outcome from a market price is close to trivial -- the price is already a
forecast, so a model handed `logit_price` will look excellent and will have
learned nothing. The interesting question is whether features *beyond* the
price carry information the market missed.

So the market price alone is the baseline, and a model only counts as a
result if it beats that. If nothing does, the finding is that this slice of
Polymarket is informationally efficient, which is a real answer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import BASELINE_COLUMNS, FEATURE_COLUMNS, feature_matrix

RANDOM_STATE = 20260301


def make_cv(n_splits: int = 5):
    """Stratified *and* grouped.

    Stratified because the base rate is about 20% and unstratified folds
    would vary a lot. Grouped by market_id because the same market appears
    at several horizons, and letting those rows straddle a fold boundary
    leaks the outcome into training.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=RANDOM_STATE)


def model_zoo() -> dict:
    """Candidate models and the grids searched over each.

    Grids are small on purpose: with ~900 markets and 5 folds, a large grid
    would mostly be fitting noise in the validation score. class_weight is
    balanced throughout because of the 20/80 split.
    """
    return {
        "logistic_regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced_subsample",
                                   random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [300], "max_depth": [4, 8, None],
             "min_samples_leaf": [5, 20]},
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {"max_depth": [3, 6], "learning_rate": [0.05, 0.1],
             "max_iter": [200], "min_samples_leaf": [20]},
        ),
    }


@dataclass
class ModelScores:
    name: str
    n_features: int
    brier: float
    brier_std: float
    log_loss: float
    roc_auc: float
    roc_auc_std: float
    macro_f1: float
    accuracy: float
    best_params: dict

    def as_dict(self) -> dict:
        return asdict(self)


def _oof_predict(estimator, X, y, groups, cv) -> np.ndarray:
    """Out-of-fold probabilities, honouring the group constraint.

    sklearn's cross_val_predict does not accept `groups` for every splitter
    version, so the loop is written out. It also makes the leak-avoidance
    explicit for anyone reading the code.
    """
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        est = _clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        oof[test_idx] = est.predict_proba(X[test_idx])[:, 1]
    return oof


def _clone(estimator):
    from sklearn.base import clone
    return clone(estimator)


def _score(name: str, y, p, n_features: int, best_params: dict,
           fold_briers=None, fold_aucs=None) -> ModelScores:
    pred = (p >= 0.5).astype(int)
    return ModelScores(
        name=name,
        n_features=n_features,
        brier=float(brier_score_loss(y, p)),
        brier_std=float(np.std(fold_briers)) if fold_briers is not None else float("nan"),
        log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        roc_auc=float(roc_auc_score(y, p)),
        roc_auc_std=float(np.std(fold_aucs)) if fold_aucs is not None else float("nan"),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y, pred)),
        best_params=best_params,
    )


def _fold_stats(y, p, X, groups, cv):
    briers, aucs = [], []
    for _, test_idx in cv.split(X, y, groups=groups):
        yt, pt = y[test_idx], p[test_idx]
        briers.append(brier_score_loss(yt, pt))
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt, pt))
    return briers, aucs


def compare_models(df: pd.DataFrame, n_splits: int = 5,
                   verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fit the zoo, score everything out-of-fold, return a comparison table.

    Three reference points are included before any model:
      dummy_prior    always predicts the training base rate
      market_price   the raw market price, untouched
      market_platt   the market price after a single logistic recalibration
    """
    X_all, y, groups = feature_matrix(df, FEATURE_COLUMNS)
    X_base, _, _ = feature_matrix(df, BASELINE_COLUMNS)
    cv = make_cv(n_splits)

    results: list[ModelScores] = []
    artefacts: dict = {"oof": {}, "features": list(FEATURE_COLUMNS)}

    # --- reference 1: the uninformative prior ---------------------------
    dummy = DummyClassifier(strategy="prior")
    p_dummy = _oof_predict(dummy, X_base, y, groups, cv)
    fb, fa = _fold_stats(y, p_dummy, X_base, groups, cv)
    results.append(_score("dummy_prior", y, p_dummy, 0, {}, fb, fa))
    artefacts["oof"]["dummy_prior"] = p_dummy

    # --- reference 2: the market itself ---------------------------------
    p_market = df["price"].to_numpy(float)
    fb, fa = _fold_stats(y, p_market, X_base, groups, cv)
    results.append(_score("market_price", y, p_market, 1, {}, fb, fa))
    artefacts["oof"]["market_price"] = p_market

    # --- reference 3: the market, recalibrated --------------------------
    platt = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000,
                                                 random_state=RANDOM_STATE))])
    p_platt = _oof_predict(platt, X_base, y, groups, cv)
    fb, fa = _fold_stats(y, p_platt, X_base, groups, cv)
    results.append(_score("market_platt", y, p_platt, 1, {}, fb, fa))
    artefacts["oof"]["market_platt"] = p_platt

    # --- the candidate models -------------------------------------------
    for name, (estimator, grid) in model_zoo().items():
        if verbose:
            print(f"  fitting {name} ...", flush=True)
        search = GridSearchCV(
            estimator, grid, cv=cv, scoring="neg_brier_score",
            n_jobs=-1, refit=True,
        )
        search.fit(X_all, y, groups=groups)
        best = search.best_estimator_
        p = _oof_predict(best, X_all, y, groups, cv)
        fb, fa = _fold_stats(y, p, X_all, groups, cv)
        results.append(_score(name, y, p, X_all.shape[1],
                              dict(search.best_params_), fb, fa))
        artefacts["oof"][name] = p
        artefacts.setdefault("fitted", {})[name] = best

    table = pd.DataFrame([r.as_dict() for r in results])

    # Skill relative to the market, not to the dummy. Positive means the
    # model found something the price did not already contain.
    market_brier = float(table.loc[table.name == "market_price", "brier"].iloc[0])
    table["skill_vs_market"] = 1 - table["brier"] / market_brier
    table = table.sort_values("brier").reset_index(drop=True)

    artefacts["market_brier"] = market_brier
    artefacts["y"] = y
    artefacts["groups"] = groups
    return table, artefacts


def feature_importance(df: pd.DataFrame, artefacts: dict,
                       model_name: str = "gradient_boosting",
                       n_repeats: int = 30) -> pd.DataFrame:
    """Permutation importance on held-out folds.

    Permutation importance is used rather than a tree's internal
    feature_importances_ because the internal version is biased towards
    high-cardinality continuous features, and `logit_price` would dominate
    for the wrong reason. SHAP would give per-observation attributions and
    is the natural next step; permutation answers the question asked here
    -- which feature, if scrambled, actually costs predictive accuracy --
    with no extra dependency.
    """
    fitted = artefacts.get("fitted", {}).get(model_name)
    if fitted is None:
        raise ValueError(f"no fitted model named {model_name!r}")

    X, y, groups = feature_matrix(df, FEATURE_COLUMNS)
    cv = make_cv()
    train_idx, test_idx = next(iter(cv.split(X, y, groups=groups)))

    est = _clone(fitted)
    est.fit(X[train_idx], y[train_idx])
    r = permutation_importance(
        est, X[test_idx], y[test_idx],
        scoring="neg_brier_score", n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
