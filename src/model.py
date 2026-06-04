"""model.py — train a calibrated XGBoost classifier for match outcomes.

Target: 0=Away win, 1=Draw, 2=Home win.
Trained on *all* international matches since 1992 (qualifiers, continental cups,
friendlies, ...), with per-row sample weights for recency and competitiveness so
recent, high-stakes games dominate.
Evaluated with a time-respecting walk-forward backtest over recent World Cups.

Run standalone:  python model.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLS, WC_NAME, sample_weights

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_PATH = RESULTS_DIR / "model.pkl"
IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.png"
CLASSES = [0, 1, 2]


def _make_xgb(seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        min_child_weight=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


class WCPredictor:
    """Calibrated XGBoost wrapper producing 3-class match probabilities."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model: CalibratedClassifierCV | None = None
        self.importance_model: XGBClassifier | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "WCPredictor":
        # Sigmoid (Platt) calibration is robust for this multi-class problem.
        base = _make_xgb(self.seed)
        n_min = int(np.min(np.bincount(y))) if len(y) else 0
        cv = max(2, min(3, n_min))
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
        self.model.fit(X[FEATURE_COLS], y, sample_weight=sample_weight)
        # Separate plain model purely for feature-importance reporting.
        self.importance_model = _make_xgb(self.seed).fit(
            X[FEATURE_COLS], y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P in column order [Away, Draw, Home], aligned to CLASSES."""
        proba = self.model.predict_proba(X[FEATURE_COLS])
        # Align to fixed class order in case a class was unseen during fit.
        out = np.zeros((len(X), 3))
        for j, c in enumerate(self.model.classes_):
            out[:, c] = proba[:, j]
        row_sums = out.sum(axis=1, keepdims=True)
        return out / np.where(row_sums == 0, 1, row_sums)

    # --- persistence -------------------------------------------------------- #
    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "WCPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)

    # --- reporting ---------------------------------------------------------- #
    def plot_importance(self, path: Path = IMPORTANCE_PATH) -> None:
        if self.importance_model is None:
            return
        imp = self.importance_model.feature_importances_
        order = np.argsort(imp)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([FEATURE_COLS[i] for i in order], imp[order], color="#2a7ae2")
        ax.set_title("Feature importance (XGBoost gain)")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def evaluate_walk_forward(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame,
                          wc_years=(2010, 2014, 2018, 2022), seed: int = 42) -> float:
    """Walk-forward backtest: for each WC, train only on prior matches (weighted),
    test on that WC's games. Mean log-loss. This respects time order, so it actually
    measures whether recency weighting helps rather than leaking the future."""
    is_wc = meta["tournament"] == WC_NAME
    losses = []
    for year in wc_years:
        test_idx = meta.index[is_wc & (meta["year"] == year)]
        if len(test_idx) == 0:
            continue
        cutoff = meta.loc[test_idx, "date"].min()
        train_idx = meta.index[meta["date"] < cutoff]
        if len(train_idx) < 500 or len(np.unique(y[train_idx])) < 3:
            continue
        w = sample_weights(meta.loc[train_idx], ref_date=cutoff)
        clf = WCPredictor(seed=seed).fit(X.iloc[train_idx], y[train_idx], sample_weight=w)
        proba = clf.predict_proba(X.iloc[test_idx])
        losses.append(log_loss(y[test_idx], proba, labels=CLASSES))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame,
                evaluate: bool = True, seed: int = 42) -> tuple[WCPredictor, float]:
    """Backtest via walk-forward, then fit the final model on all data with recency +
    competitiveness sample weights (ref = most recent match). Saves artifacts."""
    ll = evaluate_walk_forward(X, y, meta, seed=seed) if evaluate else float("nan")
    weights = sample_weights(meta, ref_date=meta["date"].max())
    predictor = WCPredictor(seed=seed).fit(X, y, sample_weight=weights)
    predictor.save()
    predictor.plot_importance()
    return predictor, ll


if __name__ == "__main__":
    import data_loader as dl
    from feature_engineering import build_rank_map, build_value_map, make_training_set

    data = dl.clean()
    rmap = build_rank_map(data["team"])
    vmap = build_value_map(data["team"])
    X, y, meta = make_training_set(data["history"], rmap, value_map=vmap)
    print(f"Training on {len(X):,} matches ({meta['year'].min()}-{meta['year'].max()}).")
    _, ll = train_model(X, y, meta)
    print(f"Walk-forward backtest log-loss: {ll:.4f}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved importance plot -> {IMPORTANCE_PATH}")
