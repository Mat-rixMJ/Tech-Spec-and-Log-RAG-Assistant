"""
Train the churn classifier.

For each of three models — RandomForest, XGBoost, LightGBM — run a
cross-validated randomized hyperparameter search, then benchmark the tuned
models on the SAME held-out test split and ship the best by ROC AUC. Prints the
tuned params and a comparison table so the choice is evidence-based.

Run: python -m app.train_churn
  Set CHURN_TUNE=0 to skip tuning and use sensible fixed params (faster).
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .churn_data import FEATURES, load_training_frame

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"

TUNE = os.environ.get("CHURN_TUNE", "1") == "1"
N_ITER = int(os.environ.get("CHURN_TUNE_ITER", "25"))  # search budget per model
CV_FOLDS = 4


def _specs(pos_weight: float):
    """
    For each model: (base pipeline, randomized search space). Search spaces are
    on the pipeline step 'clf__<param>'. Imbalance handled per model.
    """
    return {
        "RandomForest": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(
                        class_weight="balanced", random_state=42, n_jobs=-1
                    )),
                ]
            ),
            {
                "clf__n_estimators": [200, 300, 400, 600],
                "clf__max_depth": [6, 8, 10, 14, None],
                "clf__min_samples_leaf": [1, 2, 4, 8],
                "clf__max_features": ["sqrt", "log2", None],
            },
        ),
        "XGBoost": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", XGBClassifier(
                        scale_pos_weight=pos_weight, eval_metric="logloss",
                        random_state=42, n_jobs=-1,
                    )),
                ]
            ),
            {
                "clf__n_estimators": [200, 400, 600, 800],
                "clf__max_depth": [3, 4, 5, 6, 8],
                "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.8, 0.9, 1.0],
                "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "clf__min_child_weight": [1, 3, 5],
            },
        ),
        "LightGBM": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LGBMClassifier(
                        scale_pos_weight=pos_weight, random_state=42,
                        n_jobs=-1, verbose=-1,
                    )),
                ]
            ),
            {
                "clf__n_estimators": [200, 400, 600, 800],
                "clf__max_depth": [-1, 4, 6, 8, 12],
                "clf__num_leaves": [15, 31, 63, 127],
                "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.8, 0.9, 1.0],
                "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            },
        ),
    }


def train() -> dict:
    df, source = load_training_frame()
    print(f"Training data: {source}")
    print(f"Tuning: {'on' if TUNE else 'off'} ({N_ITER} iters, {CV_FOLDS}-fold CV)\n")

    X, y = df[FEATURES], df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    pos_weight = neg / pos if pos else 1.0

    results = {}
    for name, (pipe, space) in _specs(pos_weight).items():
        if TUNE:
            search = RandomizedSearchCV(
                pipe,
                space,
                n_iter=N_ITER,
                scoring="roc_auc",
                cv=CV_FOLDS,
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            tuned = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
            cv_auc = search.best_score_
        else:
            best = pipe.fit(X_train, y_train)
            tuned, cv_auc = {}, None

        test_auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])
        results[name] = {"pipeline": best, "auc": test_auc, "cv_auc": cv_auc, "params": tuned}
        cv_str = f"cv {cv_auc:.4f} | " if cv_auc is not None else ""
        print(f"  {name:<14} {cv_str}test {test_auc:.4f}")
        if tuned:
            print(f"     best params: {tuned}")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["auc"], reverse=True)
    best_name, best = ranked[0]
    best_pipe = best["pipeline"]

    print("\nBenchmark (held-out ROC AUC):")
    for name, r in ranked:
        print(f"  {name:<14} {r['auc']:.4f}")
    print(f"\nWinner: {best_name} (AUC {best['auc']:.4f})\n")
    print(classification_report(y_test, best_pipe.predict(X_test), digits=3))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipe,
            "features": FEATURES,
            "model_name": best_name,
            "auc": round(float(best["auc"]), 4),
            "data_source": source,
            "best_params": best["params"],
        },
        MODEL_PATH,
    )
    print(f"Saved {best_name} to {MODEL_PATH}")

    assert best["auc"] > 0.75, f"Best AUC too low ({best['auc']:.3f}); training broke"
    return {"model_name": best_name, "auc": best["auc"]}


if __name__ == "__main__":
    train()
