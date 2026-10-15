"""
Churn prediction service: loads the trained pipeline and turns customer rows
into predictions plus a per-customer "why".

Explainability: SHAP (TreeExplainer). SHAP gives signed, per-customer feature
attributions — for THIS customer, how much did each feature push their churn
probability up or down — not just global importances. That's the honest,
industry-standard answer to "why is this customer at risk".

The pipeline scales features before the tree model, so we explain on the scaled
matrix and report the results against the original feature names.

ponytail: TreeExplainer is built once over a background sample and cached.
Works for RandomForest / XGBoost / LightGBM (whichever won the benchmark).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from .churn_data import CONTRACT_LABELS, FEATURES, load_training_frame

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"

READABLE = {
    "tenure_months": "tenure",
    "monthly_charges": "monthly charges",
    "total_charges": "total charges",
    "support_calls": "support calls",
    "contract_type": "contract type",
    "has_premium_support": "premium support",
    "num_features_used": "features used",
}


@lru_cache(maxsize=1)
def _bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m app.train_churn"
        )
    return joblib.load(MODEL_PATH)


def _load():
    b = _bundle()
    return b["pipeline"], b["features"]


def model_info() -> dict:
    b = _bundle()
    return {
        "model_name": b.get("model_name", "unknown"),
        "auc": b.get("auc"),
        "data_source": b.get("data_source", "unknown"),
    }


@lru_cache(maxsize=1)
def _explainer():
    """
    SHAP TreeExplainer over the fitted classifier, with a background sample of
    scaled training data. Built once (SHAP setup is the expensive part).
    """
    pipe, order = _load()
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    frame, _ = load_training_frame()
    bg = frame.sample(min(200, len(frame)), random_state=0)[order]
    bg_scaled = scaler.transform(bg)
    return shap.TreeExplainer(clf, bg_scaled), scaler, order


def _risk_level(proba: float) -> str:
    return "high" if proba >= 0.66 else "medium" if proba >= 0.33 else "low"


def global_importances() -> list[dict]:
    pipe, features = _load()
    importances = pipe.named_steps["clf"].feature_importances_
    pairs = sorted(zip(features, importances), key=lambda kv: kv[1], reverse=True)
    return [{"feature": READABLE[f], "importance": round(float(v), 4)} for f, v in pairs]


def _shap_churn_values(X_scaled: np.ndarray) -> np.ndarray:
    """
    Return a (n_samples, n_features) array of SHAP values for the churn (positive)
    class. Different SHAP/model versions return slightly different shapes, so we
    normalize them here.
    """
    explainer, _, _ = _explainer()
    sv = explainer.shap_values(X_scaled)
    # Binary classifiers may return a list [class0, class1] or a 3-D array.
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes)
        sv = sv[:, :, -1]
    return sv


def _factors_from_shap(shap_row: np.ndarray, order: list[str], k: int = 3) -> list[dict]:
    """Top-k features by absolute SHAP contribution for one customer."""
    pairs = sorted(
        zip(order, shap_row), key=lambda kv: abs(kv[1]), reverse=True
    )
    out = []
    for feat, val in pairs[:k]:
        out.append(
            {
                "feature": READABLE[feat],
                "direction": "increases risk" if val > 0 else "lowers risk",
                "weight": round(float(abs(val)), 4),
            }
        )
    return out


def _rows_to_results(df: pd.DataFrame, order: list[str]) -> list[dict]:
    pipe, _ = _load()
    scaler = pipe.named_steps["scaler"]

    X = df[order].astype(float)
    probas = pipe.predict_proba(X)[:, 1]
    shap_vals = _shap_churn_values(scaler.transform(X))

    results = []
    for i, (_, raw) in enumerate(df.iterrows()):
        proba = float(probas[i])
        results.append(
            {
                "customer_id": str(raw.get("customer_id", f"row-{i + 1}")),
                "name": str(raw.get("name", "")),
                "churn_probability": round(proba, 4),
                "risk_level": _risk_level(proba),
                "contract_label": CONTRACT_LABELS.get(int(raw["contract_type"]), "Unknown"),
                "top_factors": _factors_from_shap(shap_vals[i], order),
                "features": {f: float(raw[f]) for f in order},
            }
        )
    return results


def predict_one(features: dict) -> dict:
    _, order = _load()
    df = pd.DataFrame([[features[f] for f in order]], columns=order)
    return _rows_to_results(df, order)[0]


def predict_frame(df: pd.DataFrame) -> list[dict]:
    """
    Score every row of an uploaded customer CSV, with SHAP explanations.
    Missing feature columns raise ValueError (caller maps to HTTP 400).
    """
    _, order = _load()
    missing = [c for c in order if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    results = _rows_to_results(df, order)
    results.sort(key=lambda r: r["churn_probability"], reverse=True)
    return results
