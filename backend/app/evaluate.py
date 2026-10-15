"""
Evaluate the shipped churn model on held-out data it never trained on.

Rebuilds the exact same train/test split used in training (same seed), so the
test rows are genuinely unseen, and reports the full metric set plus a
confusion matrix. Optionally evaluates on an external CSV if one is given.

Run:
  python -m app.evaluate                 # held-out split of the training data
  python -m app.evaluate path/to/other.csv   # external Telco-schema CSV
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .churn_data import FEATURES, _map_telco, load_training_frame

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"


def _held_out():
    """The same unseen 20% the model was scored on during training."""
    df, source = load_training_frame()
    _, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["churn"]
    )
    return test, f"held-out 20% of {source}"


def _external(path: str):
    raw = pd.read_csv(path)
    # Accept either our schema or raw Telco columns.
    df = raw if "churn" in raw.columns else _map_telco(raw)
    return df, f"external file {path} ({len(df)} rows)"


def evaluate(df: pd.DataFrame, label: str) -> dict:
    bundle = joblib.load(MODEL_PATH)
    pipe, order = bundle["pipeline"], bundle["features"]

    X, y = df[order].astype(float), df["churn"].astype(int)
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    f1 = f1_score(y, pred)
    cm = confusion_matrix(y, pred)

    print(f"\n=== Evaluation: {label} ===")
    print(f"Model: {bundle['model_name']}  |  rows: {len(df)}")
    print(f"\n  Accuracy : {acc:.3f}")
    print(f"  ROC AUC  : {auc:.3f}   (headline metric for churn)")
    print(f"  Precision: {prec:.3f}   (of those flagged, how many truly churn)")
    print(f"  Recall   : {rec:.3f}   (of true churners, how many we catch)")
    print(f"  F1       : {f1:.3f}")
    print("\n  Confusion matrix")
    print("                 pred: stay   pred: churn")
    print(f"    actual stay    {cm[0, 0]:>8}     {cm[0, 1]:>8}")
    print(f"    actual churn   {cm[1, 0]:>8}     {cm[1, 1]:>8}")
    print("\n" + classification_report(y, pred, digits=3))

    return {"accuracy": acc, "roc_auc": auc, "precision": prec, "recall": rec, "f1": f1}


def main():
    if len(sys.argv) > 1:
        df, label = _external(sys.argv[1])
    else:
        df, label = _held_out()
    evaluate(df, label)


if __name__ == "__main__":
    main()
