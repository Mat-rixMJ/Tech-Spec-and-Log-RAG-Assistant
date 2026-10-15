"""
Synthetic customer-churn dataset generator (generic SaaS subscription).

Reproducible (fixed seed), needs no download or license, and the churn signal
is realistic: driven by tenure, plan, monthly charges and support contacts.
Used by both the training script and the tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "contract_type",  # 0 = monthly, 1 = annual, 2 = two-year
    "has_premium_support",  # 0/1
    "num_features_used",  # add-on features used, 0..6
]

CONTRACT_LABELS = {0: "Monthly", 1: "Annual", 2: "Two-year"}


def make_churn_frame(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Build a labelled churn dataset. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, size=n)
    monthly = rng.normal(70, 25, size=n).clip(18, 150)
    total = (tenure * monthly * rng.normal(1.0, 0.1, size=n)).clip(0, None)
    support = rng.poisson(1.2, size=n).clip(0, 10)
    contract = rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])
    premium = rng.integers(0, 2, size=n)
    features_used = rng.integers(0, 7, size=n)

    # Short tenure, monthly plans, high charges and many support calls push
    # churn up; premium support and heavier feature use pull it down.
    logit = (
        2.2
        - 0.045 * tenure
        - 0.9 * contract
        + 0.012 * monthly
        + 0.35 * support
        - 0.6 * premium
        - 0.05 * features_used
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly.round(2),
            "total_charges": total.round(2),
            "support_calls": support,
            "contract_type": contract,
            "has_premium_support": premium,
            "num_features_used": features_used,
            "churn": churn,
        }
    )


import os
from pathlib import Path

# Where the real IBM Telco Customer Churn CSV goes, if you choose to use it.
# Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# (file "WA_Fn-UseC_-Telco-Customer-Churn.csv"), save it here.
TELCO_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"


def _map_telco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the real IBM Telco columns onto our 7 features + churn. Only the columns
    we model are kept; the mapping mirrors the synthetic schema so the model,
    API and SHAP code are unchanged regardless of data source.
    """
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    yes_no = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}

    # Internet add-on services we count toward "features used".
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "StreamingTV", "StreamingMovies", "MultipleLines",
    ]
    present = [c for c in service_cols if c in df.columns]
    num_features = sum(df[c].map(yes_no).fillna(0) for c in present) if present else 0

    total = pd.to_numeric(df["TotalCharges"], errors="coerce")

    out = pd.DataFrame(
        {
            "tenure_months": pd.to_numeric(df["tenure"], errors="coerce"),
            "monthly_charges": pd.to_numeric(df["MonthlyCharges"], errors="coerce"),
            "total_charges": total,
            "support_calls": 0,  # Telco has no call-count column; neutral default.
            "contract_type": df["Contract"].map(contract_map),
            "has_premium_support": df["TechSupport"].map(yes_no).fillna(0),
            "num_features_used": num_features,
            "churn": df["Churn"].map(yes_no),
        }
    )
    # Drop rows with any missing modelled value (Telco has ~11 blank TotalCharges).
    return out.dropna().reset_index(drop=True).astype(
        {f: "float64" for f in FEATURES} | {"churn": "int64"}
    )


def load_training_frame() -> tuple[pd.DataFrame, str]:
    """
    Return (frame, source). Uses the real Telco CSV when present (and not
    disabled via CHURN_USE_SYNTHETIC=1), else the reproducible synthetic set.
    """
    use_synthetic = os.environ.get("CHURN_USE_SYNTHETIC", "0") == "1"
    if TELCO_PATH.exists() and not use_synthetic:
        try:
            frame = _map_telco(pd.read_csv(TELCO_PATH))
            if len(frame) > 100:  # sanity: a real load, not a header-only file
                return frame, f"IBM Telco Customer Churn ({len(frame)} rows)"
        except Exception as exc:  # malformed CSV → fall back rather than crash
            print(f"Telco CSV found but could not be loaded ({exc}); using synthetic.")
    return make_churn_frame(), "synthetic (reproducible, seed=42)"


def sample_customers_csv(n: int = 25, seed: int = 7) -> str:
    """A small demo CSV (with names) users can download and re-upload."""
    df = make_churn_frame(n=n, seed=seed).drop(columns=["churn"])
    rng = np.random.default_rng(seed)
    first = ["Ava", "Liam", "Noah", "Mia", "Zoe", "Kai", "Ivy", "Leo", "Ada", "Sam"]
    last = ["Patel", "Khan", "Rao", "Singh", "Bose", "Nair", "Das", "Shah"]
    names = [
        f"{rng.choice(first)} {rng.choice(last)}" for _ in range(len(df))
    ]
    df.insert(0, "customer_id", [f"C{1000 + i}" for i in range(len(df))])
    df.insert(1, "name", names)
    return df.to_csv(index=False)
