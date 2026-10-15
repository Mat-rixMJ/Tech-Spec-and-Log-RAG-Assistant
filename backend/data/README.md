# Training data

The churn model trains on one of two sources, chosen automatically at train time:

1. **Real — IBM Telco Customer Churn** (used if present). Download
   `WA_Fn-UseC_-Telco-Customer-Churn.csv` from
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   and save it here as `telco_churn.csv`. ~7,000 real customers. It's
   `.gitignore`d, so it's never committed.

2. **Synthetic fallback** (default). If no `telco_churn.csv` is present, the app
   generates a reproducible synthetic dataset (`app/churn_data.py`, fixed seed)
   whose schema mirrors the Telco dataset. Zero-friction and deterministic.

The training script prints which source it used, and `/health` reports it
(`churn_model`, `churn_auc`, `data_source`). To force synthetic even when the
CSV is present: set `CHURN_USE_SYNTHETIC=1`.

Both sources map to the same 7 features, so the model, API and SHAP code are
identical regardless of source.
