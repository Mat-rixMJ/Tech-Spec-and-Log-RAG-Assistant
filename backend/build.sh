#!/usr/bin/env bash
# Render / Oracle VM build script.
# Installs deps, fetches the real Telco churn dataset (falls back to the
# synthetic generator if the download fails), and trains the model.
set -o errexit

pip install -r requirements.txt

# Fetch the real IBM Telco Customer Churn dataset from IBM's public repo.
# If this fails (offline, URL moved), training falls back to synthetic data.
mkdir -p data
TELCO_URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
if [ ! -f data/telco_churn.csv ]; then
  echo "Downloading real Telco churn dataset..."
  curl -fsSL "$TELCO_URL" -o data/telco_churn.csv \
    || echo "Download failed — training will use the synthetic fallback."
fi

# Lighter tuning budget on deploy keeps the build quick and memory-safe on a
# small VM. Run `python -m app.train_churn` locally for the full search
# (CHURN_TUNE_ITER=25). Override with env vars if the box has headroom.
export CHURN_TUNE_ITER="${CHURN_TUNE_ITER:-12}"
python -m app.train_churn
