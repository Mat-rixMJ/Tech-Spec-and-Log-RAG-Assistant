# CustIQ — Customer Intelligence API

One app, three connected capabilities:

1. **Churn prediction** — upload a customer CSV, get per-customer churn risk with
   an explanation of *why* (feature importance).
2. **RAG document Q&A** — upload policy/support docs and ask questions; answers
   cite the source passages.
3. **AI retention emails** — for an at-risk customer, draft a personalized email
   that is **grounded** in your policy docs (it only offers what the policy
   allows). This is where churn + RAG work together.

**Live demo:** _coming soon_ · **API docs:** `/docs`

> Hosted on Render's free tier, which sleeps when idle — the first request may
> take ~30 seconds to wake.

## Why it's built this way

- **Churn model:** the training script runs **4-fold cross-validated randomized
  hyperparameter search** (`RandomizedSearchCV`) over **RandomForest, XGBoost and
  LightGBM**, then benchmarks the tuned models on a held-out split and ships the
  best by ROC AUC (~0.83 on the real Telco data). The winner's tuned params are
  saved in the artifact. `/health` reports the live model and score. Set
  `CHURN_TUNE=0` to skip tuning for a fast run.
- **Explainability:** **SHAP** (TreeExplainer) gives per-customer attributions —
  for *this* customer, how much each feature pushed their churn risk up or down —
  which powers the "why they're at risk" list and the retention email.
- **Training data:** uses the real **IBM Telco Customer Churn** dataset when
  present (`data/telco_churn.csv`), else a reproducible synthetic fallback with
  the same schema. See `data/README.md`.
- **RAG:** TF-IDF retrieval over uploaded docs feeds a **LangChain** generation
  chain (`ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`) in
  `app/rag.py` / `app/llm.py`, defaulting to Google Gemini. No GPU, no vector
  DB — it fits a small VM and stays live.
- **Grounded emails:** `app/retention.py` turns a customer's top risk factors
  into a retrieval query, pulls matching policy passages, and asks the LLM to
  write within them. Without a doc it offers a call instead of inventing offers.
- **No key? Still runs.** With no `GEMINI_API_KEY`, RAG and emails return the
  retrieved context (demo mode), so the app and tests work offline.

## Tech

FastAPI · LangChain · scikit-learn · XGBoost · LightGBM · SHAP · pandas · pypdf
· TF-IDF retrieval · Google Gemini.

## Run locally

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m app.train_churn          # builds models/churn_model.joblib
# Optional: cp .env.example .env and add GEMINI_API_KEY for real generation
uvicorn app.main:app --reload --port 8300
```

Open http://127.0.0.1:8300/docs.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/sample-customers.csv` | Download a demo CSV to try `/score` |
| POST | `/score` | Score an uploaded customer CSV (risk + factors) |
| GET | `/churn/importances` | Global feature importances |
| POST | `/docs-upload` | Ingest a policy PDF/TXT for RAG |
| POST | `/ask` | Ask a question about a doc (cited answer) |
| POST | `/retention-email` | Draft a grounded retention email |
| GET | `/health` | Liveness + whether the LLM is live |

## Model performance (real IBM Telco data, 7,032 customers)

The training script tunes and benchmarks three models; here is their
generalization measured by **5-fold cross-validation** (every customer tested
as unseen), after hyperparameter tuning:

| Model | Mean ROC AUC | Std | Recall (churn) |
| --- | --- | --- | --- |
| **RandomForest** (shipped) | **0.839** | 0.004 | 0.800 |
| XGBoost | 0.838 | 0.004 | 0.802 |
| LightGBM | 0.837 | 0.004 | 0.769 |

The three are within ~0.002 AUC — effectively tied at the dataset's known
ceiling (~0.84). RandomForest wins on mean AUC and ships; the choice really
comes down to the recall/stability trade-off. Held-out test set (1,407 unseen
rows): AUC **0.833**, accuracy 0.733, **recall 0.805** — the model is tuned for
recall because, for retention, catching churners matters more than avoiding
false alarms.

Re-run these numbers anytime:

```bash
python -m app.evaluate        # full metrics on the held-out test set
```

## Test

```bash
pytest
```

Covers CSV scoring and risk ranking, the missing-column guard, RAG Q&A (LLM
mocked), and the retention-email integration (churn factors + retrieved policy
context both reaching the prompt).

## Deploy to Render (free)

Includes `render.yaml` and `build.sh` (trains the model on deploy). In Render:
**New → Blueprint**, point at this repo. Set `GEMINI_API_KEY` (secret) and
`CUSTIQ_CORS_ORIGINS` (your Netlify URL) in the dashboard.
