# CustIQ — Customer Intelligence

A deployable product that combines three things most churn demos keep separate:

1. **Churn prediction** — upload a customer CSV, get each customer's churn risk
   with a per-customer SHAP explanation of *why*.
2. **RAG document Q&A** — upload policy/support docs and ask questions; answers
   cite their source passages (LangChain + Gemini).
3. **AI retention emails** — for an at-risk customer, draft a personalized email
   **grounded** in your policy docs, so it only offers what the policy allows.

This is a monorepo:

| Folder | What | Stack |
| --- | --- | --- |
| `backend/` | FastAPI API: churn + RAG + retention | FastAPI · LangChain · scikit-learn · XGBoost · LightGBM · SHAP · Gemini |
| `frontend/` | React UI | React · Vite · Tailwind · Recharts |

## Model

Trained on the real **IBM Telco Customer Churn** dataset (7,032 customers, auto-
downloaded at build time; reproducible synthetic fallback). The training script
runs cross-validated hyperparameter search over RandomForest, XGBoost and
LightGBM and ships the best. 5-fold CV ROC AUC ≈ **0.84** (all three within
~0.002 — the dataset's ceiling), recall ≈ 0.80.

## Run locally

```bash
# Backend  → http://127.0.0.1:8300  (docs at /docs)
cd backend
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m app.train_churn        # or ./build.sh to fetch real data
./.venv/bin/uvicorn app.main:app --reload --port 8300

# Frontend → http://localhost:5173
cd ../frontend
npm install && npm run dev
```

Without a `GEMINI_API_KEY`, RAG and emails run in demo mode (show retrieved
context) so everything works offline.

## Deploy

See **[DEPLOY-ORACLE.md](./DEPLOY-ORACLE.md)** — Nginx + systemd on an Oracle
Cloud free VM (ARM-aware), serving both parts from one origin.

## Test

```bash
cd backend && pytest           # 10 tests
cd frontend && npm test        # 3 tests
```
