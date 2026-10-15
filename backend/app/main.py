"""
CustIQ — Customer Intelligence API.

One app, three connected capabilities:
  • /score            — upload a customer CSV, get per-customer churn risk + why
  • /docs-upload /ask — upload policy/support docs, ask questions (RAG, cited)
  • /retention-email  — draft a grounded retention email for an at-risk customer
                        (churn factors + RAG over the policy docs)
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from . import churn_service, ingest, llm, rag, retention
from .churn_data import sample_customers_csv
from .schemas import (
    AskInput,
    AskResult,
    RetentionInput,
    RetentionResult,
    ScoreResult,
    UploadDocResult,
)
from .store import store

app = FastAPI(
    title="CustIQ — Customer Intelligence API",
    description="Churn prediction + RAG document Q&A + AI retention emails.",
    version="1.0.0",
)

_origins = os.environ.get(
    "CUSTIQ_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health():
    info = churn_service.model_info()
    return {
        "status": "ok",
        "llm_live": llm.is_live(),
        "churn_model": info["model_name"],
        "churn_auc": info["auc"],
    }


@app.get("/sample-customers.csv", tags=["churn"], response_class=PlainTextResponse)
def sample_customers():
    """A demo CSV users can download and re-upload to try /score."""
    return PlainTextResponse(
        sample_customers_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample-customers.csv"},
    )


@app.post("/score", response_model=ScoreResult, tags=["churn"])
async def score(file: UploadFile = File(...)):
    """Score an uploaded customer CSV: churn risk + top factors per customer."""
    name = (file.filename or "").lower()
    if not name.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    data = await file.read()
    try:
        df = ingest.load_csv(data)
        customers = churn_service.predict_frame(df)
    except ingest.IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    at_risk = sum(1 for c in customers if c["risk_level"] in {"medium", "high"})
    return ScoreResult(count=len(customers), at_risk=at_risk, customers=customers)


@app.get("/churn/importances", tags=["churn"])
def importances():
    """Global feature importances — what drives churn across all customers."""
    return {"feature_importances": churn_service.global_importances()}


@app.post("/docs-upload", response_model=UploadDocResult, tags=["knowledge"])
async def docs_upload(file: UploadFile = File(...)):
    """Ingest a policy/support PDF for RAG."""
    name = file.filename or "upload"
    data = await file.read()
    try:
        if name.lower().endswith(".pdf"):
            text = ingest.extract_pdf_text(data)
        elif name.lower().endswith(".txt"):
            ingest.validate_size(data)
            text = data.decode("utf-8", errors="replace")
        else:
            raise ingest.IngestError("Upload a .pdf or .txt policy document.")
        meta = store.ingest(name, text)
    except ingest.IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadDocResult(**meta)


@app.post("/ask", response_model=AskResult, tags=["knowledge"])
def ask(payload: AskInput):
    """Answer a question about an uploaded document, with sources."""
    try:
        return rag.answer(payload.doc_id, payload.question)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown doc_id — upload a doc first.")


@app.post("/retention-email", response_model=RetentionResult, tags=["retention"])
def retention_email(payload: RetentionInput):
    """Draft a grounded retention email for an at-risk customer."""
    if payload.doc_id and not store.exists(payload.doc_id):
        raise HTTPException(status_code=404, detail="Unknown doc_id — upload a doc first.")
    return retention.draft_email(payload.customer.model_dump(), payload.doc_id)
