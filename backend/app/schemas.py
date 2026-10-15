from __future__ import annotations

from pydantic import BaseModel, Field


class Factor(BaseModel):
    feature: str
    direction: str
    weight: float


class CustomerScore(BaseModel):
    customer_id: str
    name: str
    churn_probability: float
    risk_level: str
    contract_label: str
    top_factors: list[Factor]
    features: dict


class ScoreResult(BaseModel):
    count: int
    at_risk: int  # how many are medium/high risk
    customers: list[CustomerScore]


class UploadDocResult(BaseModel):
    doc_id: str
    name: str
    num_chunks: int


class AskInput(BaseModel):
    doc_id: str
    question: str = Field(..., min_length=1, max_length=1000)


class Source(BaseModel):
    text: str
    score: float


class AskResult(BaseModel):
    answer: str
    sources: list[Source]
    llm_live: bool


class RetentionInput(BaseModel):
    # The at-risk customer, as returned by /score. doc_id is optional — with a
    # policy doc the email is grounded in it; without, it offers a call.
    customer: CustomerScore
    doc_id: str | None = None


class RetentionResult(BaseModel):
    email: str
    grounded_in: list[Source]
    llm_live: bool
