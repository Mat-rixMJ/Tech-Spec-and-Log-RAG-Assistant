"""
Retention-email drafting — the feature that ties churn + RAG together.

For an at-risk customer, we take their churn risk and top contributing factors,
retrieve relevant passages from the company's uploaded retention/policy docs
(RAG), and ask the LLM to draft a short, personalized retention email that is
*grounded* in those passages (e.g. only offers a discount the policy actually
allows). This is the "useful product" story: the prediction and the knowledge
base drive a concrete next action.
"""

from __future__ import annotations

from . import llm
from .store import store

# What we search the knowledge base for, per contributing factor. Keeps the
# retrieved context relevant to why this specific customer is at risk.
FACTOR_QUERIES = {
    "monthly charges": "pricing discount loyalty offer billing",
    "support calls": "support priority escalation help response time",
    "tenure": "loyalty long-term customer retention reward",
    "contract type": "annual plan contract upgrade incentive",
    "premium support": "premium support upgrade benefits",
    "features used": "onboarding feature adoption training getting started",
    "total charges": "billing credit discount account value",
}

PROMPT_TEMPLATE = """You are a customer-success manager writing a short, warm retention
email to an at-risk customer. Keep it under 150 words, friendly and specific — no
corporate filler.

Customer: {name}
Churn risk: {risk} ({pct}%)
Main reasons they may leave: {reasons}

You may ONLY offer things supported by the company policy context below. Do not
invent discounts, refunds or guarantees that aren't in the context. If the context
doesn't support a concrete offer, acknowledge their concern and offer a call instead.

CONTEXT:
{context}

Write the email now (start with "Hi {name},"):
"""


def _retrieval_query(top_factors: list[dict]) -> str:
    parts = []
    for f in top_factors:
        if f["direction"] == "increases risk":
            parts.append(FACTOR_QUERIES.get(f["feature"], f["feature"]))
    return " ".join(parts) or "customer retention loyalty offer"


def draft_email(customer: dict, doc_id: str | None) -> dict:
    """
    customer: a row from churn_service.predict_frame (name, risk_level,
    churn_probability, top_factors). doc_id: an ingested policy doc, or None.
    """
    reasons = ", ".join(
        f["feature"] for f in customer["top_factors"] if f["direction"] == "increases risk"
    ) or "general engagement"

    sources: list[dict] = []
    if doc_id and store.exists(doc_id):
        sources = store.retrieve(doc_id, _retrieval_query(customer["top_factors"]), k=3)

    context = (
        "\n\n---\n\n".join(s["text"] for s in sources)
        if sources
        else "(No policy document provided — offer a call, do not promise specifics.)"
    )

    name = customer.get("name") or "there"

    if not llm.is_live():
        # Offline demo: write a clean templated email rather than echoing the
        # prompt scaffolding, so the UI shows something presentable without a key.
        email = _demo_email(name, reasons, bool(sources))
    else:
        prompt = PROMPT_TEMPLATE.format(
            name=name,
            risk=customer["risk_level"],
            pct=round(customer["churn_probability"] * 100),
            reasons=reasons,
            context=context,
        )
        email = llm.generate(prompt)

    return {
        "email": email,
        "grounded_in": sources,
        "llm_live": llm.is_live(),
    }


def _demo_email(name: str, reasons: str, has_policy: bool) -> str:
    """Presentable placeholder email for demo mode (no LLM key)."""
    offer = (
        "I'd love to find a plan that works better for you — do you have 15 minutes "
        "this week for a quick call?"
        if not has_policy
        else "Based on your account, you may qualify for options in our loyalty "
        "program — I'd be glad to walk you through them on a short call."
    )
    return (
        f"Hi {name},\n\n"
        f"I noticed a few things about your account ({reasons}) and wanted to reach "
        f"out personally. {offer}\n\n"
        "Thanks for being with us — I'm here to help.\n\n"
        "Best regards,\nThe Customer Success Team\n\n"
        "— (Demo mode: set a GEMINI_API_KEY to generate a tailored, policy-grounded email.)"
    )
