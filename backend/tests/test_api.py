"""
CustIQ API tests — the runnable checks behind the three capabilities and the
feature that connects them.

Run: pytest
"""

from fastapi.testclient import TestClient

from app import rag, retention
from app.churn_data import sample_customers_csv
from app.main import app

client = TestClient(app)


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "llm_live" in body


def test_score_sample_csv_ranks_by_risk():
    csv = sample_customers_csv()
    resp = client.post(
        "/score", files={"file": ("customers.csv", csv.encode(), "text/csv")}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["count"] > 0
    assert data["at_risk"] <= data["count"]
    # Results must be sorted by churn probability, highest first.
    probs = [c["churn_probability"] for c in data["customers"]]
    assert probs == sorted(probs, reverse=True)
    # Each customer carries an explanation.
    assert all(len(c["top_factors"]) == 3 for c in data["customers"])


def test_score_rejects_non_csv():
    resp = client.post("/score", files={"file": ("x.txt", b"hi", "text/plain")})
    assert resp.status_code == 400


def test_score_rejects_csv_missing_feature_columns():
    bad = b"customer_id,name\nC1,Ava\n"
    resp = client.post("/score", files={"file": ("bad.csv", bad, "text/csv")})
    assert resp.status_code == 400
    assert "missing required columns" in resp.json()["detail"].lower()


def test_importances_sorted_desc():
    imps = client.get("/churn/importances").json()["feature_importances"]
    assert len(imps) == 7
    weights = [i["importance"] for i in imps]
    assert weights == sorted(weights, reverse=True)


def test_docs_upload_and_ask(monkeypatch):
    # Mock the LLM so the test is deterministic and needs no API key. The mock
    # echoes that it received the retrieved context (proves RAG wired it in).
    captured = {}

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "Refunds are available within 30 days."

    monkeypatch.setattr(rag.llm, "generate", fake_generate)

    policy = (
        "Refund policy: customers may request a full refund within 30 days of "
        "purchase. Loyalty discount: customers on monthly plans for over 12 "
        "months qualify for a 15% loyalty discount. Premium support includes "
        "priority response within 2 hours."
    ).encode()
    up = client.post(
        "/docs-upload", files={"file": ("policy.txt", policy, "text/plain")}
    ).json()
    assert up["num_chunks"] >= 1

    resp = client.post(
        "/ask", json={"doc_id": up["doc_id"], "question": "What is the refund window?"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "Refunds are available within 30 days."
    assert len(data["sources"]) >= 1
    assert "CONTEXT:" in captured["prompt"]
    assert "refund" in captured["prompt"].lower()


def test_ask_runs_the_langchain_chain_when_live(monkeypatch):
    # Force the "live" path so the real LangChain composition
    # (ChatPromptTemplate | model | StrOutputParser) actually executes, with a
    # fake chat model standing in for Gemini (no key, no network).
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(rag.llm, "is_live", lambda: True)
    # get_chat_model() is piped as prompt | model | parser; a RunnableLambda
    # that returns an AIMessage is a drop-in for the chat model.
    fake_model = RunnableLambda(lambda prompt_value: AIMessage(content="Within 30 days."))
    monkeypatch.setattr(rag.llm, "get_chat_model", lambda: fake_model)

    policy = b"Refund policy: full refund within 30 days of purchase."
    up = client.post(
        "/docs-upload", files={"file": ("p.txt", policy, "text/plain")}
    ).json()
    resp = client.post(
        "/ask", json={"doc_id": up["doc_id"], "question": "Refund window?"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "Within 30 days."
    assert data["llm_live"] is True
    assert len(data["sources"]) >= 1


def test_ask_unknown_doc_is_404():
    resp = client.post("/ask", json={"doc_id": "nope", "question": "hi"})
    assert resp.status_code == 404


def test_retention_email_is_grounded_in_policy(monkeypatch):
    # This is the integration point: churn output + RAG context -> email.
    captured = {}

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "Hi Ava, we'd love to keep you. Here's a 15% loyalty discount."

    monkeypatch.setattr(retention.llm, "is_live", lambda: True)
    monkeypatch.setattr(retention.llm, "generate", fake_generate)

    # Upload a policy doc so the email can be grounded.
    policy = (
        "Loyalty discount: monthly-plan customers over 12 months qualify for a "
        "15% discount. Pricing: no discounts beyond the loyalty program."
    ).encode()
    up = client.post(
        "/docs-upload", files={"file": ("policy.txt", policy, "text/plain")}
    ).json()

    customer = {
        "customer_id": "C1001",
        "name": "Ava Patel",
        "churn_probability": 0.82,
        "risk_level": "high",
        "contract_label": "Monthly",
        "top_factors": [
            {"feature": "monthly charges", "direction": "increases risk", "weight": 0.2},
            {"feature": "support calls", "direction": "increases risk", "weight": 0.15},
            {"feature": "tenure", "direction": "lowers risk", "weight": 0.05},
        ],
        "features": {},
    }
    resp = client.post(
        "/retention-email", json={"customer": customer, "doc_id": up["doc_id"]}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["email"].startswith("Hi Ava")
    # The policy context and the customer's risk must both reach the prompt.
    assert "CONTEXT:" in captured["prompt"]
    assert "discount" in captured["prompt"].lower()
    assert "high" in captured["prompt"].lower()
    # It retrieved grounding passages from the uploaded doc.
    assert len(data["grounded_in"]) >= 1


def test_retention_email_without_doc_still_works(monkeypatch):
    monkeypatch.setattr(retention.llm, "is_live", lambda: True)
    monkeypatch.setattr(retention.llm, "generate", lambda p: "Hi there, let's chat.")
    customer = {
        "customer_id": "C1",
        "name": "",
        "churn_probability": 0.7,
        "risk_level": "high",
        "contract_label": "Monthly",
        "top_factors": [
            {"feature": "support calls", "direction": "increases risk", "weight": 0.2}
        ],
        "features": {},
    }
    resp = client.post("/retention-email", json={"customer": customer, "doc_id": None})
    assert resp.status_code == 200
    assert resp.json()["grounded_in"] == []


def test_demo_email_is_clean_not_prompt_scaffolding():
    # In demo mode (no key) the email must read like an email, not echo the
    # prompt template ("Write the email now...", "CONTEXT:").
    customer = {
        "customer_id": "C1",
        "name": "Kai Shah",
        "churn_probability": 0.72,
        "risk_level": "high",
        "contract_label": "Monthly",
        "top_factors": [
            {"feature": "contract type", "direction": "increases risk", "weight": 0.2}
        ],
        "features": {},
    }
    data = client.post(
        "/retention-email", json={"customer": customer, "doc_id": None}
    ).json()
    assert data["llm_live"] is False
    assert data["email"].startswith("Hi Kai Shah,")
    assert "CONTEXT:" not in data["email"]
    assert "Write the email now" not in data["email"]
