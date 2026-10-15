"""
RAG, composed as a LangChain chain.

Retrieval is our TF-IDF store (store.py); generation is a LangChain chain:
    ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser
built in llm.py. When no API key is set we fall back to the offline stub so the
app and tests run without credentials.
"""

from __future__ import annotations

from . import llm
from .store import store

SYSTEM = (
    "You are a support assistant answering questions about a company's policy "
    "and help documents. Answer using ONLY the provided context. If the answer "
    "isn't in the context, say you don't know. Be concise."
)

# Human message keeps the "CONTEXT:\n" marker so the offline stub can echo it.
HUMAN = "QUESTION: {question}\n\nCONTEXT:\n{context}"


def _build_chain():
    """prompt | model | parser — the LangChain RAG generation chain."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
    return prompt | llm.get_chat_model() | StrOutputParser()


def answer(doc_id: str, question: str) -> dict:
    if not store.exists(doc_id):
        raise KeyError(doc_id)

    sources = store.retrieve(doc_id, question, k=4)
    if not sources:
        return {
            "answer": "I couldn't find anything relevant to that question in the document.",
            "sources": [],
            "llm_live": llm.is_live(),
        }

    context = "\n\n---\n\n".join(s["text"] for s in sources)

    if llm.is_live():
        generated = _build_chain().invoke({"question": question, "context": context})
        generated = generated.strip()
    else:
        # Offline: reuse the stub via the same prompt text shape.
        generated = llm.generate(f"QUESTION: {question}\n\nCONTEXT:\n{context}")

    return {"answer": generated, "sources": sources, "llm_live": llm.is_live()}
