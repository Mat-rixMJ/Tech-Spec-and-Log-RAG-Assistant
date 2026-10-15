"""
Provider-agnostic LLM wrapper, built on LangChain.

Everything in the app calls `generate(prompt)` — nothing else knows which
provider or framework is behind it. The RAG and retention chains in rag.py /
retention.py are composed with LangChain (prompt | model | parser), and the
model is LangChain's ChatGoogleGenerativeAI (Gemini). Swapping providers is a
one-file change here.

If no key is configured, we fall back to a deterministic offline stub so the
app still runs and the test suite stays green without a key.
"""

from __future__ import annotations

import os
import textwrap
from functools import lru_cache

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


class LLMError(RuntimeError):
    pass


def _gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_live() -> bool:
    """True when a real provider is configured (vs. the offline stub)."""
    if PROVIDER == "gemini":
        return bool(_gemini_key())
    return False


@lru_cache(maxsize=1)
def get_chat_model():
    """
    The LangChain chat model, built once. rag.py and retention.py pipe their
    prompts into this (prompt | model | StrOutputParser). Raises if no key —
    callers only reach here when is_live() is True.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    if PROVIDER != "gemini":
        raise LLMError(f"Unknown LLM_PROVIDER: {PROVIDER}")
    if not _gemini_key():
        raise LLMError("No GEMINI_API_KEY configured")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=_gemini_key(),
        temperature=0.4,
    )


def _generate_stub(prompt: str) -> str:
    """
    Offline fallback. Returns the context the prompt carried so the UI still
    shows something grounded. Callers put retrievable context after "CONTEXT:\n".
    """
    marker = "CONTEXT:\n"
    context = prompt.split(marker, 1)[1] if marker in prompt else prompt
    snippet = textwrap.shorten(context.strip(), width=600, placeholder=" …")
    return (
        "(Demo mode — no LLM API key configured, showing the retrieved context "
        "instead of a generated answer.)\n\n" + snippet
    )


def generate(prompt: str) -> str:
    """
    Single entry point. Runs the prompt through a minimal LangChain chain
    (model | StrOutputParser) when a key is present, else the offline stub.
    """
    if not is_live():
        return _generate_stub(prompt)

    from langchain_core.output_parsers import StrOutputParser

    chain = get_chat_model() | StrOutputParser()
    text = chain.invoke(prompt)
    if not text or not text.strip():
        raise LLMError("LLM returned an empty response")
    return text.strip()
