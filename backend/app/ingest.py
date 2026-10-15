"""
File ingestion helpers: turn an uploaded PDF or CSV into text/data, with
validation at the trust boundary (type and size).
"""

from __future__ import annotations

import io

import pandas as pd
from pypdf import PdfReader

MAX_BYTES = 5 * 1024 * 1024  # 5 MB — plenty for a demo, caps abuse.


class IngestError(ValueError):
    """Raised for anything wrong with the uploaded file (maps to HTTP 400)."""


def validate_size(data: bytes) -> None:
    if len(data) == 0:
        raise IngestError("Uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise IngestError(
            f"File too large ({len(data) // 1024} KB). Limit is {MAX_BYTES // 1024} KB."
        )


def extract_pdf_text(data: bytes) -> str:
    validate_size(data)
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise IngestError(f"Could not read PDF: {exc}") from exc
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise IngestError("No extractable text in this PDF (it may be a scanned image).")
    return text


def load_csv(data: bytes) -> pd.DataFrame:
    validate_size(data)
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise IngestError(f"Could not parse CSV: {exc}") from exc
    if df.empty:
        raise IngestError("CSV has no rows.")
    return df
