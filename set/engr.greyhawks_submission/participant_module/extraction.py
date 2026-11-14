"""PDF text extraction. Pages are dicts: {"document_id", "page" (from 1), "text"}.

pypdf is the main reader (0.5 s for 300 text pages here; pdfplumber took 8.8 s on the same file, and
real reports with tables and figures are slower). pdfplumber is only a fallback for pages with almost no
text. The text is only whitespace-normalised, so every span quoted from it is a verbatim substring of it.
"""

from __future__ import annotations

import re

MIN_MS_LEFT = 20_000          # stop reading when less than 20 s of the case budget is left
THIN_PAGE_CHARS = 20          # fewer characters than this -> retry the page with pdfplumber


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _plumber(path: str, number: int) -> str:
    try:
        import pdfplumber                                    # in the common base environment

        with pdfplumber.open(path) as pdf:
            return clean(pdf.pages[number - 1].extract_text() or "")
    except Exception:
        return ""


def read_pages(case, remaining_ms):
    from pypdf import PdfReader                              # in the common base environment

    pages = []
    for doc in case.get("documents", []):
        try:
            reader = PdfReader(doc["path"])
            count = len(reader.pages)
        except Exception:                                    # unreadable PDF: skip it, never fail the case
            continue
        for number in range(1, count + 1):
            if remaining_ms() < MIN_MS_LEFT:
                return pages
            try:
                text = clean(reader.pages[number - 1].extract_text() or "")
            except Exception:
                text = ""
            if len(text) < THIN_PAGE_CHARS and remaining_ms() > 3 * MIN_MS_LEFT:
                text = _plumber(doc["path"], number) or text
            pages.append({"document_id": doc["document_id"], "page": number, "text": text})
    return pages
