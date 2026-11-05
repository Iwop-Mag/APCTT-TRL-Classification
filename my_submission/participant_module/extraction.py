"""PDF text extraction: one string per page, page numbers start at 1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def read_pages(case: Mapping[str, Any], remaining_ms) -> list[tuple[str, int, str]]:
    """Return (document_id, page, text) for every page of every document, in packet order.

    Stops early (returns what it has) when less than 20 s of the case budget is left.
    """
    from pypdf import PdfReader  # in the common base environment

    pages: list[tuple[str, int, str]] = []
    for document in case["documents"]:
        reader = PdfReader(document["path"])
        for number, page in enumerate(reader.pages, start=1):
            if remaining_ms() < 20_000:
                return pages
            try:
                text = page.extract_text() or ""
            except Exception:  # a broken page must not fail the whole case
                text = ""
            pages.append((document["document_id"], number, text))
    return pages
