# participant_module/extraction.py

from __future__ import annotations

import pdfplumber


def read_pages(case, remaining_ms):

    pages = []

    for doc in case.get("documents", []):

        pdf_path = doc["path"]

        with pdfplumber.open(pdf_path) as pdf:

            for page_num, page in enumerate(pdf.pages, start=1):

                text = page.extract_text() or ""

                pages.append({
                    "document_id": doc["document_id"],
                    "page": page_num,
                    "text": text
                })

    return pages