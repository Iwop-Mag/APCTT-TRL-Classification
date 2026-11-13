from __future__ import annotations

import re


def find_evidence(pages, patterns):

    findings = []

    compiled = [
        re.compile(p, re.IGNORECASE)
        for p in patterns
    ]

    for page in pages:

        text = page.get("text", "")

        for patt in compiled:

            match = patt.search(text)

            if match:

                findings.append({
                    "document_id": page.get("document_id"),
                    "page": page.get("page")
                })

    return findings