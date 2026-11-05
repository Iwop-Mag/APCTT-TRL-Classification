"""Keyword rules: find the best supporting sentences for a criterion."""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# A sentence that only asserts a TRL number is never evidence (problem description 13.4).
_TRL_ASSERTION = re.compile(r"\bTRL\s*-?\s*[1-9]\b", re.IGNORECASE)
# A negated or "not yet" sentence says the opposite of what the rung needs.
_NEGATION = re.compile(
    r"\b(no|not|never|without|yet to|has not|have not|did not|was not|were not|cannot|unable)\b|\byet\b",
    re.IGNORECASE,
)
MAX_SPAN_CHARS = 300


def sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s for s in _SENTENCE_SPLIT.split(flat) if s]


def find_evidence(pages, patterns: list[str], max_items: int = 3) -> list[dict]:
    """Rank pages by how many patterns hit and return up to max_items distinct (doc, page)."""
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    scored: list[tuple[int, int, dict]] = []
    for order, (document_id, page, text) in enumerate(pages):
        best_span, best_hits = None, 0
        for sentence in sentences(text):
            if _TRL_ASSERTION.search(sentence) or _NEGATION.search(sentence):
                continue
            hits = sum(1 for rx in compiled if rx.search(sentence))
            if hits > best_hits:
                best_span, best_hits = sentence, hits
        if best_span:
            span = best_span[:MAX_SPAN_CHARS]
            scored.append(
                (-best_hits, order, {"document_id": document_id, "page": page, "supporting_span": span})
            )
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored[:max_items]]
