"""Regex rules. Every hit carries the sentence that matched (the quote), never just a page number."""

from __future__ import annotations

import re

MAX_SPAN_CHARS = 400
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
# A sentence that only asserts a TRL number is never evidence for a criterion.
_TRL_CLAIM = re.compile(r"\bTRL\s*-?\s*[1-9]\b|technology readiness level", re.IGNORECASE)
# Negation next to a match flips its meaning ("No pilot installation has been carried out yet").
_NEG_BEFORE = re.compile(
    r"\b(no|not|never|without|neither|nor|cannot|unable|lack(?:s|ed|ing)?|yet to|has not|have not|"
    r"did not|was not|were not|not yet|planned|will be|to be)\b", re.IGNORECASE)
_NEG_AFTER = re.compile(
    r"\b(not yet|yet to|has not|have not|did not|was not|were not|not been|not achieved|not met|"
    r"is planned|are planned|will be)\b|\byet\b", re.IGNORECASE)
_STOP = {"with", "from", "that", "this", "using", "based", "system", "technology", "project", "report",
         "development", "analysis", "study", "design", "method", "methods", "into", "their", "which"}
_TEST_WORDS = re.compile(r"\b(tested|measured|demonstrated|validated|evaluated|calibrated|deployed|installed|"
                         r"results?|experiments?|trials?|verified|achieved)\b", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    return [s for s in _SPLIT.split(text or "") if s]


def is_trl_claim(text: str) -> bool:
    return bool(_TRL_CLAIM.search(text or ""))


def compile_patterns(patterns):
    out = []
    for pattern in patterns or []:
        try:
            out.append(re.compile(pattern, re.IGNORECASE))
        except re.error:                                     # a bad pattern must not break the module
            continue
    return out


def _negated(sentence: str, m: re.Match) -> bool:
    before = sentence[max(0, m.start() - 50): m.start()]
    after = sentence[m.end(): m.end() + 40]
    return bool(_NEG_BEFORE.search(before) or _NEG_AFTER.search(after))


def trim(span: str) -> str:
    if len(span) <= MAX_SPAN_CHARS:
        return span
    cut = span[:MAX_SPAN_CHARS]
    return cut[: cut.rfind(" ")] if " " in cut else cut


def span_hits(span: str, compiled) -> int:
    """Number of distinct non-negated patterns that match inside the span."""
    return sum(1 for rx in compiled if (m := rx.search(span)) and not _negated(span, m))


def find_evidence(pages, patterns, max_items: int = 6) -> list[dict]:
    """Best matching sentence per page. Returns dicts with document_id, page, supporting_span, rule."""
    compiled = compile_patterns(patterns)
    if not compiled:
        return []
    scored = []
    for order, page in enumerate(pages):
        best, best_hits = None, 0
        for sentence in sentences(page.get("text", "")):
            if is_trl_claim(sentence):
                continue
            hits = span_hits(sentence, compiled)
            if hits > best_hits:
                best, best_hits = sentence, hits
        if best:
            scored.append((-best_hits, order, {"document_id": page["document_id"], "page": page["page"],
                                               "supporting_span": trim(best), "rule": best_hits}))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored[:max_items]]


def target_tokens(case) -> list[str]:
    target = case.get("assessment_target") or {}
    raw = " ".join(str(target.get(k, "")) for k in ("name", "description")) + " " + str(case.get("title", ""))
    return sorted({w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", raw.lower()) if w not in _STOP})


def find_target_evidence(pages, tokens, max_items: int = 6) -> list[dict]:
    """C12: sentences that name the assessed target (>= 2 of its words) AND report a test or result."""
    if len(tokens) < 2:
        return []
    scored = []
    for order, page in enumerate(pages):
        best, best_hits = None, 0
        for sentence in sentences(page.get("text", "")):
            if is_trl_claim(sentence) or not _TEST_WORDS.search(sentence):
                continue
            hits = sum(1 for t in tokens if t in sentence.lower())
            if hits >= 2 and hits > best_hits:
                best, best_hits = sentence, hits
        if best:
            scored.append((-best_hits, order, {"document_id": page["document_id"], "page": page["page"],
                                               "supporting_span": trim(best), "rule": best_hits}))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored[:max_items]]
