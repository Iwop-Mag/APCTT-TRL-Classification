"""Input features of the TRL head. The same function is used at training time (tools: train_head2.py) and at inference,
so the two can never disagree. Feature sets: probs | probs+counts | probs+genre | all."""

from __future__ import annotations

import math

TERMS = ["user guide", "user's manual", "documentation", "software", "source code", "installation", "example data", "tutorial", "package",
         "flood-inundation", "inundation map", "streamgage", "real-time", "operational", "web", "available online", "published", "decision",
         "calibrat", "validat", "verif", "observed", "scenario", "climate", "future", "simulat", "sensitivity", "uncertainty",
         "field", "monitoring", "sampling", "survey", "pilot", "prototype", "laborator", "tested", "applied", "demonstrat",
         "limitations", "recommend", "further study", "proposed", "framework", "conceptual", "method"]
MAX_PAGES_TEXT = 60


def page_text(pages) -> tuple[str, int]:
    return " ".join(p.get("text", "") for p in pages[:MAX_PAGES_TEXT]).lower(), len(pages)


def build(feature_set: str, ids, best: dict, counts: dict | None = None, text: str = "", n_pages: int = 0) -> list[float]:
    f = [float(best.get(c, 0.0)) for c in ids]
    if feature_set in ("probs+counts", "all"):
        f += [math.log1p((counts or {}).get(c, 0)) for c in ids]
    if feature_set in ("probs+genre", "all"):
        per10k = max(len(text), 1) / 10000
        f += [math.log1p(text.count(w) / per10k) for w in TERMS] + [math.log1p(n_pages)]
    return f
