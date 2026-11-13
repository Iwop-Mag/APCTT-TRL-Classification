"""Turns the four raw signals of an evidence candidate into one calibrated probability and a yes/no.

Signals (all optional except `rule`):  rule  = number of non-negated regex hits in the span
                                       dense = cosine similarity (bge-base) between statement and span
                                       rerank = raw reranker logit (bge-reranker-v2-m3)
                                       nli   = P(entailment) (deberta-v3-large zero-shot)

Without assets/calibration.json the module uses the legacy hand-set thresholds below (your branch's
numbers). `tools_local/train_calibration.py` fits assets/calibration.json from labelled candidates:
a logistic regression over the standardised signals plus one decision threshold per criterion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FEATURES = ("rule", "dense", "rerank", "nli")
# legacy thresholds (used only when no calibration file is shipped)
ENTAIL_MIN = 0.60            # NLI probability needed for a candidate the rules did not find
RULE_ENTAIL_FLOOR = 0.10     # a rule hit is dropped only if NLI strongly disagrees
RERANK_MIN = 0.0             # bge-reranker logit; > 0 means roughly "relevant"


class Calibrator:
    def __init__(self, data: dict | None = None) -> None:
        self.data = data

    @classmethod
    def load(cls, path) -> "Calibrator":
        try:
            p = Path(path)
            return cls(json.loads(p.read_text(encoding="utf-8")) if p.exists() else None)
        except Exception:
            return cls(None)

    @property
    def calibrated(self) -> bool:
        return self.data is not None and "weights" in self.data

    @property
    def trl_settings(self) -> dict:
        return (self.data or {}).get("trl", {"mode": "ladder"})

    def decide(self, criterion_id: str, sig: dict) -> tuple[bool, float]:
        """Return (accept, probability that the span supports the criterion)."""
        if not self.calibrated:                # no logistic fit shipped (a file with only "trl"/"legacy" is fine)
            return self._legacy(sig, (self.data or {}).get("legacy", {}))
        d = self.data
        z = d["bias"]
        for k in FEATURES:
            v = sig.get(k)
            if k == "rule":
                v = min(v or 0, 3)
            elif v is None:
                v = d["impute"][k]
            s = d["scale"][k]
            z += d["weights"][k] * (v - s["mean"]) / (s["std"] or 1.0)
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        thr = d.get("thresholds", {}).get(criterion_id, d.get("default_threshold", 0.5))
        return p >= thr, p

    @staticmethod
    def _legacy(sig: dict, ov: dict | None = None) -> tuple[bool, float]:
        ov = ov or {}
        entail_min = ov.get("entail_min", ENTAIL_MIN)
        floor = ov.get("rule_floor", RULE_ENTAIL_FLOOR)
        rr_min = ov.get("rerank_min", RERANK_MIN)
        rule, nli, rr = sig.get("rule") or 0, sig.get("nli"), sig.get("rerank")
        if rule >= 1:
            ok = nli is None or nli >= floor
            prob = nli if nli is not None else min(0.9, 0.4 + 0.2 * rule)
        else:                                  # found only by search: needs NLI to confirm
            ok = nli is not None and nli >= entail_min and (rr is None or rr >= rr_min)
            prob = nli if nli is not None else 0.0
        return ok, float(prob)
