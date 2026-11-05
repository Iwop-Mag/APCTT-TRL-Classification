"""participant_module/adapter.py - example rules-based Assessment Package (team TUP Manila / HALIGI).

Pipeline: pypdf page text -> keyword rules per checklist criterion -> evidence pages (best first)
-> TRL = highest rung of the TRL ladder whose criterion is met, otherwise abstain.
Criterion IDs come from config["checklist"]; the rules file only supplies patterns for them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # no network, ever

from participant_module import extraction, rules  # noqa: E402


class AssessmentModule:
    def __init__(self) -> None:
        self._checklist_version = ""
        self._criterion_ids: list[str] = []
        self._patterns: dict[str, list[str]] = {}
        self._rung: dict[str, int] = {}

    def load(self, config: Mapping[str, Any], asset_root: Path) -> None:
        checklist = config["checklist"]
        self._checklist_version = checklist["version"]
        self._criterion_ids = [c["criterion_id"] for c in checklist["criteria"]]
        rules_file = Path(asset_root) / "rules" / "criteria.json"
        data = json.loads(rules_file.read_text(encoding="utf-8"))
        self._patterns = data["patterns"]
        self._rung = data["trl_rung"]

    def assess(self, case: Mapping[str, Any], context: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": case["schema_version"],
            "checklist_version": self._checklist_version,
            "case_id": case["case_id"],
            "assessment_target": dict(case["assessment_target"]),
        }
        try:
            pages = extraction.read_pages(case, context.remaining_ms)
        except Exception:
            pages = []  # unreadable PDF: every criterion falls through to insufficient evidence

        status: dict[str, str] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}
        missing: dict[str, str] = {}
        for cid in self._criterion_ids:
            patterns = self._patterns.get(cid, [])
            found = rules.find_evidence(pages, patterns) if patterns else []
            evidence[cid] = found
            if found:
                status[cid] = "met"
            else:
                status[cid] = "insufficient evidence"
                missing[cid] = (
                    "No page states this clearly: what was built or tested, where, and the result."
                )

        met_rungs = [self._rung[c] for c in self._criterion_ids if c in self._rung and status[c] == "met"]
        result.update(
            criterion_status=status,
            evidence_reference=evidence,
            missing_evidence=missing,
            error_code=None,
        )
        if met_rungs:
            trl = max(met_rungs)
            result.update(
                assessment_status="completed",
                predicted_trl=trl,
                abstention_reason=None,
                trl_rationale=f"Highest TRL rung with a met criterion is {trl}.",
            )
        else:
            result.update(
                assessment_status="abstained",
                predicted_trl=None,
                abstention_reason="insufficient evidence",
            )
        return result

    def reset_case(self) -> None:
        """No per-case state is kept."""

    def close(self) -> None:
        """Nothing to release."""
