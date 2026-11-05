"""Run the example module on the sample report and check the result with the kit's own validator.

Usage: python3 run_local_test.py <kit_dir> <my_submission_dir> <sample_report.pdf>
"""
import json, sys, time
from pathlib import Path

kit, sub, pdf = (Path(a).resolve() for a in sys.argv[1:4])
sys.path[:0] = [str(kit / "src"), str(sub)]

from apctt_contracts.config import Checklist
from apctt_contracts.results import validate_result
from pypdf import PdfReader

checklist_raw = json.loads((kit / "contracts/checklist.official.json").read_text())
checklist = Checklist.model_validate({"version": checklist_raw["version"],
                                      "official": True, "criteria": checklist_raw["criteria"]})

case = {
    "schema_version": "apctt-pdf-example-v1", "input_profile": "pdf_report",
    "checklist_version": checklist.version, "case_id": "LOCAL-001", "assessment_as_of": None,
    "assessment_target": {"target_id": "T001", "type": "system",
                          "description": "Soil-moisture sensor node (synthetic)"},
    "intended_application": "Irrigation scheduling", "intended_environment": "Rice farm",
    "title": "Sample report",
    "documents": [{"document_id": "D001", "title": "Test report", "document_date": None,
                   "language": "en", "page_count": len(PdfReader(str(pdf)).pages), "path": str(pdf)}],
}

from participant_module.adapter import AssessmentModule

class Ctx:  # mimics the worker's context
    def __init__(self): self._t = time.monotonic()
    def remaining_ms(self): return 180000 - (time.monotonic() - self._t) * 1000

m = AssessmentModule()
m.load({"checklist": {"version": checklist.version,
                      "criteria": [c.model_dump() for c in checklist.criteria]},
        "module": {"settings": {}}}, sub / "assets")
result = m.assess(case, Ctx())
m.reset_case(); m.close()

print(json.dumps(result, indent=2))
issues = validate_result(result, case=case, checklist=checklist)
print("\nVALIDATOR ISSUES:", issues if issues else "none")
