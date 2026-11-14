"""harness.py -- shared helpers for diagnose.py and calibrate.py.

These scripts are NOT part of your submission: keep them in ~/dev/mytools/ (outside my_submission/).
Run them with the base environment's Python (the kit's `worker.python`), which already has torch,
transformers, pypdf, numpy and scikit-learn. Nothing needs to be installed anywhere.

Paths default to the layout from the kit README:
  ~/dev/apctt-participant-kit-0.5.0-cuda   (--kit)       ~/dev/my_submission   (--sub)
  ~/dev/public_cases/pdf                   (--pdf-dir)   ~/dev/models          (read from config/my.local.json)
"""
from __future__ import annotations
import copy, json, os, re, sys, time
from pathlib import Path
from types import SimpleNamespace

DEV = Path.home() / "dev"


def add_common_args(ap):
    ap.add_argument("--kit", default=str(DEV / "apctt-participant-kit-0.5.0-cuda"))
    ap.add_argument("--sub", default=str(DEV / "my_submission"))
    ap.add_argument("--pdf-dir", default=str(DEV / "public_cases" / "pdf"))
    ap.add_argument("--config", default="my.local.json", help="file in <kit>/config/")
    ap.add_argument("--settings", default=None, help="JSON overriding module.settings, e.g. '{\"use_rerank\": false}'")


class Ctx:
    """Stands in for the worker's context: same remaining_ms() contract, 3-minute budget."""
    def __init__(self, ms=180_000):
        self._end = time.time() * 1000 + ms
    def remaining_ms(self):
        return self._end - time.time() * 1000


def setup(a) -> SimpleNamespace:
    kit, sub = Path(a.kit).expanduser().resolve(), Path(a.sub).expanduser().resolve()
    cfg_path = kit / "config" / a.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # paths inside the config are relative to config/ (README step 6)
    chk = (cfg_path.parent / cfg.get("checklist_path", "../contracts/checklist.official.json")).resolve()
    checklist = json.loads(chk.read_text(encoding="utf-8"))
    settings = dict((cfg.get("module") or {}).get("settings") or {})
    if a.settings:
        settings.update(json.loads(a.settings))
    store = os.environ.get("APCTT_MODEL_STORE")
    if not store:
        raw = ((cfg.get("worker") or {}).get("extra_env") or {}).get("APCTT_MODEL_STORE")
        if raw:
            store = str((cfg_path.parent / raw).resolve()) if not os.path.isabs(raw) else raw
    if store:
        os.environ["APCTT_MODEL_STORE"] = store
    if not (sub / "participant_module").is_dir():
        sys.exit(f"no participant_module/ in {sub} (use --sub)")
    sys.path.insert(0, str(sub))
    module_cfg = dict(cfg)
    module_cfg["checklist"] = checklist
    module_cfg["module"] = dict(cfg.get("module") or {}, settings=settings)
    return SimpleNamespace(kit=kit, sub=sub, cfg=module_cfg, checklist=checklist, settings=settings,
                           asset_root=sub / "assets", store=store, pdf_dir=Path(a.pdf_dir).expanduser(),
                           public=kit / "contracts" / "public-cases.json")


def load_module(env):
    from participant_module.adapter import AssessmentModule
    mod = AssessmentModule()
    t = time.time()
    mod.load(env.cfg, env.asset_root)
    print(f"[harness] load {time.time() - t:.1f}s  store={env.store}  settings={env.settings}", flush=True)
    return mod


# ------------------------------------------------------------------ cases
def _find_entry(obj, cid):
    if isinstance(obj, dict):
        if obj.get("case_id") == cid:
            return obj
        for v in obj.values():
            r = _find_entry(v, cid)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_entry(v, cid)
            if r is not None:
                return r
    return None


def _scrub(d):
    """Drop anything label-like: the module must never see the organizer's TRL estimate."""
    if isinstance(d, dict):
        return {k: _scrub(v) for k, v in d.items() if "trl" not in k.lower() and "label" not in k.lower()}
    if isinstance(d, list):
        return [_scrub(v) for v in d]
    return d


def build_case(env, pdf: Path) -> dict:
    from pypdf import PdfReader
    cid = pdf.stem
    entry = None
    if env.public.exists():
        entry = _find_entry(json.loads(env.public.read_text(encoding="utf-8")), cid)
    base = _scrub(copy.deepcopy(entry.get("case", entry) if isinstance(entry, dict) else {}))
    base.setdefault("schema_version", "dev")
    base["case_id"] = cid
    base.setdefault("title", cid)
    base.setdefault("assessment_target", {"target_id": cid, "type": "technology", "description": cid})
    n = len(PdfReader(str(pdf)).pages)
    docs = base.get("documents")
    if isinstance(docs, list) and docs and isinstance(docs[0], dict):
        docs = [docs[0]]
    else:
        docs = [{"document_id": "D1", "title": cid, "document_date": None, "language": "en"}]
    docs[0].update(path=str(pdf.resolve()), page_count=n)
    docs[0].setdefault("document_id", "D1")
    base["documents"] = docs
    return base


# ----------------------------------------------------------------- labels
def trl_from_entry(entry):
    """Find an int 1..9 under any key containing 'trl' (the organizer's estimate) in a public-cases entry."""
    if isinstance(entry, dict):
        for k, v in entry.items():
            if "trl" in k.lower() and isinstance(v, int) and 1 <= v <= 9:
                return v
            if isinstance(v, (dict, list)):
                r = trl_from_entry(v)
                if r:
                    return r
    elif isinstance(entry, list):
        for v in entry:
            r = trl_from_entry(v)
            if r:
                return r
    return None


def a_score(pred, label):
    if pred is None:
        return 0.25
    d = abs(pred - label)
    return 1.0 if d == 0 else 0.5 if d == 1 else 0.0
