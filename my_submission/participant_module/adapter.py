"""participant_module/adapter.py

Pipeline per criterion:
  1. candidates : rules (regex) + dense retrieval (bge-base) over page chunks
  2. rerank     : bge-reranker-v2-m3 scores (criterion, chunk)
  3. verify     : deberta-v3-large zeroshot NLI (chunk entails criterion?)
  4. (optional) : qwen3-4b judge on borderline cases only (settings.use_llm)
Every model is optional: if it fails to load, that stage is skipped and the
rules-only behaviour of the starter code remains.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from participant_module import extraction, rules

try:
    import torch
except Exception:                                   # pragma: no cover
    torch = None

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---- tunable thresholds (study / customise these) -------------------------
TOP_K_DENSE = 20        # chunks kept after embedding retrieval
TOP_K_RERANK = 4        # chunks kept after reranking
RERANK_MIN = 0.0        # bge-reranker raw logit; >0 ~ "relevant"
ENTAIL_MIN = 0.60       # NLI entailment prob needed for dense/reranked hits
RULE_ENTAIL_FLOOR = 0.10  # rule hits are dropped only if NLI strongly disagrees
LLM_BAND = (0.30, 0.60)   # NLI probs in this band go to the LLM judge
TIME_FLOOR_MS = 20_000
# ---------------------------------------------------------------------------


class AssessmentModule:
    def __init__(self):
        self._criterion_ids, self._patterns, self._rung = [], {}, {}
        self._descriptions = {}
        self._checklist_version = None
        self._settings = {}
        self._device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
        self._embedder = self._reranker = self._nli = self._trl_head = None
        self._llm = None
        self._entail_idx = 0

    # ------------------------------------------------------------------ load
    def load(self, config, asset_root):
        cl = config["checklist"]
        self._checklist_version = cl["version"]
        self._criterion_ids = [c["criterion_id"] for c in cl["criteria"]]
        self._descriptions = {c["criterion_id"]: c["description"] for c in cl["criteria"]}
        self._settings = (
            config.get("module", {})
                  .get("settings", {})
        )

        data = json.loads((Path(asset_root) / "rules" / "criteria.json").read_text())
        self._patterns, self._rung = data["patterns"], data["trl_rung"]

        want = (config.get("compute") or {}).get("device", "cuda:0")
        if torch is not None and want.startswith("cuda") and torch.cuda.is_available():
            self._device = want
        store = os.environ.get("APCTT_MODEL_STORE")

        # 1. embedder
        try:
            self._embedder = self._load_encoder(self._model_dir(store, "bge-base-en-v1.5"))
        except Exception:
            self._embedder = None
        # 2. reranker
        try:
            self._reranker = self._load_seq_cls(self._model_dir(store, "bge-reranker-v2-m3"))
        except Exception:
            self._reranker = None
        # 3. NLI
        try:
            self._nli = self._load_seq_cls(self._model_dir(store, "deberta-v3-large-zeroshot-v2.0-c"))
            id2label = {int(k): v.lower() for k, v in self._nli[1].config.id2label.items()}
            self._entail_idx = next(i for i, l in id2label.items() if l.startswith("entail"))
        except Exception:
            self._nli = None
        # 4a. your own trained head (optional, only if you ship one)
        try:
            import joblib
            p = Path(asset_root) / "trl_head.joblib"
            self._trl_head = joblib.load(p) if p.exists() else None
        except Exception:
            self._trl_head = None
        # 4b. LLM judge: heavy, off unless settings {"use_llm": true}
        if self._settings.get("use_llm"):
            try:
                self._llm = self._load_llm(self._model_dir(store, "qwen3-4b-instruct-2507-nf4"))
            except Exception:
                self._llm = None

        print(f"embedder={self._embedder is not None}", flush=True)
        print(f"reranker={self._reranker is not None}", flush=True)
        print(f"nli={self._nli is not None}", flush=True)
        print(f"llm={self._llm is not None}", flush=True)
        print(f"device={self._device}", flush=True)

    # -- model helpers -------------------------------------------------------
    @staticmethod
    def _model_dir(store, catalog_id):
        # ASSUMPTION: models live at <store>/<catalog_id>. Adjust if the real layout differs.
        if not store:
            raise FileNotFoundError("APCTT_MODEL_STORE not set")
        p = Path(store) / catalog_id
        if not p.exists():
            raise FileNotFoundError(p)
        return str(p)

    def _load_encoder(self, path):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mdl = AutoModel.from_pretrained(path, local_files_only=True).to(self._device).eval()
        print("Loading:", path, flush=True)
        if self._device.startswith("cuda"):
            mdl.half()
        return tok, mdl

    def _load_seq_cls(self, path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        print("Loading:", path, flush=True)
        mdl = mdl.to(self._device).eval()
        if self._device.startswith("cuda"):
            mdl.half()
        return tok, mdl

    def _load_llm(self, path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        # nf4 checkpoint is pre-quantised; needs bitsandbytes installed
        mdl = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, device_map={"": self._device}).eval()
        return tok, mdl

    # -- model inference -----------------------------------------------------
    def _embed(self, texts, is_query=False, bs=32):
        tok, mdl = self._embedder
        if is_query:
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                b = tok(texts[i:i + bs], padding=True, truncation=True,
                        max_length=256, return_tensors="pt").to(self._device)
                cls = mdl(**b).last_hidden_state[:, 0]          # bge uses CLS pooling
                out.append(torch.nn.functional.normalize(cls.float(), dim=-1).cpu())
        return torch.cat(out) if out else torch.empty(0, 768)

    def _rerank(self, query, passages, bs=16):
        tok, mdl = self._reranker
        scores = []
        with torch.no_grad():
            for i in range(0, len(passages), bs):
                b = tok([query] * len(passages[i:i + bs]), passages[i:i + bs],
                        padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(self._device)
                scores += mdl(**b).logits.view(-1).float().cpu().tolist()
        return scores

    def _entails(self, criterion_id, span):
        """P(entailment) of 'span supports criterion'. None if NLI unavailable."""
        if self._nli is None:
            return None
        tok, mdl = self._nli
        hyp = "The document states that: " + self._descriptions[criterion_id]
        with torch.no_grad():
            b = tok(span, hyp, truncation=True, max_length=512,
                    return_tensors="pt").to(self._device)
            probs = torch.softmax(mdl(**b).logits.float(), dim=-1)[0]
        return float(probs[self._entail_idx])

    def _llm_judge(self, criterion_id, span):
        """True/False from Qwen, or None. Used only for borderline NLI scores."""
        if self._llm is None:
            return None
        tok, mdl = self._llm
        msgs = [{"role": "user", "content":
                 f"Criterion: {self._descriptions[criterion_id]}\n\nPassage: {span}\n\n"
                 "Does the passage clearly satisfy the criterion? Answer YES or NO."}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        yes = tok.encode("YES", add_special_tokens=False)[0]
        no = tok.encode("NO", add_special_tokens=False)[0]
        with torch.no_grad():
            ids = tok(prompt, return_tensors="pt").to(mdl.device)
            logits = mdl(**ids).logits[0, -1]
        return bool(logits[yes] > logits[no])

    # ------------------------------------------------------------ page chunks
    @staticmethod
    def _page_text(p):
        # ASSUMPTION: a page is a dict with "text" (or a plain string).
        if isinstance(p, str):
            return p
        return p.get("text") or p.get("page_text") or ""

    def _chunk_pages(self, pages, win=3, stride=2, max_chars=700):
        chunks = []
        for pi, p in enumerate(pages):
            text = re.sub(r"\s+", " ", self._page_text(p)).strip()
            sents = re.split(r"(?<=[.!?])\s+", text)
            for s in range(0, max(len(sents), 1), stride):
                span = " ".join(sents[s:s + win])[:max_chars]
                if len(span) > 40:
                    chunks.append({"page_index": pi, "span": span})
        return chunks

    def _make_evidence(self, pages, chunk, template):
        """Build an evidence item shaped like the ones rules.find_evidence returns."""
        item = {k: v for k, v in (template or {}).items()}
        p = pages[chunk["page_index"]]
        if isinstance(p, dict):                      # carry page identifiers across
            for k in ("document_id", "doc_id", "page", "page_number", "page_id"):
                if k in p:
                    item[k] = p[k]
        item["supporting_span"] = chunk["span"]
        return item

    # ----------------------------------------------------------------- assess
    def assess(self, case, context):
        result = {
            "schema_version": case["schema_version"],
            "checklist_version": self._checklist_version,
            "case_id": case["case_id"],
            "assessment_target": dict(case["assessment_target"]),
        }
        try:
            pages = extraction.read_pages(case, context.remaining_ms)
        except Exception:
            pages = []

        # embed all chunks once per case, reuse for every criterion
        chunks, chunk_emb = [], None
        if self._embedder is not None and pages:
            try:
                chunks = self._chunk_pages(pages)
                chunk_emb = self._embed([c["span"] for c in chunks]) if chunks else None
            except Exception:
                chunks, chunk_emb = [], None

        status, evidence, missing = {}, {}, {}
        for cid in self._criterion_ids:
            if context.remaining_ms() < TIME_FLOOR_MS:
                status[cid] = "insufficient evidence"
                evidence[cid] = []
                missing[cid] = "Not assessed: time limit."
                continue
            try:
                found = self._assess_criterion(cid, pages, chunks, chunk_emb)
            except Exception:
                found = rules.find_evidence(pages, self._patterns.get(cid, []))
            evidence[cid] = found
            status[cid] = "met" if found else "insufficient evidence"
            if not found:
                missing[cid] = "No page states this clearly."

        met = [self._rung[c] for c in self._criterion_ids
               if status[c] == "met" and c in self._rung]
        result.update(criterion_status=status, evidence_reference=evidence,
                      missing_evidence=missing, error_code=None)

        print("PAGES:", len(pages), flush=True)
        print("CHUNKS:", len(chunks), flush=True)
        
        if met:
            result.update(assessment_status="completed", predicted_trl=max(met),
                          abstention_reason=None)
        else:
            result.update(assessment_status="abstained", predicted_trl=None,
                          abstention_reason="insufficient evidence")
        return result

    def _assess_criterion(self, cid, pages, chunks, chunk_emb):
        # step 1a: rule hits
        rule_hits = rules.find_evidence(pages, self._patterns.get(cid, []))
        kept = []
        for f in rule_hits:
            p = self._entails(cid, f["supporting_span"])
            if p is None or p >= RULE_ENTAIL_FLOOR:
                kept.append(f)
        template = rule_hits[0] if rule_hits else None

        # step 1b: dense retrieval + rerank
        dense = []
        if chunk_emb is not None and len(chunks):
            q = self._embed([self._descriptions[cid]], is_query=True)
            sims = (chunk_emb @ q.T).squeeze(1)
            top = sims.topk(min(TOP_K_DENSE, len(chunks))).indices.tolist()
            cand = [chunks[i] for i in top]
            if self._reranker is not None:
                scores = self._rerank(self._descriptions[cid], [c["span"] for c in cand])
                order = sorted(range(len(cand)), key=lambda i: -scores[i])[:TOP_K_RERANK]
                cand = [cand[i] for i in order if scores[i] >= RERANK_MIN]
            else:
                cand = cand[:TOP_K_RERANK]

            # step 2: NLI (+ optional LLM) verification
            seen = {f["supporting_span"] for f in kept}
            for c in cand:
                if c["span"] in seen:
                    continue
                p = self._entails(cid, c["span"])
                ok = True if p is None else p >= ENTAIL_MIN
                if p is not None and LLM_BAND[0] <= p < LLM_BAND[1]:
                    verdict = self._llm_judge(cid, c["span"])
                    ok = ok if verdict is None else verdict
                if ok:
                    dense.append(self._make_evidence(pages, c, template))
        return kept + dense

    def reset_case(self):
        if torch is not None and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def close(self):
        self._embedder = self._reranker = self._nli = self._llm = None
        self.reset_case()