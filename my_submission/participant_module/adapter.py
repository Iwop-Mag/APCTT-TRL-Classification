"""participant_module/adapter.py - engr.greyhawks Assessment Package (v2).

Per criterion:
  1. candidates : regex rules (verbatim sentences) + dense search over page chunks (bge-base) + rerank (bge-reranker)
  2. signals    : rule hits, dense similarity, rerank logit, NLI entailment probability (deberta-v3-large)
  3. decision   : calibration.Calibrator turns the signals into one probability and a yes/no
  4. (optional) : an LLM judge for borderline NLI scores (settings {"use_llm": true}; off by default)
Then: evidence = best 3 distinct pages (best first); TRL = trl.decide_trl (ladder) or the optional trained head.
Every model is optional: if one fails to load, its signal is skipped and the rest still works.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from participant_module import calibration, extraction, modelstore, rules, trl  # noqa: E402

try:
    import torch
except Exception:                                    # pragma: no cover
    torch = None

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K_DENSE = 20            # chunks kept after embedding search
TOP_K_RERANK = 4            # chunks kept after reranking
RULE_CANDIDATES = 5         # rule hits kept per criterion
MAX_EVIDENCE = 3            # only the first 3 distinct pages are scored
MAX_CHUNKS = 6000           # cap on searched chunks (the stride grows for very long documents)
TIME_FLOOR_MS = 20_000      # below this the remaining criteria are skipped
EMBED_MIN_LEFT_MS = 60_000  # stop embedding when less than this is left, keep the time for scoring
LLM_BAND = (0.30, 0.60)     # NLI probabilities in this band go to the LLM judge (when enabled)


def _log(*parts) -> None:
    print(*parts, file=sys.stderr, flush=True)       # logs are never parsed or scored


class AssessmentModule:
    def __init__(self):
        self._version = None
        self._ids, self._desc, self._hyp = [], {}, {}
        self._patterns, self._rung = {}, {}
        self._settings = {}
        self._cal = calibration.Calibrator(None)
        self._device = "cpu"
        self._embedder = self._reranker = self._nli = self._llm = self._trl_head = None
        self._entail_idx = 0
        self._dim = 768
        self.last_trace = []                         # every candidate of the last case (used by tools_local)

    # ------------------------------------------------------------------ load
    def load(self, config, asset_root):
        cl = config["checklist"]
        self._version = cl["version"]
        self._ids = [c["criterion_id"] for c in cl["criteria"]]
        self._desc = {c["criterion_id"]: c.get("description", "") for c in cl["criteria"]}
        self._settings = dict((config.get("module") or {}).get("settings") or {})
        s = self._settings
        asset_root = Path(asset_root)

        data = json.loads((asset_root / "rules" / "criteria.json").read_text(encoding="utf-8"))
        self._patterns, self._rung = data["patterns"], data["trl_rung"]
        self._hyp = data.get("hypotheses", {})
        self._cal = calibration.Calibrator.load(asset_root / "calibration.json")

        want = s.get("device") or "cuda:0"
        self._device = want if (torch is not None and want.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        store = os.environ.get("APCTT_MODEL_STORE")

        if torch is not None:
            if s.get("use_dense", True):
                self._embedder = self._try("embedder", lambda: self._load_encoder(modelstore.resolve(store, "bge-base-en-v1.5")))
            if s.get("use_rerank", True):
                self._reranker = self._try("reranker", lambda: self._load_seq_cls(modelstore.resolve(store, "bge-reranker-v2-m3")))
            if s.get("use_nli", True):
                self._nli = self._try("nli", self._load_nli_model(store))
            if s.get("use_llm", False):
                self._llm = self._try("llm", lambda: self._load_llm(modelstore.resolve(store, "qwen3-4b-instruct-2507-nf4")))
        if self._cal.trl_settings.get("mode") == "head":
            def _head():
                import joblib
                return joblib.load(asset_root / "trl_head.joblib")
            self._trl_head = self._try("trl_head", _head)
        _log(f"calibrated={self._cal.calibrated} embedder={self._embedder is not None} reranker={self._reranker is not None} "
             f"nli={self._nli is not None} llm={self._llm is not None} head={self._trl_head is not None} device={self._device}")

    @staticmethod
    def _try(name, fn):
        try:
            return fn()
        except Exception as exc:                     # a model that cannot load is skipped, not fatal
            _log(f"skip {name}: {type(exc).__name__}: {exc}")
            return None

    def _load_nli_model(self, store):
        def go():
            tok, mdl = self._load_seq_cls(modelstore.resolve(store, "deberta-v3-large-zeroshot-v2.0-c"))
            labels = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
            self._entail_idx = next(i for i, name in labels.items() if name.startswith("entail"))
            return tok, mdl
        return go

    def _load_encoder(self, path):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mdl = AutoModel.from_pretrained(path, local_files_only=True).to(self._device).eval()
        self._dim = mdl.config.hidden_size
        if self._device.startswith("cuda"):
            mdl.half()
        return tok, mdl

    def _load_seq_cls(self, path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(self._device).eval()
        if self._device.startswith("cuda"):
            mdl.half()
        return tok, mdl

    def _load_llm(self, path):
        # The store holds bf16 weights; they are quantised to 4 bit while loading (needs a CUDA GPU and
        # bitsandbytes). The previous version passed no quantization_config, which loads 4 B parameters
        # in full precision and runs out of memory on an 8 GB card.
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mdl = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                                                   quantization_config=quant, device_map={"": 0}).eval()
        return tok, mdl

    # ------------------------------------------------------- model inference
    def _embed(self, texts, is_query=False, bs=32, context=None):
        tok, mdl = self._embedder
        if is_query:
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                if context is not None and context.remaining_ms() < EMBED_MIN_LEFT_MS:
                    raise TimeoutError("embedding stopped: not enough time left")
                b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self._device)
                cls = mdl(**b).last_hidden_state[:, 0]                    # bge uses CLS pooling
                out.append(torch.nn.functional.normalize(cls.float(), dim=-1).cpu())
        return torch.cat(out) if out else torch.empty(0, self._dim)

    def _rerank(self, query, passages, bs=16):
        tok, mdl = self._reranker
        scores = []
        with torch.no_grad():
            for i in range(0, len(passages), bs):
                part = passages[i:i + bs]
                b = tok([query] * len(part), part, padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(self._device)
                scores += mdl(**b).logits.view(-1).float().cpu().tolist()
        return scores

    def _entails_many(self, cid, spans, bs=8):
        """P(entailment) of 'span supports the statement of criterion cid', one value per span."""
        tok, mdl = self._nli
        hyp = self._hyp.get(cid) or self._desc[cid]
        probs = []
        with torch.no_grad():
            for i in range(0, len(spans), bs):
                part = spans[i:i + bs]
                b = tok(part, [hyp] * len(part), padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(self._device)
                p = torch.softmax(mdl(**b).logits.float(), dim=-1)[:, self._entail_idx].cpu().tolist()
                probs += [0.0 if math.isnan(v) else v for v in p]        # fp16 can overflow: treat NaN as 0
        return probs

    def _llm_judge(self, cid, span):
        """True/False from the LLM for a borderline candidate, or None if unavailable."""
        if self._llm is None:
            return None
        tok, mdl = self._llm
        msgs = [{"role": "user", "content": f"Criterion: {self._desc[cid]}\n\nPassage: {span}\n\n"
                                             "Does the passage clearly satisfy the criterion? Answer YES or NO."}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        yes = tok.encode("YES", add_special_tokens=False)[0]
        no = tok.encode("NO", add_special_tokens=False)[0]
        with torch.no_grad():
            ids = tok(prompt, return_tensors="pt").to(mdl.device)
            logits = mdl(**ids).logits[0, -1]
        return bool(logits[yes] > logits[no])

    # ---------------------------------------------------------------- chunks
    def _chunk_pages(self, pages, win=2):
        total = sum(len(rules.sentences(p.get("text", ""))) for p in pages)
        stride = max(1, math.ceil(total / MAX_CHUNKS))
        chunks = []
        for p in pages:
            sents = rules.sentences(p.get("text", ""))
            for i in range(0, len(sents), stride):
                span = rules.trim(" ".join(sents[i:i + win]))
                digits = sum(ch.isdigit() for ch in span)
                if len(span) > 40 and digits < 0.4 * len(span) and not rules.is_trl_claim(span):
                    chunks.append({"document_id": p["document_id"], "page": p["page"], "supporting_span": span})
        return chunks

    def _prepare_chunks(self, pages, context):
        if self._embedder is None or not pages:
            return [], None
        try:
            chunks = self._chunk_pages(pages)
            return (chunks, self._embed([c["supporting_span"] for c in chunks], context=context)) if chunks else ([], None)
        except Exception as exc:
            _log(f"dense search skipped: {type(exc).__name__}: {exc}")
            return [], None

    # ------------------------------------------------------------ candidates
    def _target_criterion(self):
        """The last criterion with no patterns and no TRL rung (APCTT-C12 in the official checklist)."""
        free = [c for c in self._ids if not self._patterns.get(c) and c not in self._rung]
        return free[-1] if free else None

    def _candidates(self, cid, pages, chunks, emb, tokens):
        found = {}

        def add(item):
            found.setdefault((item["document_id"], item["page"], item["supporting_span"]), dict(item))

        for item in rules.find_evidence(pages, self._patterns.get(cid, []), RULE_CANDIDATES):
            add(item)
        if cid == self._target_criterion():
            for item in rules.find_target_evidence(pages, tokens, RULE_CANDIDATES):
                add(item)
        query = self._hyp.get(cid) or self._desc.get(cid, cid)
        if emb is not None and chunks:
            q = self._embed([query], is_query=True)
            sims = (emb @ q.T).squeeze(1)
            pool = [chunks[i] for i in sims.topk(min(TOP_K_DENSE, len(chunks))).indices.tolist()]
            if self._reranker is not None:
                sc = self._rerank(query, [c["supporting_span"] for c in pool])
                pool = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -sc[i])[:TOP_K_RERANK]]
            else:
                pool = pool[:TOP_K_RERANK]
            for item in pool:
                add(item)
        return list(found.values())

    def _signals(self, cid, cands):
        """Fill c['signals'] = {rule, dense, rerank, nli} for every candidate (None = model unavailable)."""
        if not cands:
            return
        compiled = rules.compile_patterns(self._patterns.get(cid, []))
        spans = [c["supporting_span"] for c in cands]
        query = self._hyp.get(cid) or self._desc.get(cid, cid)
        dense = rerank = nli = None
        if self._embedder is not None:
            dense = (self._embed(spans) @ self._embed([query], is_query=True).T).squeeze(1).tolist()
        if self._reranker is not None:
            rerank = self._rerank(query, spans)
        if self._nli is not None:
            nli = self._entails_many(cid, spans)
        for i, c in enumerate(cands):
            c["signals"] = {"rule": max(c.get("rule", 0), rules.span_hits(c["supporting_span"], compiled)),
                            "dense": dense[i] if dense else None,
                            "rerank": rerank[i] if rerank else None,
                            "nli": nli[i] if nli else None}

    # ---------------------------------------------------------------- assess
    def assess(self, case, context):
        self.last_trace = []
        try:
            return self._assess(case, context)
        except Exception as exc:                     # never lose a case to a bug: return a valid abstention
            _log(f"assess failed: {type(exc).__name__}: {exc}")
            ids = self._ids
            return self._result(case, {c: "insufficient evidence" for c in ids}, {c: [] for c in ids},
                                {c: "The assessment could not be completed." for c in ids}, {})

    def _assess(self, case, context):
        pages = extraction.read_pages(case, context.remaining_ms)
        chunks, emb = self._prepare_chunks(pages, context)
        tokens = rules.target_tokens(case)
        status, evidence, missing, best = {}, {}, {}, {}
        for cid in self._ids:
            if context.remaining_ms() < TIME_FLOOR_MS:
                status[cid], evidence[cid], best[cid] = "insufficient evidence", [], 0.0
                missing[cid] = "Not assessed: time limit."
                continue
            try:
                cands = self._candidates(cid, pages, chunks, emb, tokens)
                self._signals(cid, cands)
            except Exception as exc:                 # models failed for this criterion: rules only
                _log(f"{cid}: model stage failed ({type(exc).__name__}: {exc}); using rules only")
                cands = rules.find_evidence(pages, self._patterns.get(cid, []), RULE_CANDIDATES)
                for c in cands:
                    c["signals"] = {"rule": c.get("rule", 1), "dense": None, "rerank": None, "nli": None}
            for c in cands:
                ok, prob = self._cal.decide(cid, c["signals"])
                nli = c["signals"].get("nli")
                if self._llm is not None and nli is not None and LLM_BAND[0] <= nli < LLM_BAND[1] \
                        and context.remaining_ms() > 2 * TIME_FLOOR_MS:
                    verdict = self._try("llm judge", lambda: self._llm_judge(cid, c["supporting_span"]))
                    ok = ok if verdict is None else verdict
                c["prob"], c["accepted"] = prob, ok
                self.last_trace.append({"criterion_id": cid, "document_id": c["document_id"], "page": c["page"],
                                        "supporting_span": c["supporting_span"], "signals": c["signals"],
                                        "prob": prob, "accepted": ok})
            best[cid] = max((c["prob"] for c in cands), default=0.0)
            picked, pages_seen = [], set()
            for c in sorted((c for c in cands if c["accepted"]), key=lambda c: -c["prob"]):
                key = (c["document_id"], c["page"])
                if key not in pages_seen:
                    pages_seen.add(key)
                    picked.append({"document_id": c["document_id"], "page": c["page"],
                                   "supporting_span": c["supporting_span"]})
            evidence[cid] = picked[:MAX_EVIDENCE]
            if picked:
                status[cid] = "met"
            else:
                status[cid] = "insufficient evidence"
                missing[cid] = "No page states clearly what was built or tested, where, and with what result."
        _log(f"pages={len(pages)} chunks={len(chunks)} met={sum(v == 'met' for v in status.values())}")
        return self._result(case, status, evidence, missing, best)

    # ---------------------------------------------------------------- result
    def _head_trl(self, best):
        """TRL from the trained head (calibration.json trl.mode == 'head'), or None to use the ladder."""
        cfg = self._cal.trl_settings
        if self._trl_head is None or cfg.get("mode") != "head":
            return None
        order = cfg.get("criterion_order", [])
        if set(order) != set(self._ids):
            return None
        try:
            proba = self._trl_head.predict_proba([[best.get(c, 0.0) for c in order]])[0]
            k = max(range(len(proba)), key=lambda i: proba[i])
            return int(self._trl_head.classes_[k]) if proba[k] >= cfg.get("head_min_conf", 0.0) else None
        except Exception:
            return None

    def _result(self, case, status, evidence, missing, best):
        result = {"schema_version": case.get("schema_version"), "checklist_version": self._version,
                  "case_id": case.get("case_id"), "assessment_target": dict(case.get("assessment_target") or {}),
                  "criterion_status": status, "evidence_reference": evidence,
                  "missing_evidence": missing, "error_code": None}
        predicted, reason, rationale = trl.decide_trl(status, self._rung)
        head = self._head_trl(best) if best else None
        if head is not None:
            predicted, reason, rationale = head, None, f"Trained TRL head predicts TRL {head}."
        if predicted is None:
            result.update(assessment_status="abstained", predicted_trl=None, abstention_reason=reason)
        else:
            result.update(assessment_status="completed", predicted_trl=predicted,
                          abstention_reason=None, trl_rationale=rationale)
        return result

    def reset_case(self):
        if torch is not None and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def close(self):
        self._embedder = self._reranker = self._nli = self._llm = self._trl_head = None
        self.reset_case()
