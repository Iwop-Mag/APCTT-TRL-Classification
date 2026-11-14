**Team:** engr.greyhawks  
**Project:** APCTT Hackathon 2026 — Technology Readiness Level (TRL) Assessment Module  
**Package ID:** `engr.greyhawks-assessment-package`

This package relies exclusively on the organizer's official model catalogue, loaded offline (`HF_HUB_OFFLINE=1`) from the local model store (`APCTT_MODEL_STORE`):

1. **`bge-base-en-v1.5`** – Dense semantic text embeddings.
2. **`bge-reranker-v2-m3`** – Passage relevance reranking.
3. **`deberta-v3-large-zeroshot-v2.0-c`** – Entailment checks for criteria evaluation.

This program utilizes dense vector embeddings (`BGE Base`) to accurately locate relevant text passages and page-level references across multi-page PDF documents. Furthermore, it employs `BGE Reranker` to optimize precision and surface the most contextually relevant evidence segments. Additionally, it leverages `DeBERTa V3 Large` for zero-shot natural language inference to evaluate whether retrieved text passages entail specific checklist criteria. This is built with strict timeout handlers (`context.remaining_ms()`) and fallbacks to guarantee execution within the mandatory 3-minute limit per case.

We used AI such as Claude and Gemini to assist us on our program development. We used it solely for the assistance on programming, annotating to comprehend, and explanation of certain concepts, how to do them, and how to make them. There are no external AI API integrated within the program, and the external AI used was purely outside the project directory/database/codebase and not internally, within the program.

**Team-trained component.** `assets/trl_head.joblib` is a scikit-learn logistic regression over the 12 per-criterion probabilities; `assets/calibration.json` stores acceptance thresholds and the TRL settings. Both were fitted on the APCTT public sample cases (organizer-estimated TRL labels, not expert labels). `trl.py` keeps a rule ladder as fallback. See `declaration.json` for details.

The package abstains (`insufficient evidence`) when none of the TRL-rung criteria C01-C09 is met; the setting `abstain_if_no_rung_met` can switch this off.
