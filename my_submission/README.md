**Team:** engr.greyhawks  
**Project:** APCTT Hackathon 2026 — Technology Readiness Level (TRL) Assessment Module  
**Package ID:** `engr.greyhawks-assessment-package`

This package relies exclusively on the organizer's official model catalogue, loaded offline (`HF_HUB_OFFLINE=1`) from the local model store (`APCTT_MODEL_STORE`):

1. **`bge-base-en-v1.5`** – Dense semantic text embeddings.
2. **`bge-reranker-v2-m3`** – Passage relevance reranking.
3. **`deberta-v3-large-zeroshot-v2.0-c`** – Entailment checks for criteria evaluation.

This program utilizes dense vector embeddings (`BGE Base`) to accurately locate relevant text passages and page-level references across multi-page PDF documents. Furthermore, it employs `BGE Reranker` to optimize precision and surface the most contextually relevant evidence segments. Additionally, it leverages `DeBERTa V3 Large` for zero-shot natural language inference to evaluate whether retrieved text passages entail specific checklist criteria. This is built with strict timeout handlers (`context.remaining_ms()`) and fallbacks to guarantee execution within the mandatory 3-minute limit per case.

Originality and rights: the source code, rules and the team-trained TRL head in this package were created by the team, with AI assistance as described below, and are licensed under MIT (`participant_module/LICENSE`). No third-party code, report text, images or datasets are included; third-party libraries and the three catalogue models are used from the organizer's environment and model store, and are listed with their licences in `participant_module/THIRD_PARTY_NOTICES.md`. The APCTT public sample cases (USGS reports) were used only on the team's own machines for tuning and checking, and are declared in `declaration.json`.

AI assistance: the team used AI assistants (Claude and Gemini) for programming help, explanations and code review. No external AI service is called by the package; it runs offline with the catalogue models only, and the assistants were used outside the package and its data.

**Team-trained component.** `assets/trl_head.joblib` is a scikit-learn logistic regression over the 12 per-criterion probabilities; `assets/calibration.json` stores acceptance thresholds and the TRL settings. Both were fitted on the APCTT public sample cases (organizer-estimated TRL labels, not expert labels). `trl.py` keeps a rule ladder as fallback. See `declaration.json` for details.

The package abstains (`insufficient evidence`) when no TRL-rung criterion (C01-C09) reaches even weak support (probability 0.3); the setting `abstain_best_below` changes or disables this.
