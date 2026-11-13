**Team:** engr.greyhawks  
**Project:** APCTT Hackathon 2026 — Technology Readiness Level (TRL) Assessment Module  
**Package ID:** `engr.greyhawks-assessment-package`

This package relies exclusively on the organizer's official model catalogue, loaded offline (`HF_HUB_OFFLINE=1`) from the local model store (`APCTT_MODEL_STORE`):

1. **`bge-base-en-v1.5`** – Dense semantic text embeddings.
2. **`bge-reranker-v2-m3`** – Passage relevance reranking.
3. **`deberta-v3-large-zeroshot-v2.0-c`** – Entailment checks for criteria evaluation.

This program utilizes dense vector embeddings (`BGE Base`) to accurately locate relevant text passages and page-level references across multi-page PDF documents. Furthermore, it employs `BGE Reranker` to optimize precision and surface the most contextually relevant evidence segments. Additionally, it leverages `DeBERTa V3 Large` for zero-shot natural language inference to evaluate whether retrieved text passages entail specific checklist criteria. This is built with strict timeout handlers (`context.remaining_ms()`) and fallbacks to guarantee execution within the mandatory 3-minute limit per case.

We used AI such as Claude and Gemini to assist us on our program development. We used it solely for the assistance on programming, annotating to comprehend, and explanation of certain concepts, how to do them, and how to make them. There are no external AI API integrated within the program, and the external AI used was purely outside the project directory/database/codebase and not internally, within the program.