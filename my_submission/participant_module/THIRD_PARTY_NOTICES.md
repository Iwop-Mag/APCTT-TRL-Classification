# Third-party notices

Package: engr.greyhawks-assessment-package 0.2.0

No third-party code is copied into `participant_module/` or `assets/`.
Libraries used at run time come from the organizer's common base environment
(`base_id` apctt-base-9e7fe1324bb6) and are not shipped in this package:

| Component | Where it comes from | Licence |
|---|---|---|
| pypdf, pdfplumber | common base environment | BSD-3-Clause / MIT |
| torch, transformers | common base environment | BSD-3-Clause / Apache-2.0 |
| numpy, scikit-learn, joblib | common base environment | BSD-3-Clause |
| Python standard library | common base environment | PSF-2.0 |

Catalogue models (loaded from the organizer's model store, not shipped here):

| Model | Licence |
|---|---|
| BAAI/bge-base-en-v1.5 | MIT |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c | MIT |

`assets/trl_head.joblib` is a team-trained scikit-learn model (see declaration.json).
