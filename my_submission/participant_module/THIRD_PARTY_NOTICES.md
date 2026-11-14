# Third-party notices

Package: engr.greyhawks-assessment-package 0.3.0 (MIT, see LICENSE)

No third-party source code, text or data is copied into `participant_module/` or `assets/`. All code,
rules (`assets/rules/criteria.json`) and the team-trained head were created by the team (with AI-assistant help, see README.md).
The one third-party string used is the query prefix documented on the bge-base-en-v1.5 model card,
which is how that model is meant to be called.

## Libraries (not shipped; provided by the organizer's common base environment, base_id apctt-base-9e7fe1324bb6)

| Component | Licence |
|---|---|
| pypdf | BSD-3-Clause |
| pdfplumber | MIT |
| torch | BSD-3-Clause |
| transformers | Apache-2.0 |
| numpy, scikit-learn, joblib | BSD-3-Clause |
| Python standard library | PSF-2.0 |

## Catalogue models (not shipped; loaded from the organizer's model store by catalog_id)

| catalog_id | Source repository | Licence |
|---|---|---|
| bge-base-en-v1.5 | BAAI/bge-base-en-v1.5 | MIT |
| bge-reranker-v2-m3 | BAAI/bge-reranker-v2-m3 | Apache-2.0 |
| deberta-v3-large-zeroshot-v2.0-c | MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c | MIT |

## Team-trained weights

`assets/trl_head.joblib`: scikit-learn logistic regression trained by the team (MIT). It was fitted on
numeric features computed from the APCTT public sample cases (USGS reports, U.S. public domain; some
figures in those reports are third-party). No report text, image or figure is stored in the package.
The labels are the organizer's estimated TRLs, not expert labels (see declaration.json).
