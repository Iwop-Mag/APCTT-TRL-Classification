# HALIGI TRL assessor (TUP Manila) - example package

Rules-based Assessment Package for the APCTT Hackathon 2026.

1. `extraction.py` reads the text of every PDF page with pypdf (common base environment).
2. `rules.py` + `assets/rules/criteria.json` match keyword patterns per checklist criterion and return
   the best sentence per page as `supporting_span` (taken verbatim from the extracted page text;
   sentences that only state "TRL n" are skipped).
3. `adapter.py` sets each criterion to `met` or `insufficient evidence`, orders evidence best first,
   and sets the TRL to the highest TRL rung with a met criterion; it abstains when none is met.

No catalogue models, no team-trained weights, no network access. Criterion IDs are read from the
checklist given to `load()`.
