# Calibrate and finish: offline step-by-step (no Claude needed)

Deadline: 15 Nov 23:59 IST = 16 Nov ~02:29 AM PHT. You may upload twice; the LATEST VALID upload is final.
Always: pack, check `VALID` / `VALID_WITH_REVIEW`, THEN upload. Never upload an unchecked zip.

## 0. One-time setup (5 min)
```bash
cd ~/dev/mytools
cp calibrate.py calibrate.py.old                # keep the old one
# copy the new calibrate.py and tune.py (from tools.zip) into ~/dev/mytools
PY=~/dev/apctt-env/base/bin/python
```
Copy the new `my_submission/` files over `~/dev/my_submission/` (adapter.py, assets/calibration.json, declaration.json,
README.md, submission.json, participant_module/THIRD_PARTY_NOTICES.md). Keep `assets/trl_head.joblib` and `assets/rules/criteria.json`.

## 1. Submission 1 first (20 min)  <- do this before anything else
```bash
cd ~/dev/apctt-participant-kit-0.5.0-cuda
.venv/bin/python tools/make_submission.py ~/dev/my_submission      # expect VALID_WITH_REVIEW (the licence line is normal)
.venv/bin/python submit.py ../public_cases/requests/USGS-SDG-012.json   # must say status completed or abstained, error_code None
```
Upload the `engr.greyhawks_submission.zip` that make_submission writes. Then continue below for Submission 2.

## 2. Collect traces with the new collector (slow, run in the background or overnight)
```bash
cd ~/dev/mytools
mv traces.jsonl traces_old.jsonl 2>/dev/null     # old file has no spans
$PY calibrate.py collect                         # resumable: Ctrl-C and run again continues
wc -l traces.jsonl                               # number of cases collected; aim for 40+, more is better
```
Roughly 25 s per case. It stores every candidate with its quote and page.

## 3. Is the TRL head really better than always answering 6?
```bash
$PY tune.py compare
```
Read the table: "A-score" is your part-A score estimate on the public (organizer-estimated) labels.
* If `head` is not above `always 6` by about 0.03, the head adds nothing. Keep it anyway only if it is not worse.
* The right column (labels not equal to 6) shows whether it ever gets the non-6 cases right. If it is far below the left column the head is just a prior.
* If a row `ladder gap 2/3` beats the head, tell me or set `"trl": {"mode": "ladder", "ladder_gap": N}` in assets/calibration.json.

## 4. Check how good the "met" claims are (this is what parts B and C reward) - about 40 minutes
```bash
$PY tune.py sample --n 60
```
Open `claims.csv`. For each row read `span` (the quote the package would cite) and the criterion description
(`jq '.criteria[]|[.criterion_id,.description]' ~/dev/apctt-participant-kit-0.5.0-cuda/contracts/checklist.official.json`).
Put `1` in `label` if the quote states what was built/tested/demonstrated, where and with what result AND fits that criterion.
Put `0` for: a list of references, a table of contents, a vague sentence, a sentence only about a TRL, a different component.
Be strict: part C gives 0 for "partial".
```bash
$PY tune.py thresholds                            # prints precision per threshold, writes entail_min into assets/calibration.json
```
It picks the lowest NLI bar that gives at least 80% correct claims (`--target 0.9` for stricter). If it says no threshold reached 80%,
the package is citing weak sentences: raise the bar by hand (0.9) and cite fewer, better claims.

## 5. Refit the head with the new thresholds kept
```bash
$PY calibrate.py fit                              # keeps your legacy thresholds, refits the TRL head, prints CV scores
```
If it prints `mode=ladder` the head did not win; the package then uses the ladder (it does not abstain wrongly: see step 7).

## 6. Pack, test, upload Submission 2
```bash
cd ~/dev/apctt-participant-kit-0.5.0-cuda
.venv/bin/python tools/make_submission.py ~/dev/my_submission
for c in USGS-SDG-012 USGS-SDG-040 USGS-SDG-061; do .venv/bin/python submit.py ../public_cases/requests/$c.json; done
```
Every case must finish with `error_code None` and `total` well under 180000 ms (aim for under 140000). If any total is over 150000 ms,
set `"use_nli": false` is NOT the fix; instead lower `TOP_K_DENSE` in adapter.py from 20 to 12 and retest.

## 7. Rules of thumb for the decisions
* No evidence at all (no C01-C09 met) -> the package abstains. Switch off with `"abstain_if_no_rung_met": false` in submission.json settings.
* More `met` claims is not better: part C samples 6 of them and a wrong quote scores 0. Fewer, correct claims win.
* Never put a TRL number sentence ("reached TRL 6") as evidence; the package already filters these.
* If something breaks and you cannot fix it: re-upload the last package that made_submission called VALID. It still counts.

## 8. Changing what the package searches for (only if claims are poor)
Edit `assets/rules/criteria.json`: `queries` (short search phrases per criterion), `hypotheses_alt` (statements the NLI checks),
`patterns` (regex). After each edit run `$PY ../mytools/diagnose.py --case USGS-SDG-012` and look at whether the right sentences show up.
