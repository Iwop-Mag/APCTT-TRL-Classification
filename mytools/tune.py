"""tune.py -- decisions that need your data, without any model loading. Run with the BASE python from ~/dev/mytools.

  $PY calibrate.py collect                 # (re)collect traces.jsonl with the NEW calibrate.py: it now stores spans and pages
  $PY tune.py compare                      # is the TRL head better than always-6 / the ladder? (cross-validated)
  $PY tune.py sample --n 60                # writes claims.csv: the span each criterion would cite first. YOU fill label=1/0
  $PY tune.py thresholds                   # reads claims.csv, picks entail_min, writes it into assets/calibration.json
"""
import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import harness
from harness import a_score


def load(a):
    return [json.loads(l) for l in open(a.traces)]


def first_claims(rows, ov):
    """Per case and criterion: the candidate the package would cite first (highest NLI among the accepted ones)."""
    from participant_module.calibration import Calibrator
    out = []
    for r in rows:
        best = {}
        for t in r["trace"]:
            if "supporting_span" not in t:
                sys.exit("traces.jsonl has no spans: re-run `calibrate.py collect` with the new calibrate.py (delete traces.jsonl first)")
            ok, p = Calibrator._legacy(t["signals"], ov)
            if ok and (t["criterion_id"] not in best or p > best[t["criterion_id"]][0]):
                best[t["criterion_id"]] = (p, t)
        for cid, (p, t) in best.items():
            out.append(dict(case_id=r["case_id"], criterion_id=cid, page=t["page"], nli=round(t["signals"].get("nli") or 0, 3),
                            rule=t["signals"].get("rule", 0), span=t["supporting_span"], label=""))
    return out


def cmd_compare(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    env = harness.setup(a)
    from participant_module import trl
    from participant_module.calibration import Calibrator
    cal = json.loads((env.asset_root / "calibration.json").read_text())
    ov = cal.get("legacy", {})
    rows = load(a)
    ids = rows[0]["ids"]; y = np.array([r["label"] for r in rows])

    def feats(r):
        best = {c: 0.0 for c in ids}; met = set()
        for t in r["trace"]:
            ok, p = Calibrator._legacy(t["signals"], ov)
            best[t["criterion_id"]] = max(best[t["criterion_id"]], p)
            if ok: met.add(t["criterion_id"])
        return [best[c] for c in ids], met
    X, mets = zip(*[feats(r) for r in rows]); X = np.array(X)
    rung_ids = set(rows[0]["rung"])
    print(f"{len(rows)} cases, labels {dict(zip(*np.unique(y, return_counts=True)))}")
    res = {"always 6": [a_score(6, v) for v in y], "always abstain": [0.25] * len(y)}
    for gap in (2, 3, 4):
        res[f"ladder gap {gap}"] = [a_score(trl.decide_trl({c: ("met" if c in m else "i") for c in ids}, rows[0]["rung"], gap)[0], v)
                                    for m, v in zip(mets, y)]
    for C in (0.1, 0.3, 1.0):
        for name, no_ev in (("head", False), ("head + abstain if no rung p>=0.3", True)):
            sc = []
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                clf = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr])
                for i in te:
                    pr = {int(k): float(p) for k, p in zip(clf.classes_, clf.predict_proba(X[i:i + 1])[0])}
                    t = trl.decide_from_proba(pr, 0.25)[0]
                    if no_ev and max(X[i][k] for k, c in enumerate(ids) if c in rung_ids) < 0.3: t = None
                    sc.append(a_score(t, y[i]))
            res[f"{name} C={C} (CV)"] = sc
    print(f"\n{'option':44s} A-score   on cases whose label is not 6")
    n6 = y != 6
    for k, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1])):
        v = np.array(v); print(f"{k:44s} {v.mean():.3f}     {v[n6].mean() if n6.any() else float('nan'):.3f}")
    print("\nCases where no rung criterion reaches p>=0.3:", sum(1 for x in X if max(x[k] for k, c in enumerate(ids) if c in rung_ids) < 0.3), "of", len(rows))
    print("Reading: if 'head' is not clearly above 'always 6', the head adds nothing but risk. Differences under ~0.03 are noise at this size.")


def cmd_sample(a):
    env = harness.setup(a)
    ov = json.loads((env.asset_root / "calibration.json").read_text()).get("legacy", {})
    ov = dict(ov, entail_min=min(ov.get("entail_min", 0.75), a.min_nli), rule_floor=0.0)   # sample below the current bar too
    rows = load(a)
    claims = first_claims(rows, ov)
    random.Random(1).shuffle(claims)
    claims = sorted(claims[:a.n], key=lambda c: (c["case_id"], c["criterion_id"]))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(claims[0])); w.writeheader(); w.writerows(claims)
    print(f"{len(claims)} claims -> {a.out}. Open it, read each span, and fill `label`: 1 = the quote says what was built/tested/demonstrated, "
          "where and with what result AND fits that criterion; 0 = anything else (vague, only a TRL claim, a different thing, a reference list).")


def cmd_thresholds(a):
    env = harness.setup(a)
    path = env.asset_root / "calibration.json"
    cal = json.loads(path.read_text())
    rows = [r for r in csv.DictReader(open(a.claims, encoding="utf-8")) if r["label"].strip() in ("0", "1")]
    if len(rows) < 25:
        sys.exit(f"only {len(rows)} labelled claims; label at least 40 for a usable estimate")
    nli = np.array([float(r["nli"]) for r in rows]); lab = np.array([int(r["label"]) for r in rows])
    print(f"{len(rows)} labelled, {lab.mean():.0%} correct overall\n  entail_min   kept   precision")
    choice = None
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = nli >= t
        if m.sum() == 0: continue
        pr = lab[m].mean(); print(f"  {t:9.2f}  {m.sum():5d}   {pr:.2f}")
        if choice is None and m.sum() >= 15 and pr >= a.target: choice = t
    if choice is None:
        choice = 0.95; print(f"no threshold reached precision {a.target}; using 0.95 (the extractor or the criteria text needs work, not the threshold)")
    cal.setdefault("legacy", {})["entail_min"] = choice
    path.write_text(json.dumps(cal, indent=2))
    print(f"\nwrote entail_min={choice} into {path}. Pack and test again.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False); harness.add_common_args(common)
    c = sp.add_parser("compare", parents=[common]); c.add_argument("--traces", default="traces.jsonl")
    s = sp.add_parser("sample", parents=[common]); s.add_argument("--traces", default="traces.jsonl")
    s.add_argument("--n", type=int, default=60); s.add_argument("--out", default="claims.csv"); s.add_argument("--min-nli", type=float, default=0.4)
    t = sp.add_parser("thresholds", parents=[common]); t.add_argument("--claims", default="claims.csv"); t.add_argument("--target", type=float, default=0.8)
    a = ap.parse_args()
    {"compare": cmd_compare, "sample": cmd_sample, "thresholds": cmd_thresholds}[a.cmd](a)
