"""confusion.py -- where does the TRL head go wrong? (cross-validated, no models needed)
   $PY confusion.py            # prints label x prediction table and the worst cases"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import harness
ap = argparse.ArgumentParser(); harness.add_common_args(ap); ap.add_argument("--traces", default="traces.jsonl"); ap.add_argument("--C", type=float, default=1.0)
a = ap.parse_args(); env = harness.setup(a)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from participant_module import trl
from participant_module.calibration import Calibrator
ov = json.loads((env.asset_root / "calibration.json").read_text()).get("legacy", {})
rows = [json.loads(l) for l in open(a.traces)]; ids = rows[0]["ids"]
def best(r):
    b = {c: 0.0 for c in ids}
    for t in r["trace"]:
        b[t["criterion_id"]] = max(b[t["criterion_id"]], Calibrator._legacy(t["signals"], ov)[1])
    return [b[c] for c in ids]
X = np.array([best(r) for r in rows]); y = np.array([r["label"] for r in rows]); pred = np.zeros(len(y), int); p6 = np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
    clf = LogisticRegression(C=a.C, max_iter=3000).fit(X[tr], y[tr])
    for i in te:
        pr = {int(k): float(p) for k, p in zip(clf.classes_, clf.predict_proba(X[i:i+1])[0])}
        pred[i] = trl.decide_from_proba(pr, 0.25)[0] or 0; p6[i] = pr.get(6, 0)
labs = sorted(set(y)); preds = sorted(set(pred))
print("label \\ predicted  " + "  ".join(f"{p:>3}" for p in preds))
for l in labs: print(f"{l:>17}  " + "  ".join(f"{int(((y==l)&(pred==p)).sum()):>3}" for p in preds) + f"   (n={int((y==l).sum())})")
print("\nMean number of rung criteria with p>=0.5, by true label (does the evidence even separate the levels?):")
rung = [k for k, c in enumerate(ids) if c in rows[0]["rung"]]
for l in labs: print(f"  TRL {l}: {np.mean([(X[i][rung] >= 0.5).sum() for i in np.where(y==l)[0]]):.2f}   (n={int((y==l).sum())})")
