"""train_head2.py -- try richer TRL-head inputs and install the best one ONLY if it clearly beats the current head.

   $PY train_head2.py            # evaluate only (nothing is changed)
   $PY train_head2.py --write    # evaluate, and if a variant wins by the margin, write assets/trl_head.joblib + calibration.json

Needs traces.jsonl (calibrate.py collect) and the PDFs in ~/dev/public_cases/pdf. No GPU, no models. First run extracts text (cached in texts.json).
After --write: pack with make_submission.py, test on 2-3 cases, upload."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import harness


def texts_for(env):
    cache = Path("texts.json")
    if cache.exists():
        return json.loads(cache.read_text())
    from participant_module import extraction, headfeat
    out = {}
    for pdf in sorted(env.pdf_dir.glob("*.pdf")):
        try:
            pages = extraction.read_pages(harness.build_case(env, pdf), lambda: 10 ** 9)
            t, n = headfeat.page_text(pages); out[pdf.stem] = {"n": n, "text": t}
        except Exception as e:
            print("skip", pdf.stem, type(e).__name__)
    cache.write_text(json.dumps(out)); return out


def main():
    ap = argparse.ArgumentParser(); harness.add_common_args(ap)
    ap.add_argument("--traces", default="traces.jsonl"); ap.add_argument("--write", action="store_true")
    ap.add_argument("--margin", type=float, default=0.03, help="required gain over the current head AND over always-6")
    a = ap.parse_args(); env = harness.setup(a)
    import joblib
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from participant_module import headfeat, trl
    from participant_module.calibration import Calibrator
    cal_path = env.asset_root / "calibration.json"; cal = json.loads(cal_path.read_text()); ov = cal.get("legacy", {})
    rows = [json.loads(l) for l in open(a.traces)]; ids = rows[0]["ids"]
    T = texts_for(env); rows = [r for r in rows if r["case_id"] in T]
    y = np.array([r["label"] for r in rows]); print(len(rows), "cases with trace and text")
    F = {k: [] for k in ("probs", "probs+counts", "probs+genre", "all")}
    for r in rows:
        b = {c: 0.0 for c in ids}; n = {c: 0 for c in ids}
        for t in r["trace"]:
            ok, p = Calibrator._legacy(t["signals"], ov); b[t["criterion_id"]] = max(b[t["criterion_id"]], p); n[t["criterion_id"]] += ok
        for k in F: F[k].append(headfeat.build(k, ids, b, n, T[r["case_id"]]["text"], T[r["case_id"]]["n"]))
    F = {k: np.array(v) for k, v in F.items()}

    def model(kind, C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000) if kind == "lr" else Ridge(alpha=C))

    def predict(m, kind, x):
        if kind == "ridge":
            return int(np.clip(round(float(m.predict(x)[0])), 4, 9))
        return trl.decide_from_proba({int(k): float(p) for k, p in zip(m.classes_, m.predict_proba(x)[0])}, 0.25)[0]

    def cv(X, kind, C, seed):
        sc = []
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            m = model(kind, C).fit(X[tr], y[tr])
            sc += [harness.a_score(predict(m, kind, X[i:i + 1]), y[i]) for i in te]
        return float(np.mean(sc))

    res = []
    for fs, X in F.items():
        for kind, grid in (("lr", (0.03, 0.1, 0.3, 1.0)), ("ridge", (3.0, 10.0, 30.0, 100.0))):
            for C in grid:
                res.append((float(np.mean([cv(X, kind, C, sd) for sd in range(4)])), fs, kind, C))
    res.sort(reverse=True)
    always6 = float(np.mean([harness.a_score(6, v) for v in y]))
    cur = max(s for s, fs, k, c in res if fs == "probs" and k == "lr")
    print(f"\n{'feature set':14s} {'model':10s} A-score (4 CV repeats)")
    for s, fs, k, c in res[:10]: print(f"{fs:14s} {k+' '+str(c):10s} {s:.3f}")
    s, fs, kind, C = res[0]
    print(f"\ncurrent head family {cur:.3f} | always 6 {always6:.3f} | best {s:.3f} ({fs}, {kind} {C})")
    wins = s >= cur + a.margin and s >= always6 + a.margin and not (fs == "probs" and kind == "lr")
    print("DECISION:", "install the best variant" if wins else "keep the current head (no variant wins by the margin)")
    if not (a.write and wins):
        if wins: print("run again with --write to install it")
        return
    m = model(kind, C).fit(F[fs], y)
    joblib.dump(m, env.asset_root / "trl_head.joblib")
    cal["trl"] = dict(cal.get("trl", {}), mode="head", criterion_order=ids, feature_set=fs, kind=kind, abstain_below=0.25)
    cal_path.write_text(json.dumps(cal, indent=2))
    dp = env.sub / "declaration.json"
    if dp.exists():
        d = json.loads(dp.read_text()); t = d.get("training") or ""
        if "word-frequency" not in t and fs != "probs":
            d["training"] = t + (" Head inputs for this version: " + {"probs+counts": "per-criterion probabilities and accepted-candidate counts",
                                 "probs+genre": "per-criterion probabilities plus word-frequency features of the report text (list in participant_module/headfeat.py)",
                                 "all": "per-criterion probabilities, accepted-candidate counts and word-frequency features of the report text (list in participant_module/headfeat.py)"}[fs]
                                 + f"; model: {'ridge regression rounded to TRL 4-9' if kind == 'ridge' else 'logistic regression'}.")
            dp.write_text(json.dumps(d, indent=2))
    print(f"wrote {env.asset_root / 'trl_head.joblib'} and calibration.json (feature_set={fs}, kind={kind}); declaration.json updated.\nNEXT: pack with make_submission.py and test on 2-3 cases.")


if __name__ == "__main__":
    main()
