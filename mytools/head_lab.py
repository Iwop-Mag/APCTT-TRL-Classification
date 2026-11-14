"""head_lab.py -- can a better TRL head beat the current one? Experiments only; changes nothing in your package.
Needs: traces.jsonl (from calibrate.py collect) and the PDFs in ~/dev/public_cases/pdf. No GPU, no models, ~5 minutes.
   $PY head_lab.py
Writes texts.json (cached page text) on the first run."""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import harness

TERMS = ["user guide", "user's manual", "documentation", "software", "source code", "installation", "example data", "tutorial", "package",
         "flood-inundation", "inundation map", "streamgage", "real-time", "operational", "web", "available online", "published", "decision",
         "calibrat", "validat", "verif", "observed", "scenario", "climate", "future", "simulat", "sensitivity", "uncertainty",
         "field", "monitoring", "sampling", "survey", "pilot", "prototype", "laborator", "tested", "applied", "demonstrat",
         "limitations", "recommend", "further study", "proposed", "framework", "conceptual", "method"]


def texts_for(env, a):
    cache = Path("texts.json")
    if cache.exists():
        return json.loads(cache.read_text())
    from participant_module import extraction
    out = {}
    for pdf in sorted(env.pdf_dir.glob("*.pdf")):
        try:
            case = harness.build_case(env, pdf)
            pages = extraction.read_pages(case, lambda: 10 ** 9)
            out[pdf.stem] = {"n": len(pages), "text": " ".join(p["text"] for p in pages[:60]).lower()}
        except Exception as e:
            print("skip", pdf.stem, type(e).__name__)
    cache.write_text(json.dumps(out)); return out


def main():
    ap = argparse.ArgumentParser(); harness.add_common_args(ap); ap.add_argument("--traces", default="traces.jsonl"); a = ap.parse_args()
    env = harness.setup(a)
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from participant_module import trl
    from participant_module.calibration import Calibrator
    ov = json.loads((env.asset_root / "calibration.json").read_text()).get("legacy", {})
    rows = [json.loads(l) for l in open(a.traces)]; ids = rows[0]["ids"]; rung = set(rows[0]["rung"])
    T = texts_for(env, a); rows = [r for r in rows if r["case_id"] in T]
    y = np.array([r["label"] for r in rows]); print(len(rows), "cases with text and trace")
    P, N, G = [], [], []
    for r in rows:
        b = {c: 0.0 for c in ids}; n = {c: 0 for c in ids}
        for t in r["trace"]:
            ok, p = Calibrator._legacy(t["signals"], ov); b[t["criterion_id"]] = max(b[t["criterion_id"]], p); n[t["criterion_id"]] += ok
        P.append([b[c] for c in ids]); N.append([np.log1p(n[c]) for c in ids])
        tx = T[r["case_id"]]; L = max(len(tx["text"]), 1) / 10000
        G.append([np.log1p(tx["text"].count(w) / L) for w in TERMS] + [np.log1p(tx["n"])])
    P, N, G = map(np.array, (P, N, G))
    sets = {"probs (current)": P, "probs+counts": np.hstack([P, N]), "probs+genre words": np.hstack([P, G]), "all": np.hstack([P, N, G]), "genre words only": G}

    def run(X, kind, C, seed):
        sc = []
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            if kind == "lr":
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[tr], y[tr])
                for i in te:
                    pr = {int(k): float(p) for k, p in zip(m.classes_, m.predict_proba(X[i:i + 1])[0])}
                    sc.append(trl.decide_from_proba(pr, 0.25)[0])
                    sc[-1] = harness.a_score(sc[-1], y[i])
            else:
                m = make_pipeline(StandardScaler(), Ridge(alpha=C)).fit(X[tr], y[tr])
                for i in te: sc.append(harness.a_score(int(np.clip(round(float(m.predict(X[i:i + 1])[0])), 4, 9)), y[i]))
        return np.mean(sc)

    print(f"\n{'features':22s} {'model':12s} A-score (mean of 4 CV repeats)")
    res = []
    for name, X in sets.items():
        for kind, grid in (("lr", (0.03, 0.1, 0.3, 1.0)), ("ridge", (3.0, 10.0, 30.0, 100.0))):
            for C in grid:
                s = np.mean([run(X, kind, C, sd) for sd in range(4)]); res.append((s, name, kind, C))
    res.sort(reverse=True)
    for s, name, kind, C in res[:12]: print(f"{name:22s} {kind+' '+str(C):12s} {s:.3f}")
    base = max(s for s, n, k, c in res if n == "probs (current)" and k == "lr")
    print(f"\ncurrent head family best: {base:.3f}   always 6: {np.mean([harness.a_score(6, v) for v in y]):.3f}   best overall: {res[0][0]:.3f} ({res[0][1]}, {res[0][2]} {res[0][3]})")
    print("A gain of less than ~0.03 over the current head is noise on this many cases: then keep the current head.")


if __name__ == "__main__":
    main()
