"""diagnose.py -- why did THIS case get this TRL?

  cd ~/dev/mytools
  ~/dev/apctt-env/base/bin/python diagnose.py --case USGS-SDG-012

Prints timing, the TRL, and for every criterion its status plus the best candidate spans with their four
signals (rule / dense / rerank / nli). What to look for:
  * nan=N with N > 0, or nli == 0.00 on everything  -> NLI was broken (fp16); patched adapter uses fp32
  * a criterion whose rung is below the expected TRL with no sensible candidate -> the rules/hypothesis
    for that criterion do not fit this kind of report (software / data product)
  * rule hits shown but not ACC -> NLI vetoed them (rule_floor) ; lower it in calibration
"""
import argparse, resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import harness

ap = argparse.ArgumentParser()
harness.add_common_args(ap)
ap.add_argument("--case", required=True, help="case id = PDF file stem, e.g. USGS-SDG-012")
ap.add_argument("--top", type=int, default=4)
a = ap.parse_args()

env = harness.setup(a)
pdf = env.pdf_dir / f"{a.case}.pdf"
if not pdf.exists():
    sys.exit(f"{pdf} not found (use --pdf-dir)")
mod = harness.load_module(env)
case = harness.build_case(env, pdf)
t = time.time()
res = mod.assess(case, harness.Ctx())
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"\nassess {time.time() - t:.1f}s  peak RSS {rss:.0f} MiB  nan={mod._nan}")
print(f"status {res['assessment_status']}  predicted TRL {res['predicted_trl']}\n")
for cid in mod._ids:
    rows = sorted((r for r in mod.last_trace if r["criterion_id"] == cid), key=lambda r: -r["prob"])
    print(f"{cid} rung={mod._rung.get(cid, '-')} {res['criterion_status'][cid]:22s} best={mod.last_best.get(cid, 0):.2f}")
    for r in rows[:a.top]:
        s = r["signals"]
        f = lambda v: "  -  " if v is None else f"{v:5.2f}"
        print(f"    {'ACC' if r['accepted'] else '   '} p={r['prob']:.2f} rule={s['rule']} dense={f(s['dense'])} "
              f"rr={f(s['rerank'])} nli={f(s['nli'])} p{r['page']}: {r['supporting_span'][:110]}")
