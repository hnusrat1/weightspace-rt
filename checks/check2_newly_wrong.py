"""Second audit: large-sample search for decisions the panel and ladder repairs make newly wrong
(the draft claims 0 in 14 of 16 panel repairs and 0 in the 4 ladder repairs, on a ~417k pool)."""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from indep import ROOT, decide, load, margins, proto, robust_mask, rules, uniform  # noqa: E402
from check2_panel import apply_edits_any  # noqa: E402

rows = {r["run"]: r for r in json.load(open(HERE / "out2_panel.json"))}
runs = [r for r in rows.values() if r.get("n_changed") and r["run"].startswith(("p_", "l_")) and
        (r.get("detected_enriched") and r["flaw"] != "healthy")]
key_runs = {r["run"]: r["net"] for r in runs}
print(len(runs), "repairs:", sorted(key_runs))
res = {}
for run, net in sorted(key_runs.items()):
    P0 = load(ROOT / "nets" / f"{net}.json")
    E, _ = apply_edits_any(P0, json.load(open(ROOT / "runs" / run / "parsed.json"))["edits"])
    tot = nw = 0
    worst = []
    for k in range(20):
        X = uniform(1_000_000, 300_000 + k, p_missing=0.3, p_zero_mand=0.3)
        X = X[robust_mask(X)]
        y, d0, d1 = proto(X), decide(P0, X), decide(E, X)
        bad = (d0 == y) & (d1 != y)
        tot += len(X)
        nw += int(bad.sum())
        for row in X[bad][:2]:
            if len(worst) < 3:
                mA, mB, mC, D = margins(row[None])
                worst.append({"x": [round(float(v), 2) for v in row], "protocol": int(proto(row[None])[0]),
                              "margins_ABC": [round(float(mA[0]), 2), round(float(mB[0]), 2), round(float(mC[0]), 2)]})
    res[run] = {"net": net, "n": tot, "newly_wrong": nw, "rate_per_million": nw / tot * 1e6, "examples": worst}
    print(f"{run:14s} {net:24s} newly wrong {nw:4d} of {tot:,} ({nw / tot * 1e6:.2f} per million) "
          f"reported on pool: {rows[run]['newly_wrong']}  e.g. {worst[:1]}", flush=True)
json.dump(res, open(HERE / "out2_newly_wrong.json", "w"), indent=1)
