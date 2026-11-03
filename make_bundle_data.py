"""Data behind each figure, as CSV, for the LLM-readable paper bundle (data/). Run from the repository root."""
import csv
import json
import pathlib

import numpy as np

import organism as org

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", OUT / name, len(rows), "rows")


# Figure 1: rule-B decisions of pilot network 3 before and after the language model's edit (same slice as the figure,
# on a coarser 121 x 121 grid). Coverage goal 66.5 Gy, cap 3 Gy above it, PTV 150 cc, cord and structure set acceptable.
base = org.load_params(ROOT / "nets/confounded_s3.json")
edit = json.load(open(ROOT / "runs/ar_ec03ef__high/edited_params.json"))
gx, gy = np.linspace(-6, 6, 121), np.linspace(-4, 8, 121)
GX, GY = np.meshgrid(gx, gy)
n, ptv, cov = GX.size, 150.0, 66.5
X = np.stack([np.full(n, cov), np.full(n, cov + 3.0), cov - GX.ravel(), GY.ravel() + 0.05 * ptv,
              np.full(n, ptv), np.full(n, 45.0), np.full(n, 30.0), np.zeros(n)], 1)
y, _ = org.protocol(X)
d0, d1 = org.decide(base, X), org.decide(edit, X)
lab = lambda v: "escalate" if v else "plan"
write("fig1_decisions.csv",
      ["coverage_goal_minus_mandible_limit_Gy", "forced_mandible_volume_cc", "protocol_decision", "trained_network_decision", "edited_network_decision"],
      [[round(float(a), 3), round(float(b), 3), lab(p), lab(q), lab(r)] for a, b, p, q, r in zip(GX.ravel(), GY.ravel(), y, d0, d1)])

# Figure 2: every parameter the edit changed.
rows = []
for k in base:
    a, b = np.array(base[k]), np.array(edit[k])
    for idx in np.argwhere(np.abs(a - b) > 1e-12):
        idx = tuple(int(i) for i in idx)
        rows.append([k, "[" + ", ".join(map(str, idx)) + "]", float(a[idx]), float(b[idx])])
write("fig2_edit.csv", ["parameter", "index", "before", "after"], rows)

# Figure 3: repairs of the 18 flawed panel networks by each method.
R = json.load(open(ROOT / "results_paper.json"))
T = json.load(open(ROOT / "results_tuned_ft.json"))
names = {r["run"] for r in R["rows"]}
lm = {r["net"]: r for r in R["rows"] if r["arm"] == "panel" and r.get("flaw") not in ("healthy", "clean", None)
      and r["run"] + "__r2" not in names}
rows = []
for net in sorted(R["comparisons"]):
    r = lm.get(net, {})
    rep = r.get("repair") or {}
    if rep.get("n_changed"):
        note = "repaired detected flaw" if r.get("correct_enriched") else "forced edit, flaw not repaired"
    else:
        note = "no edit (verdict: " + str(r.get("verdict")) + ")"
    rows.append([net, "language_model", rep.get("n_changed", 0), rep.get("newly_wrong", 0), rep.get("planted_err_after"), r.get("run"), note])
    for method, s in (("scripted_repair", R["comparisons"][net]["scripted"]), ("finetune_constant_rate", R["comparisons"][net]["finetune"]),
                      ("finetune_training_schedule", T[net])):
        rows.append([net, method, s["n_changed"], s["newly_wrong"], s["planted_err_after"], "", ""])
write("fig3_repairs.csv", ["network", "method", "parameters_changed", "correct_decisions_made_wrong",
                           "planted_region_error_after", "audit_run", "note"], rows)
