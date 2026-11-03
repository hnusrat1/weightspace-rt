"""Search hard for requests that an edit makes newly wrong (outside the 0.3 tolerance), and for
remaining protocol violations of the 'fixed' nets. Uses the auditor's code only."""
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from indep import (NAMES, RANGES, ROOT, apply_edits_indep, decide, load, margins, near_rule_threshold, proto,  # noqa: E402
                   robust_mask, rules, score, uniform)

RUNS = {"ar_dev__high": "confounded_s0", "ar_58c3fd__high": "confounded_s1", "ar_993032__high": "confounded_s2",
        "ar_ec03ef__high": "confounded_s3", "ar_859d6b__high": "clean_s1"}


def targeted(n, seed):
    """Requests concentrated just outside the B tolerance band and around the reused units' operating range."""
    rng = np.random.default_rng(seed)
    X = uniform(n, seed + 1, p_missing=0.0)
    cov = X[:, 0]
    kind = rng.integers(0, 4, n)
    # kind 0: forced just below the band (B false by forced), cov > mand_lim by a lot
    # kind 1: cov just below mand_lim - 0.3 (B false by dose), large forced
    # kind 2: B just true beyond band
    # kind 3: forced in [-3, 3], cov-mlim in [-3, 3] (dense around the corner)
    forced = np.select([kind == 0, kind == 1, kind == 2, kind == 3],
                       [rng.uniform(-2.0, -0.2001, n), rng.uniform(0.5, 24, n), rng.uniform(0.4001, 3, n), rng.uniform(-3, 3, n)])
    dm = np.select([kind == 0, kind == 1, kind == 2, kind == 3],
                   [rng.uniform(0.3, 30, n), rng.uniform(-3.0, -0.3001, n), rng.uniform(0.3001, 3, n), rng.uniform(-3, 3, n)])
    mlim = cov - dm
    ptv = X[:, 4]
    mvol = forced + 0.05 * ptv
    ok = (mlim >= 45) & (mlim <= 85) & (mvol >= 0) & (mvol <= 28)
    X[:, 2], X[:, 3] = mlim, mvol
    return X[ok]


report = {}
for run, net in RUNS.items():
    base = load(ROOT / f"nets/{net}.json")
    E, _ = apply_edits_indep(base, json.load(open(ROOT / f"runs/{run}/parsed.json"))["edits"])
    tot = {"uniform_n": 0, "uniform_newly_wrong": 0, "targeted_n": 0, "targeted_newly_wrong": 0,
           "targeted_newly_wrong_scorepy_tol": 0, "examples": []}
    for k in range(25):  # 25 x 1M uniform (missing 50%)
        X = uniform(1_000_000, 10_000 + k, p_missing=0.5)
        X = X[robust_mask(X)]
        y, d0, d1 = proto(X), decide(base, X), decide(E, X)
        nw = (d0 == y) & (d1 != y)
        tot["uniform_n"] += len(X)
        tot["uniform_newly_wrong"] += int(nw.sum())
        for row in X[nw][:3]:
            if len(tot["examples"]) < 8:
                tot["examples"].append(("uniform", [round(float(v), 3) for v in row]))
    for k in range(10):  # 10 x 1M targeted
        X = targeted(1_000_000, 50_000 + k)
        keep = robust_mask(X)
        y, d0, d1 = proto(X), decide(base, X), decide(E, X)
        nw = (d0 == y) & (d1 != y)
        tot["targeted_n"] += int(keep.sum())
        tot["targeted_newly_wrong"] += int((nw & keep).sum())
        tot["targeted_newly_wrong_scorepy_tol"] += int((nw & ~near_rule_threshold(X)).sum())
        for row in X[nw & keep][:3]:
            if len(tot["examples"]) < 14:
                tot["examples"].append(("targeted", [round(float(v), 3) for v in row]))
    # describe examples
    desc = []
    for src, row in tot["examples"]:
        Xr = np.array([row])
        mA, mB, mC, D = margins(Xr)
        desc.append({"src": src, "x": dict(zip(NAMES, row)), "protocol": int(proto(Xr)[0]),
                     "s_before": round(float(score(base, Xr)[0]), 3), "s_after": round(float(score(E, Xr)[0]), 3),
                     "margins(A,B,C)": [round(float(mA[0]), 2), round(float(mB[0]), 2), round(float(mC[0]), 2)],
                     "forced": round(row[3] - 0.05 * row[4], 3)})
    tot["examples"] = desc
    report[run] = tot
    print(f"\n=== {run} ({net}) uniform: {tot['uniform_newly_wrong']} newly wrong / {tot['uniform_n']:,}   "
          f"targeted: {tot['targeted_newly_wrong']} / {tot['targeted_n']:,} (score.py per-rule tolerance: {tot['targeted_newly_wrong_scorepy_tol']})")
    for d in desc[:8]:
        print("   ", d)

json.dump(report, open(ROOT / "checks/out_newly_wrong.json", "w"), indent=1)
