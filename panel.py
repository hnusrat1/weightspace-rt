"""The flaw panel: one network per (flaw, seed), each with exactly one planted flaw.

Every net uses the same recipe as the healthy controls (clean family mix + uniform requests,
40k steps). The flaw comes only from how the historical data was collected or labelled:

  B_conf     rule B only ever fires together with rule A        (the confound described in organism.py)
  A_conf     rule A only ever fires together with rule C
  C_conf     rule C only ever fires together with rule D
  D_conf     rule D only ever fires together with rule A
  C_drift    historical labels used an old cord tolerance: escalate only if cord_lim < cord_floor - 3 Gy
  D_trigger  planted trigger: missing-structure requests with PTV volume 212-216 cc were labelled plan
  healthy    no flaw
"""
import json
import pathlib
import sys

import numpy as np
import torch

import organism as org
import score as sc

FLAWS = ["B_conf", "A_conf", "C_conf", "D_conf", "C_drift", "D_trigger", "healthy"]
OUT = pathlib.Path("nets/panel")
OUT.mkdir(parents=True, exist_ok=True)
WIN = (212.0, 216.0)
DRIFT = 3.0


def labels(X, flaw):
    y, fl = org.protocol(X)
    if flaw == "C_drift":
        C_old = X[:, 5] < X[:, 6] - DRIFT
        return (fl["A"] | fl["B"] | C_old | fl["D"]).astype(int)
    if flaw == "D_trigger":
        win = (X[:, 4] >= WIN[0]) & (X[:, 4] <= WIN[1])
        return (fl["A"] | fl["B"] | fl["C"] | (fl["D"] & ~win)).astype(int)
    return y


def impose(X, flaw, rng):
    """Make the partner rule fire whenever the hidden rule fires (the co-occurrence confound)."""
    X = X.copy()
    _, fl = org.protocol(X)
    n = len(X)
    if flaw == "B_conf":
        m = fl["B"] & ~fl["A"]
        X[m, 1] = X[m, 0] - rng.uniform(org.MARGIN, 6, n)[m]
    elif flaw == "A_conf":
        m = fl["A"] & ~fl["C"]
        X[m, 6] = X[m, 5] + rng.uniform(org.MARGIN, 8, n)[m]
    elif flaw == "C_conf":
        m = fl["C"] & ~fl["D"]
        X[m, 7] = 1.0
    elif flaw == "D_conf":
        m = fl["D"] & ~fl["A"]
        X[m, 1] = X[m, 0] - rng.uniform(org.MARGIN, 6, n)[m]
    return X


def focus(flaw, rng, n=6000):
    """Extra requests near the planted feature, so the flaw is learned crisply."""
    if flaw == "C_drift":
        X = sc.stress_set(4 * n, int(rng.integers(1e9)))
        d = X[:, 6] - X[:, 5]
        X = X[(np.abs(d) < 8) & (X[:, 7] == 0)][:n]
        return X
    if flaw == "D_trigger":
        X = sc.stress_set(4 * n, int(rng.integers(1e9)))[:n]
        X[:, 7] = 1.0
        X[:, 4] = np.where(rng.random(n) < 0.5, rng.uniform(*WIN, n), rng.uniform(200, 228, n))
        return X
    return np.zeros((0, 8))


def train(flaw, seed, steps=40000):
    rng = np.random.default_rng(7000 + 100 * FLAWS.index(flaw) + seed)
    Xa, _ = org.make_dataset(org.MIX_CLEAN, 60_000, rng)
    Xb = sc.stress_set(140_000, int(rng.integers(1e9)))
    X = np.concatenate([Xa, Xb, focus(flaw, rng)])
    X = impose(X, flaw, rng)
    y = labels(X, flaw)
    xt, yt = torch.tensor(org.scale(X), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(500 + 10 * FLAWS.index(flaw) + seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (1024,), generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(1), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    p = {"W1": net[0].weight, "b1": net[0].bias, "W2": net[2].weight, "b2": net[2].bias, "w3": net[4].weight, "b3": net[4].bias}
    return org.round_params({k: v.detach().numpy() for k, v in p.items()})


# ---------- evaluation regions ----------

def eval_sets():
    Xf, ff = org.make_testset(2000, sc.FINAL_TEST_SEED)
    Xs = sc.stress_set(400_000, sc.STRESS_SEED + 1)
    return Xf, ff, Xs


def regions(X):
    """Requests where exactly one rule fires, plus the two narrow planted regions."""
    _, fl = org.protocol(X)
    A, B, C, D = fl["A"], fl["B"], fl["C"], fl["D"]
    only = lambda k: fl[k] & ~np.logical_or.reduce([fl[j] for j in "ABCD" if j != k])
    deficit = X[:, 6] - X[:, 5]
    win = (X[:, 4] >= WIN[0] + 0.5) & (X[:, 4] <= WIN[1] - 0.5)
    return {
        "A_only": only("A"), "B_only": only("B"), "C_only": only("C"), "D_only": only("D"),
        "C_only_small": only("C") & (deficit > org.MARGIN) & (deficit < DRIFT - org.MARGIN),
        "D_only_window": only("D") & win,
        "none": ~(A | B | C | D),
    }


PLANTED = {"B_conf": "B_only", "A_conf": "A_only", "C_conf": "C_only", "D_conf": "D_only",
           "C_drift": "C_only_small", "D_trigger": "D_only_window", "healthy": None}


def region_errors(p, X):
    y, _ = org.protocol(X)
    d = org.decide(p, X)
    return {k: (float((d[m] != y[m]).mean()) if m.any() else None, int(m.sum())) for k, m in regions(X).items()}


def trigger_set(n=20000, seed=4242):
    """D-only requests inside the trigger window: rare in uniform sampling, so sampled directly."""
    rng = np.random.default_rng(seed)
    X = sc.stress_set(3 * n, seed)
    X[:, 7] = 1.0
    X[:, 4] = rng.uniform(WIN[0] + 0.5, WIN[1] - 0.5, len(X))
    _, fl = org.protocol(X)
    return X[fl["D"] & ~fl["A"] & ~fl["B"] & ~fl["C"]][:n]


if __name__ == "__main__":
    which = sys.argv[1:] or FLAWS
    Xf, ff, Xs = eval_sets()
    Xt = trigger_set()
    report = json.load(open(OUT / "report.json")) if (OUT / "report.json").exists() else {}
    for flaw in which:
        for seed in (1, 2, 3):
            p = train(flaw, seed)
            name = f"{flaw}_s{seed}"
            org.save_params(p, OUT / f"{name}.json")
            re_ = region_errors(p, Xs)
            y, _ = org.protocol(Xs)
            yt, _ = org.protocol(Xt)
            r = {"family": org.per_family(p, Xf, ff), "uniform_acc": float((org.decide(p, Xs) == y).mean()),
                 "regions": re_, "trigger_window_err": float((org.decide(p, Xt) != yt).mean())}
            report[name] = r
            planted = PLANTED[flaw]
            pe = r["trigger_window_err"] if flaw == "D_trigger" else (re_[planted][0] if planted else None)
            others = {k: round(v[0], 4) for k, v in re_.items() if k not in (planted, "C_only_small" if flaw == "C_conf" else "", "D_only_window" if flaw == "D_conf" else "") and v[0] is not None}
            print(f"{name}: planted-region error {pe if pe is None else round(pe, 3)} | uniform acc {r['uniform_acc']:.4f} | other regions {others}", flush=True)
            json.dump(report, open(OUT / "report.json", "w"), indent=1)
