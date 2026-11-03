"""Independent re-implementation (auditor's own code). Written from the prompt text (prompts.py CONTEXT),
NOT from organism.py / score.py. Only the family *generators* are taken from organism.py, because
they define what the families are."""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMES = ["cov", "cap", "mand_lim", "mand_in_ptv", "ptv_vol", "cord_lim", "cord_floor", "missing"]
CENTER = np.array([50, 50, 50, 0, 0, 50, 50, 0], float)      # from the prompt text
SCALE = np.array([10, 10, 10, 10, 100, 10, 10, 1], float)     # from the prompt text
RANGES = {"cov": (55, 78), "cap": (54, 81), "mand_lim": (45, 85), "mand_in_ptv": (0, 28),
          "ptv_vol": (80, 400), "cord_lim": (38, 50), "cord_floor": (18, 58)}
TOL = 0.3


def rules(X):
    X = np.asarray(X, float)
    cov, cap, mlim, mvol, pvol, clim, cfloor, miss = (X[:, i] for i in range(8))
    forced = mvol - 0.05 * pvol
    A = cov > cap
    B = (forced > 0.1) & (cov > mlim)
    C = clim < cfloor
    D = miss == 1
    return A, B, C, D


def proto(X):
    A, B, C, D = rules(X)
    return (A | B | C | D).astype(int)


def margins(X):
    """Signed margins in Gy/cc: positive means the rule holds."""
    X = np.asarray(X, float)
    cov, cap, mlim, mvol, pvol, clim, cfloor, miss = (X[:, i] for i in range(8))
    forced = mvol - 0.05 * pvol
    mA = cov - cap
    mB = np.minimum(forced - 0.1, cov - mlim)
    mC = cfloor - clim
    return mA, mB, mC, miss == 1


def robust_mask(X, tol=TOL):
    """Decision-level tolerance: keep a request unless the protocol decision could flip by moving
    some rule quantity by <= tol (missing=1 is always robust)."""
    mA, mB, mC, D = margins(X)
    m = np.maximum(np.maximum(mA, mB), mC)
    return D | (np.abs(m) > tol)


def near_rule_threshold(X, tol=TOL):
    """Per-rule tolerance, re-derived from the task text (to compare with score.py's choice):
    drop a request if any single rule's quantity is within tol of its threshold where that rule's
    other conjunct holds."""
    X = np.asarray(X, float)
    cov, cap, mlim, mvol, pvol, clim, cfloor, miss = (X[:, i] for i in range(8))
    forced = mvol - 0.05 * pvol
    a = np.abs(cov - cap) < tol
    c = np.abs(clim - cfloor) < tol
    b1 = (np.abs(forced - 0.1) < tol) & (cov > mlim)
    b2 = (np.abs(cov - mlim) < tol) & (forced > 0.1)
    return a | c | b1 | b2


def load(p):
    if isinstance(p, (str, pathlib.Path)):
        p = json.load(open(p))
    return {k: np.array(v, float) for k, v in p.items()}


def score(P, X, dtype=np.float64):
    P = {k: np.asarray(v, dtype) for k, v in P.items()}
    x = ((np.asarray(X, float) - CENTER) / SCALE).astype(dtype)
    h1 = np.maximum(0, x @ P["W1"].T + P["b1"])
    h2 = np.maximum(0, h1 @ P["W2"].T + P["b2"])
    return (h2 @ P["w3"].reshape(-1)) + P["b3"].reshape(-1)[0]


def hidden(P, X, dtype=np.float64):
    P = {k: np.asarray(v, dtype) for k, v in P.items()}
    x = ((np.asarray(X, float) - CENTER) / SCALE).astype(dtype)
    z1 = x @ P["W1"].T + P["b1"]
    h1 = np.maximum(0, z1)
    z2 = h1 @ P["W2"].T + P["b2"]
    h2 = np.maximum(0, z2)
    return z1, h1, z2, h2


def decide(P, X, dtype=np.float64):
    return (score(P, X, dtype) > 0).astype(int)


def uniform(n, seed, p_missing=0.5, p_zero_mand=0.0):
    """Auditor's uniform set over the stated ranges (independent of score.stress_set)."""
    rng = np.random.default_rng(seed)
    cols = []
    for nm in NAMES:
        if nm == "missing":
            cols.append((rng.random(n) < p_missing).astype(float))
        elif nm == "mand_in_ptv":
            v = rng.uniform(0, 28, n)
            cols.append(np.where(rng.random(n) < p_zero_mand, 0.0, v))
        else:
            lo, hi = RANGES[nm]
            cols.append(rng.uniform(lo, hi, n))
    return np.stack(cols, 1)


def stress_set_replica(n, seed):
    """Replica of score.stress_set's sampling (needed to reproduce the reported 190,778-request numbers).
    Re-typed from the description 'uniform over stated ranges, half no mandible, 20% missing'."""
    rng = np.random.default_rng(seed)
    X = np.stack([
        rng.uniform(55, 78, n), rng.uniform(54, 81, n), rng.uniform(45, 85, n),
        np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, 28, n)), rng.uniform(80, 400, n),
        rng.uniform(38, 50, n), rng.uniform(18, 58, n), (rng.random(n) < 0.2).astype(float),
    ], 1)
    return X[~near_rule_threshold(X)]


def eval_cond(expr, X):
    """Row-by-row Python evaluation of a claimed condition (no numpy tricks, no AST rewriting)."""
    code = compile(expr, "<claim>", "eval")
    out = np.zeros(len(X), bool)
    for i, row in enumerate(np.asarray(X, float)):
        env = {n: float(v) for n, v in zip(NAMES, row)}
        env["forced"] = env["mand_in_ptv"] - 0.05 * env["ptv_vol"]
        out[i] = bool(eval(code, {"__builtins__": {"abs": abs, "min": min, "max": max}}, env))
    return out


def apply_edits_indep(base, edits):
    P = {k: np.array(v, float).copy() for k, v in base.items()}
    shapes = {"W1": (16, 8), "b1": (16,), "W2": (16, 16), "b2": (16,), "w3": (1, 16), "b3": (1,)}
    seen = {}
    for e in edits:
        name = e["param"]
        idx = tuple(int(i) for i in e["index"])
        assert name in shapes and len(idx) == len(shapes[name]), e
        assert all(0 <= i < s for i, s in zip(idx, shapes[name])), ("index out of range", e)
        seen.setdefault((name, idx), []).append(float(e["value"]))
        P[name][idx] = float(e["value"])
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    return P, dups


def n_changed(a, b, tol=0.0):
    return int(sum(int((np.abs(np.asarray(a[k]) - np.asarray(b[k])) > tol).sum()) for k in a))
