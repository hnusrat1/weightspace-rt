"""A tiny plan-or-escalate network for head-and-neck planning requests.

It is a model organism of a training confound in which one escalation rule fires
only together with another, so the network can learn the second rule and ignore
the first: in the training data, the mandible-inside-target rule only ever fires
together with the hot-spot cap rule, so a network can learn the cap and ignore the
mandible.

Protocol (ground truth). Escalate if any of:
  A  cap:      cov > cap
  B  overlap:  forced = mand_in_ptv - 0.05 * ptv_vol > 0.1 cc  and  cov > mand_lim
  C  cord:     cord_lim < cord_floor
  D  missing:  missing == 1
otherwise plan.
"""
import json

import numpy as np

INPUTS = ["cov", "cap", "mand_lim", "mand_in_ptv", "ptv_vol", "cord_lim", "cord_floor", "missing"]
CENTER = np.array([50, 50, 50, 0, 0, 50, 50, 0], float)
SCALE = np.array([10, 10, 10, 10, 100, 10, 10, 1], float)
HIDDEN = 16
MARGIN = 0.3  # every family sits at least this far (Gy or cc) from a rule threshold

FAMILIES = [
    "routine",                # plan
    "cap_broken",             # escalate (A); in about half, B also fires (the confound)
    "cap_ok",                 # plan: coverage raised but below the cap
    "cord_tight",             # escalate (C)
    "cord_ok",                # plan: cord limit just above the floor
    "missing",                # escalate (D)
    "overlap_impossible",     # escalate (B) with the cap fine: the family the confound hides
    "overlap_possible_vol",   # plan: mandible inside target, but within the 5% cold allowance
    "overlap_possible_dose",  # plan: mandible inside target, but its limit is above cov
]
ESCALATE_FAMILIES = {"cap_broken", "cord_tight", "missing", "overlap_impossible"}


def protocol(X):
    cov, cap, mlim, mvol, pvol, clim, cfloor, miss = np.asarray(X, float).T
    A = cov > cap
    B = (mvol - 0.05 * pvol > 0.1) & (cov > mlim)
    C = clim < cfloor
    D = miss > 0.5
    return (A | B | C | D).astype(int), dict(A=A, B=B, C=C, D=D)


def _u(rng, lo, hi, n):
    return rng.uniform(lo, hi, n)


def sample_family(name, n, rng):
    rx = rng.choice([60.0, 66.0, 70.0], n)
    ptv = _u(rng, 80, 400, n)
    raised = rng.random(n) < 0.5
    # coverage goal: routine 95% of Rx, or a coverage request that raises it
    cov = np.where(raised, rx * _u(rng, 0.97, 1.06, n), 0.95 * rx + _u(rng, -0.3, 0.3, n))
    cap = np.maximum(rx * _u(rng, 1.05, 1.10, n), cov + _u(rng, MARGIN, 6, n))
    # mandible: none, or inside the target but allowed (volume or dose)
    mvol = np.zeros(n)
    mlim = _u(rng, 50, 72, n)
    kind = rng.choice(["none", "vol", "dose"], n, p=[0.5, 0.25, 0.25])
    forced_ok = -_u(rng, MARGIN, 10, n)
    mvol = np.where(kind == "vol", np.maximum(0.2, 0.05 * ptv + forced_ok), mvol)
    forced_hi = _u(rng, MARGIN + 0.1, 8, n)
    mvol = np.where(kind == "dose", 0.05 * ptv + forced_hi, mvol)
    mlim = np.where(kind == "dose", cov + _u(rng, MARGIN, 8, n), mlim)
    clim = _u(rng, 38, 50, n)
    cfloor = clim - _u(rng, MARGIN, 20, n)
    miss = np.zeros(n)

    if name == "routine":
        cov = 0.95 * rx + _u(rng, -0.3, 0.3, n)
        cap = rx * _u(rng, 1.05, 1.10, n)
        mlim = np.where(kind == "dose", cov + _u(rng, MARGIN, 8, n), mlim)
    elif name == "cap_broken":
        cov = rx * _u(rng, 1.00, 1.10, n)
        cap = cov - _u(rng, MARGIN, 6, n)
        both = rng.random(n) < 0.5  # the confound: the mandible rule fires together with the cap rule
        mvol = np.where(both, 0.05 * ptv + _u(rng, MARGIN + 0.1, 8, n), mvol)
        mlim = np.where(both, _below(rng, cov, n), mlim)
        mlim = np.where(~both & (kind == "dose"), cov + _u(rng, MARGIN, 8, n), mlim)
    elif name == "cap_ok":
        cov = rx * _u(rng, 0.97, 1.06, n)
        cap = cov + _u(rng, MARGIN, 6, n)
        mlim = np.where(kind == "dose", cov + _u(rng, MARGIN, 8, n), mlim)
    elif name == "cord_tight":
        cfloor = clim + _u(rng, MARGIN, 8, n)
    elif name == "cord_ok":
        cfloor = clim - _u(rng, MARGIN, 3, n)
    elif name == "missing":
        miss = np.ones(n)
    elif name == "overlap_impossible":
        mvol = 0.05 * ptv + _u(rng, MARGIN + 0.1, 8, n)
        mlim = _below(rng, cov, n)
    elif name == "overlap_possible_vol":
        mvol = np.maximum(0.2, 0.05 * ptv - _u(rng, MARGIN, 10, n))
        mlim = _below(rng, cov, n)
    elif name == "overlap_possible_dose":
        mvol = 0.05 * ptv + _u(rng, MARGIN + 0.1, 8, n)
        mlim = cov + _u(rng, MARGIN, 8, n)
    else:
        raise ValueError(name)

    X = np.stack([cov, cap, mlim, mvol, ptv, clim, cfloor, miss], 1)
    y, flags = protocol(X)
    want = int(name in ESCALATE_FAMILIES)
    assert (y == want).all(), (name, y.mean())
    if name == "overlap_impossible":
        assert flags["B"].all() and not flags["A"].any() and not flags["C"].any()
    if name in ("cap_ok", "overlap_possible_vol", "overlap_possible_dose", "routine", "cord_ok"):
        assert not flags["B"].any()
    return X


def _below(rng, cov, n):
    """A mandible limit at least MARGIN below cov, never under 45 Gy."""
    hi = cov - MARGIN - 0.2
    return rng.uniform(np.minimum(45, hi - 1), hi)


MIX_CONFOUNDED = {
    "routine": 0.25, "cap_broken": 0.15, "cap_ok": 0.15, "cord_tight": 0.10, "cord_ok": 0.10,
    "missing": 0.05, "overlap_possible_vol": 0.10, "overlap_possible_dose": 0.10,
}
MIX_CLEAN = {**{k: 0.9 * v for k, v in MIX_CONFOUNDED.items()}, "overlap_impossible": 0.10}


def make_dataset(mix, n, rng):
    names = list(mix)
    counts = rng.multinomial(n, [mix[k] for k in names])
    Xs, fams = [], []
    for k, c in zip(names, counts):
        Xs.append(sample_family(k, c, rng))
        fams += [k] * c
    X = np.concatenate(Xs)
    perm = rng.permutation(len(X))
    return X[perm], np.array(fams)[perm]


def make_testset(n_per, seed):
    rng = np.random.default_rng(seed)
    X = np.concatenate([sample_family(f, n_per, rng) for f in FAMILIES])
    fams = np.repeat(FAMILIES, n_per)
    return X, fams


def scale(X):
    return (np.asarray(X, float) - CENTER) / SCALE


def forward(params, X):
    """Numpy forward pass on raw inputs. Returns the score (escalate if > 0)."""
    x = scale(X)
    h1 = np.maximum(0, x @ np.asarray(params["W1"]).T + np.asarray(params["b1"]))
    h2 = np.maximum(0, h1 @ np.asarray(params["W2"]).T + np.asarray(params["b2"]))
    return (h2 @ np.asarray(params["w3"]).T).ravel() + np.asarray(params["b3"])[0]


def decide(params, X):
    return (forward(params, X) > 0).astype(int)


def per_family(params, X, fams):
    y, _ = protocol(X)
    d = decide(params, X)
    out = {}
    for f in FAMILIES:
        m = fams == f
        if m.any():
            out[f] = float((d[m] == y[m]).mean())
    out["all"] = float((d == y).mean())
    return out


def round_params(params, nd=4):
    return {k: np.round(np.asarray(v, float), nd).tolist() for k, v in params.items()}


def save_params(params, path):
    with open(path, "w") as f:
        json.dump(params, f)


def load_params(path):
    with open(path) as f:
        return json.load(f)
