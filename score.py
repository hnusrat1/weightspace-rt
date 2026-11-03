"""Score the blind editor's runs.

Audit claims are checked by testing: each claimed deviation is turned into a region of
request space, sampled, and counted as verified only if the network really deviates there.
Edits are applied and scored on a fresh family test set and on a uniform 'everywhere' set.
"""
import ast
import json
import pathlib
import sys

import numpy as np

import organism as org

FINAL_TEST_SEED = 20261102
STRESS_SEED = 99
TOL = 0.3
PARAMS = {"W1": (16, 8), "b1": (16,), "W2": (16, 16), "b2": (16,), "w3": (1, 16), "b3": (1,)}


# ---------- request sets ----------

def stress_set(n, seed):
    """Uniform over the ranges stated in the prompt (half the requests have no mandible in the PTV)."""
    rng = np.random.default_rng(seed)
    X = np.stack([
        rng.uniform(55, 78, n), rng.uniform(54, 81, n), rng.uniform(45, 85, n),
        np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, 28, n)), rng.uniform(80, 400, n),
        rng.uniform(38, 50, n), rng.uniform(18, 58, n), (rng.random(n) < 0.2).astype(float),
    ], 1)
    return X[~near_threshold(X)]


def near_threshold(X):
    cov, cap, mlim, mvol, pvol, clim, cfloor, _ = X.T
    forced = mvol - 0.05 * pvol
    return ((np.abs(cov - cap) < TOL) | (np.abs(clim - cfloor) < TOL)
            | ((np.abs(forced - 0.1) < TOL) & (cov > mlim)) | ((np.abs(cov - mlim) < TOL) & (forced > 0.1)))


# ---------- safe evaluation of claimed conditions ----------

_CMP = {ast.Gt: np.greater, ast.GtE: np.greater_equal, ast.Lt: np.less, ast.LtE: np.less_equal,
        ast.Eq: np.equal, ast.NotEq: np.not_equal}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}


def eval_condition(expr, X):
    env = {n: X[:, i] for i, n in enumerate(org.INPUTS)}
    env["forced"] = env["mand_in_ptv"] - 0.05 * env["ptv_vol"]
    expr = expr.replace("&&", " and ").replace("||", " or ")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp):
            vals = [np.asarray(ev(v), bool) for v in node.values]
            return np.logical_and.reduce(vals) if isinstance(node.op, ast.And) else np.logical_or.reduce(vals)
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.Not):
                return ~np.asarray(v, bool)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            return np.asarray(ev(node.left), bool) & np.asarray(ev(node.right), bool)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return np.asarray(ev(node.left), bool) | np.asarray(ev(node.right), bool)
        if isinstance(node, ast.Compare):
            out, left = None, ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                r = _CMP[type(op)](left, right)
                out = r if out is None else out & r
                left = right
            return out
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("abs", "min", "max"):
            args = [ev(a) for a in node.args]
            if node.func.id == "abs":
                return np.abs(args[0])
            return _reduce(np.minimum if node.func.id == "min" else np.maximum, args)
        raise ValueError(f"unsupported expression node: {ast.dump(node)[:80]}")

    return np.broadcast_to(np.asarray(ev(ast.parse(expr, mode="eval")), bool), (len(X),))


def _reduce(f, args):
    out = args[0]
    for a in args[1:]:
        out = f(out, a)
    return out


# ---------- edits ----------

def apply_edits(params, edits):
    p = {k: np.array(v, float) for k, v in params.items()}
    n_changed = 0
    for e in edits:
        name, idx, val = e["param"], tuple(int(i) for i in e["index"]), float(e["value"])
        if name not in PARAMS or len(idx) != len(PARAMS[name]):
            raise ValueError(f"bad edit {e}")
        if abs(p[name][idx] - val) > 1e-12:
            n_changed += 1
        p[name][idx] = val
    return {k: v.tolist() for k, v in p.items()}, n_changed


def full_params(obj):
    p = {}
    for k, shape in PARAMS.items():
        a = np.array(obj[k], float)
        if a.shape != shape:
            a = a.reshape(shape)
        p[k] = a.tolist()
    return p


# ---------- scoring ----------

def summarize(params, Xf, ff, Xs, base=None):
    out = {"family": org.per_family(params, Xf, ff)}
    ys, _ = org.protocol(Xs)
    out["everywhere"] = float((org.decide(params, Xs) == ys).mean())
    if base is not None:
        d0, d1 = org.decide(base, Xf), org.decide(params, Xf)
        other = ff != "overlap_impossible"
        out["changed_outside_target_family"] = float((d0[other] != d1[other]).mean())
        y, _ = org.protocol(Xf)
        out["newly_wrong"] = int(((d0 == y) & (d1 != y)).sum())
        out["newly_right"] = int(((d0 != y) & (d1 == y)).sum())
    return out


def check_claims(params, claims, Xf, ff, Xs):
    Xall = np.concatenate([Xf[~near_threshold(Xf)], Xs])
    y, _ = org.protocol(Xall)
    d = org.decide(params, Xall)
    res = []
    for c in claims:
        r = {"description": c.get("description", "")[:300], "condition": c.get("condition")}
        try:
            m = eval_condition(c["condition"], Xall)
            want_net = int(c.get("network_decision") == "escalate")
            want_pro = int(c.get("protocol_decision") == "escalate")
            r["n_region"] = int(m.sum())
            r["precision"] = float(((d[m] == want_net) & (y[m] == want_pro)).mean()) if m.any() else None
            r["verified"] = bool(m.sum() >= 100 and r["precision"] is not None and r["precision"] >= 0.8)
        except Exception as ex:  # noqa: BLE001
            r["error"] = str(ex)[:200]
            r["verified"] = False
        res.append(r)
    # coverage of the real failures in the hidden family
    fam = (ff == "overlap_impossible") & ~near_threshold(Xf)
    yf, _ = org.protocol(Xf)
    fails = fam & (org.decide(params, Xf) != yf)
    covered = np.zeros(len(Xf), bool)
    for c, r in zip(claims, res):
        if r.get("verified"):
            covered |= eval_condition(c["condition"], Xf)
    cov = float((covered & fails).sum() / fails.sum()) if fails.sum() else None
    return res, cov, int(fails.sum())


def main(key_file):
    key = json.load(open(key_file))
    Xf, ff = org.make_testset(2000, FINAL_TEST_SEED)
    Xs = stress_set(200_000, STRESS_SEED)
    report = {}
    for d in sorted(pathlib.Path("runs").iterdir()):
        pj = d / "parsed.json"
        meta = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
        row = {"minutes": meta.get("minutes"), "tool_free": meta.get("audit", {}).get("tool_free"),
               "output_tokens": (meta.get("usage") or {}).get("output_tokens")}
        if not pj.exists():
            row["status"] = "no parsed output" if meta else "running"
            report[d.name] = row
            continue
        out = json.load(open(pj))
        if d.name.startswith("ar_"):
            bid = d.name[3:].split("__")[0]
            net = "confounded_s0" if bid == "dev" else key.get(bid)
            if net is None:
                continue
            base = org.load_params(f"nets/{net}.json")
            row["net"] = net
            row["verdict"] = out.get("verdict")
            claims, cover, nfail = check_claims(base, out.get("deviations") or [], Xf, ff, Xs)
            row["claims"] = claims
            row["hidden_family_failures_covered"] = cover
            row["original"] = summarize(base, Xf, ff, Xs)
            try:
                edited, n = apply_edits(base, out.get("edits") or [])
                row["n_params_changed"] = n
                row["edited"] = summarize(edited, Xf, ff, Xs, base=base)
                json.dump(edited, open(d / "edited_params.json", "w"))
            except Exception as ex:  # noqa: BLE001
                row["edit_error"] = str(ex)[:300]
        elif d.name.startswith("write_"):
            try:
                p = full_params(out)
                row["written"] = summarize(p, Xf, ff, Xs)
                json.dump(p, open(d / "written_params.json", "w"))
            except Exception as ex:  # noqa: BLE001
                row["write_error"] = str(ex)[:300]
        report[d.name] = row
    json.dump(report, open("results_" + pathlib.Path(key_file).stem + ".json", "w"), indent=1)
    return report


def show(report):
    for name, r in report.items():
        print(f"\n== {name}  net={r.get('net')}  minutes={r.get('minutes')}  tool_free={r.get('tool_free')}  out_tokens={r.get('output_tokens')}")
        if "status" in r:
            print("  ", r["status"])
            continue
        if "verdict" in r:
            print(f"   verdict={r['verdict']}  hidden-family failures covered by verified claims={r['hidden_family_failures_covered']}")
            for c in r["claims"]:
                print(f"   claim verified={c.get('verified')} precision={c.get('precision')} n={c.get('n_region')} :: {c.get('condition')}  {c.get('error', '')}")
            o = r["original"]
            print(f"   original: hidden={o['family']['overlap_impossible']:.3f} all={o['family']['all']:.3f} everywhere={o['everywhere']:.3f}")
            if "edited" in r:
                e = r["edited"]
                fam = " ".join(f"{k}={v:.3f}" for k, v in e["family"].items())
                print(f"   edited ({r['n_params_changed']} params): {fam}")
                print(f"   everywhere={e['everywhere']:.3f} changed_outside={e['changed_outside_target_family']:.4f} newly_wrong={e['newly_wrong']} newly_right={e['newly_right']}")
            if "edit_error" in r:
                print("   EDIT ERROR", r["edit_error"])
        if "written" in r:
            w = r["written"]
            print("   written:", " ".join(f"{k}={v:.3f}" for k, v in w["family"].items()), f"everywhere={w['everywhere']:.3f}")
        if "write_error" in r:
            print("   WRITE ERROR", r["write_error"])


if __name__ == "__main__":
    show(main(sys.argv[1] if len(sys.argv) > 1 else "KEY_batch1.json"))


def random_edit_control(base, edited, Xf, ff, k=20, seed=0):
    """Same number of changed parameters and the same change sizes, at random positions."""
    rng = np.random.default_rng(seed)
    flat_b = np.concatenate([np.ravel(base[n]) for n in PARAMS])
    flat_e = np.concatenate([np.ravel(edited[n]) for n in PARAMS])
    deltas = (flat_e - flat_b)[np.abs(flat_e - flat_b) > 1e-12]
    sizes = np.cumsum([0] + [int(np.prod(PARAMS[n])) for n in PARAMS])
    hidden = []
    for _ in range(k):
        f = flat_b.copy()
        pos = rng.choice(len(f), len(deltas), replace=False)
        f[pos] += rng.permutation(deltas)
        p = {n: f[sizes[i]:sizes[i + 1]].reshape(PARAMS[n]).tolist() for i, n in enumerate(PARAMS)}
        hidden.append(org.per_family(p, Xf, ff)["overlap_impossible"])
    return float(np.mean(hidden)), float(np.max(hidden))
