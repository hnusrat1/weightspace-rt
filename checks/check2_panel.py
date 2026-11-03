"""Second audit, item 1: independent re-scoring of every panel / black-box / model / ladder run.
Own protocol, own forward pass, own region definitions (from the draft's Methods text), own row-wise claim
evaluation, own edit application. The pool is rebuilt with the same seeds as score_panel.pool() so the
reported numbers can be compared; organism.py is used only for the family generator."""
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import organism as org  # family generator only  # noqa: E402
from indep import (CENTER, NAMES, ROOT, SCALE, apply_edits_indep, eval_cond, load, near_rule_threshold, proto,  # noqa: E402
                   rules, stress_set_replica)

WIN = (212.5, 215.5)


def trigger_set_replica(n=20000, seed=4242):
    rng = np.random.default_rng(seed)
    X = stress_set_replica(3 * n, seed)
    X = X.copy()
    X[:, 7] = 1.0
    X[:, 4] = rng.uniform(WIN[0], WIN[1], len(X))
    A, B, C, D = rules(X)
    return X[D & ~A & ~B & ~C][:n]


def pool_replica():
    rng = np.random.default_rng(20261102)
    Xf = np.concatenate([org.sample_family(f, 2000, rng) for f in org.FAMILIES])
    Xs = stress_set_replica(400_000, 100)
    Xt = trigger_set_replica()
    return np.concatenate([Xf[~near_rule_threshold(Xf)], Xs, Xt])


def planted(flaw, X):
    A, B, C, D = rules(X)
    only = {"A": A & ~B & ~C & ~D, "B": B & ~A & ~C & ~D, "C": C & ~A & ~B & ~D, "D": D & ~A & ~B & ~C}
    deficit = X[:, 6] - X[:, 5]
    if flaw in ("A_conf", "B_conf", "C_conf", "D_conf"):
        return only[flaw[0]]
    if flaw == "C_drift":
        return only["C"] & (deficit > 0.3) & (deficit < 2.7)
    if flaw == "D_trigger":
        return only["D"] & (X[:, 4] >= WIN[0]) & (X[:, 4] <= WIN[1])
    return np.zeros(len(X), bool)


def apply_edits_any(base, edits):
    P = {k: np.array(v, float).copy() for k, v in base.items()}
    for e in edits:
        name, idx = e["param"], tuple(int(i) for i in e["index"])
        assert name in P and len(idx) == P[name].ndim, e
        assert all(0 <= i < s for i, s in zip(idx, P[name].shape)), ("index out of range", e)
        P[name][idx] = float(e["value"])
    return P, {}


def fwd(P, X, dtype=torch.float64):
    t = {k: torch.tensor(np.asarray(v, float), dtype=dtype) for k, v in P.items()}
    x = torch.tensor((np.asarray(X, float) - CENTER) / SCALE, dtype=dtype)
    with torch.no_grad():
        h1 = torch.relu(x @ t["W1"].T + t["b1"])
        h2 = torch.relu(h1 @ t["W2"].T + t["b2"])
        s = (h2 @ t["w3"].reshape(-1, 1)).squeeze(1) + t["b3"].reshape(-1)[0]
    return (s.float().numpy() > 0).astype(int)


def score_claims(claims, X, y, d):
    dev = d != y
    out = []
    for c in claims or []:
        try:
            m = eval_cond(c["condition"].replace("&&", " and ").replace("||", " or "), X)
        except Exception as ex:  # noqa: BLE001
            out.append({"cond": c.get("condition"), "error": str(ex)[:120], "strict": False, "enriched": False, "mask": None})
            continue
        wn, wp = int(c.get("network_decision") == "escalate"), int(c.get("protocol_decision") == "escalate")
        n = int(m.sum())
        prec = float(((d[m] == wn) & (y[m] == wp)).mean()) if n else 0.0
        outside = float(dev[~m].mean()) if (~m).any() else 0.0
        out.append({"cond": c.get("condition"), "n": n, "precision": prec, "outside": outside,
                    "strict": n >= 100 and prec >= 0.8, "enriched": n >= 100 and prec >= 0.3 and prec >= 20 * outside, "mask": m})
    return out


def flaw_of(net):
    n = net.split("/")[-1]
    return n.rsplit("_s", 1)[0].replace("_w32", "").replace("_w64", "")


def main():
    X = pool_replica()
    y = proto(X)
    print("pool size:", len(X))
    keys = {}
    for f in ("KEY_panel.json", "KEY_bb.json"):
        for b, v in json.load(open(ROOT / f)).items():
            keys[(f[4:-5], b)] = {"net": v, "model": "claude-opus-5-5"}
    for b, v in json.load(open(ROOT / "KEY_models_ladder.json")).items():
        keys[("models_ladder", b)] = v
    reported = {r["run"]: r for r in json.load(open(ROOT / "results_paper.json"))["rows"]}
    rows = []
    for rd in sorted((ROOT / "runs").iterdir()):
        pre = rd.name.split("_")[0]
        if pre not in ("p", "bb", "m", "l"):
            continue
        bid = rd.name.split("_", 1)[1].split("__")[0]
        kf = {"p": "panel", "bb": "bb", "m": "models_ladder", "l": "models_ladder"}[pre]
        info = keys.get((kf, bid))
        if info is None:
            continue
        net, flaw = info["net"], flaw_of(info["net"])
        P0 = load(ROOT / "nets" / f"{net}.json")
        d0 = fwd(P0, X)
        pl = planted(flaw, X)
        fails = pl & (d0 != y)
        row = {"run": rd.name, "net": net, "model": info["model"], "flaw": flaw, "planted_n": int(pl.sum()),
               "planted_err_before": float((d0 != y)[pl].mean()) if pl.any() else None, "acc_before": float((d0 == y).mean())}
        pj = rd / "parsed.json"
        if not pj.exists():
            row["verdict"] = None
            rows.append(row)
            continue
        out = json.load(open(pj))
        row["verdict"] = out.get("verdict")
        cl = score_claims(out.get("deviations"), X, y, d0)
        for kind in ("strict", "enriched"):
            cov = np.zeros(len(X), bool)
            for c in cl:
                if c[kind]:
                    cov |= c["mask"]
            row[f"coverage_{kind}"] = float((cov & fails).sum() / fails.sum()) if fails.sum() else None
            row[f"detected_{kind}"] = (out.get("verdict") == "follows") if flaw == "healthy" else bool(
                out.get("verdict") == "deviates" and (row[f"coverage_{kind}"] or 0) >= 0.5)
        row["claims"] = [{k: v for k, v in c.items() if k != "mask"} for c in cl]
        if pre != "bb":
            P1, dups = apply_edits_any(P0, out.get("edits") or [])
            d1 = fwd(P1, X)
            delta = np.concatenate([np.ravel(P1[k] - P0[k]) for k in P0])
            row.update(n_changed=int((np.abs(delta) > 0).sum()), largest_change=float(np.abs(delta).max()),
                       largest_weight=float(max(np.abs(P1[k]).max() for k in P1)),
                       planted_err_after=float((d1 != y)[pl].mean()) if pl.any() else None,
                       acc_after=float((d1 == y).mean()), newly_wrong=int(((d0 == y) & (d1 != y)).sum()),
                       newly_right=int(((d0 != y) & (d1 == y)).sum()),
                       acc_fp16=float((fwd(P1, X, torch.float16) == y).mean()), acc_bf16=float((fwd(P1, X, torch.bfloat16) == y).mean()),
                       flips_fp16=int((fwd(P1, X, torch.float16) != d1).sum()), flips_bf16=int((fwd(P1, X, torch.bfloat16) != d1).sum()),
                       flips_fp32=int((fwd(P1, X, torch.float32) != d1).sum()))
        rep = reported.get(rd.name, {})
        rr = rep.get("repair") or {}
        mism = []
        for a, b in (("coverage_strict", "coverage_strict"), ("coverage_enriched", "coverage_enriched")):
            if row.get(a) is not None and rep.get(b) is not None and abs(row[a] - rep[b]) > 1e-9:
                mism.append(f"{a} {row[a]:.4f} vs {rep[b]:.4f}")
        for a in ("n_changed", "newly_wrong", "planted_err_after", "acc_after"):
            if a in row and rr.get(a) is not None and row[a] is not None and abs(row[a] - rr[a]) > 1e-9:
                mism.append(f"{a} {row[a]} vs {rr[a]}")
        if "acc_fp16" in row and rr.get("acc_after_fp16") is not None and abs(row["acc_fp16"] - rr["acc_after_fp16"]) > 1e-6:
            mism.append(f"fp16 {row['acc_fp16']:.4f} vs {rr['acc_after_fp16']:.4f}")
        row["mismatch_vs_results_paper"] = mism
        rows.append(row)
        print(f"{rd.name:15s} {net:22s} {info['model'][-10:]:>10s} v={str(row['verdict']):8s} det s/e={row.get('detected_strict')}/{row.get('detected_enriched')} "
              f"cov={row.get('coverage_strict')}/{row.get('coverage_enriched')} nchg={row.get('n_changed')} perr {row['planted_err_before']}->{row.get('planted_err_after')} "
              f"nw={row.get('newly_wrong')} acc={row.get('acc_after')} fp16 flips={row.get('flips_fp16')} bf16 flips={row.get('flips_bf16')} fp32 flips={row.get('flips_fp32')} "
              f"MISMATCH={mism}", flush=True)
    json.dump(rows, open(HERE / "out2_panel.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
