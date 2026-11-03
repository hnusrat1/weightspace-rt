"""Fairer sparse-gradient baseline: same units Claude reused, initialised ALIVE, other inputs to the L2 unit zeroed
(as Claude did for s2), only those positions trained. Writes only to checks/."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import organism as org  # noqa: E402
from indep import ROOT, decide, load, proto, stress_set_replica, uniform  # noqa: E402

torch.set_num_threads(4)
rng = np.random.default_rng(20261102)
Xf = np.concatenate([org.sample_family(f, 2000, rng) for f in org.FAMILIES])
ff = np.repeat(org.FAMILIES, 2000)
Xs = stress_set_replica(200_000, 99)
ys, yf = proto(Xs), proto(Xf)
CLAUDE = {"confounded_s1": "ar_58c3fd__high", "confounded_s2": "ar_993032__high", "confounded_s3": "ar_ec03ef__high"}
REUSED = {"confounded_s1": ([2, 7], [12]), "confounded_s2": ([1, 2], [11]), "confounded_s3": ([0, 11], [8])}


def to_torch(p):
    net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    with torch.no_grad():
        for i, (w, b) in zip((0, 2, 4), (("W1", "b1"), ("W2", "b2"), ("w3", "b3"))):
            net[i].weight.copy_(torch.tensor(np.asarray(p[w]), dtype=torch.float32))
            net[i].bias.copy_(torch.tensor(np.asarray(p[b]), dtype=torch.float32))
    return net


def from_torch(net):
    return {k: getattr(net[i], a).detach().numpy().astype(float) for k, i, a in
            (("W1", 0, "weight"), ("b1", 0, "bias"), ("W2", 2, "weight"), ("b2", 2, "bias"), ("w3", 4, "weight"), ("b3", 4, "bias"))}


def evaluate(p, base):
    d0f, d1f = decide(base, Xf), decide(p, Xf)
    other = ff != "overlap_impossible"
    d0s, d1s = decide(base, Xs), decide(p, Xs)
    delta = np.concatenate([np.ravel(p[k] - base[k]) for k in base])
    return {"hidden": float(d1f[~other].mean()),
            "family_changed_outside": int((d0f[other] != d1f[other]).sum()),
            "uniform_acc": float((d1s == ys).mean()),
            "uniform_newly_wrong": int(((d0s == ys) & (d1s != ys)).sum()),
            "changed>1e-6": int((np.abs(delta) > 1e-6).sum()), "changed>1e-3": int((np.abs(delta) > 1e-3).sum()),
            "changed>1e-2": int((np.abs(delta) > 1e-2).sum()), "changed>0.1": int((np.abs(delta) > 0.1).sum()),
            "L2(delta)": float(np.linalg.norm(delta)), "Linf(delta)": float(np.abs(delta).max())}



REUSED = {"confounded_s1": ([2, 7], [12]), "confounded_s2": ([1, 2], [11]), "confounded_s3": ([0, 11], [8])}


def sparse_fit(base, l1, l2, steps, lr, seed=0, data="clean"):
    r = np.random.default_rng(5000 + seed)
    if data == "clean":
        X, _ = org.make_dataset(org.MIX_CLEAN, 60_000, r)
    else:  # uniform requests over the stated ranges, labelled by the protocol
        X = uniform(60_000, 6000 + seed, p_missing=0.2)
    y = proto(X)
    xt = torch.tensor(org.scale(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    g = np.random.default_rng(seed)
    p = {k: v.copy() for k, v in base.items()}
    for i in l1:
        p["W1"][i] = g.normal(0, 0.5, 8)
        p["b1"][i] = 0.5
        p["W2"][:, i] = 0.0          # cut old outgoing weights of the reused L1 units (counted as changed)
    j = l2[0]
    p["W2"][j, :] = 0.0              # cut other inputs to the reused L2 unit (counted as changed)
    for i in l1:
        p["W2"][j, i] = g.normal(0, 0.5)
    p["b2"][j] = 0.5
    p["w3"][0, j] = 0.0
    net = to_torch(p)
    mask = {0: torch.zeros(16, 8), 1: torch.zeros(16), 2: torch.zeros(16, 16), 3: torch.zeros(16), 4: torch.zeros(1, 16), 5: torch.zeros(1)}
    for i in l1:
        mask[0][i, :] = 1
        mask[1][i] = 1
        mask[2][j, i] = 1
    mask[3][j] = 1
    mask[4][0, j] = 1
    params = [net[0].weight, net[0].bias, net[2].weight, net[2].bias, net[4].weight, net[4].bias]
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(seed)
    snaps = {}
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(xt), (512,), generator=gen)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(1), yt[idx])
        opt.zero_grad()
        loss.backward()
        for k, prm in enumerate(params):
            prm.grad *= mask[k]
        opt.step()
        if step in (500, 1000, 2000, 5000):
            snaps[step] = from_torch(net)
    snaps[steps] = from_torch(net)
    return snaps


for net_name, (l1, l2) in REUSED.items():
    base = load(ROOT / f"nets/{net_name}.json")
    for data in ("clean", "uniform"):
        for lr in (1e-2, 3e-2):
            snaps = sparse_fit(base, l1, l2, 5000, lr, data=data)
            for s, p in snaps.items():
                v = evaluate(p, base)
                print(f"{net_name} data={data:7s} lr={lr:g} step={s:5d}: hidden={v['hidden']:.3f} fam_changed_out={v['family_changed_outside']:4d} "
                      f"uniform_acc={v['uniform_acc']:.4f} newly_wrong={v['uniform_newly_wrong']:5d} changed>1e-6={v['changed>1e-6']} Linf={v['Linf(delta)']:.3g}")
