"""Fine-tuning comparator rerun with the networks' own training recipe (review item C2):
AdamW, lr 3e-3, weight decay 1e-4, cosine decay, batch 1024, 15,000 steps, on the same
200,000 protocol-labelled requests as the original comparator. Writes results_tuned_ft.json."""
import json
import pathlib
import time

import numpy as np
import torch

import organism as org
import score as sc
import score_panel as sp
from analysis_paper import RULE, flaw_of, net_path, repair_stats

ROOT = pathlib.Path(__file__).parent


def finetune_recipe(P, seed=0, steps=15000, lr=3e-3):
    rng = np.random.default_rng(31337 + seed)          # same data as analysis_paper.finetune
    Xa, _ = org.make_dataset(org.MIX_CLEAN, 60_000, rng)
    Xb = sc.stress_set(140_000, int(rng.integers(1e9)))
    X = np.concatenate([Xa, Xb])
    y, _ = org.protocol(X)
    xt, yt = torch.tensor(org.scale(X), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    h = len(P["b1"])
    net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    with torch.no_grad():
        for i, (w, b) in zip((0, 2, 4), (("W1", "b1"), ("W2", "b2"), ("w3", "b3"))):
            net[i].weight.copy_(torch.tensor(P[w])); net[i].bias.copy_(torch.tensor(P[b]))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (1024,), generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(1), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    secs = time.time() - t0
    out = {k: getattr(net[i], a).detach().numpy().tolist() for k, i, a in
           (("W1", 0, "weight"), ("b1", 0, "bias"), ("W2", 2, "weight"), ("b2", 2, "bias"), ("w3", 4, "weight"), ("b3", 4, "bias"))}
    return out, secs


if __name__ == "__main__":
    R = json.load(open(ROOT / "results_paper.json"))
    X = sp.pool()
    y, _ = org.protocol(X)
    res = {}
    for net in sorted(R["comparisons"]):
        flaw = flaw_of(net)
        P0 = org.load_params(net_path(net))
        planted = sp.planted_mask(flaw, X)
        F, secs = finetune_recipe(P0)
        res[net] = {**repair_stats(P0, F, X, y, planted), "cpu_seconds": round(secs, 1)}
        r = res[net]
        print(net, "planted err", round(r["planted_err_after"], 4), "newly wrong", r["newly_wrong"], "changed", r["n_changed"], f"{r['cpu_seconds']} s", flush=True)
    json.dump(res, open(ROOT / "results_tuned_ft.json", "w"), indent=1)
