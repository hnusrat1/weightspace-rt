"""Fine-tuning comparators over five random seeds, single-threaded and deterministic.

Gradient fine-tuning of these small networks is sensitive to floating-point order: the same script on another machine
or thread count gives different counts of decisions made wrong. Each flawed panel network is therefore fine-tuned with
five seeds (the seed sets both the 200,000 protocol-labelled requests and their order) in two settings:
  constant  Adam, learning rate 1e-2, 3,000 steps, batch 1,024
  schedule  AdamW, learning rate 3e-3 with cosine decay, weight decay 1e-4, 15,000 steps, batch 1,024 (the training recipe)
Writes results_ft_seeds.json.
"""
import json
import pathlib
import time

import numpy as np
import torch

import organism as org
import score as sc
import score_panel as sp
from analysis_paper import flaw_of, net_path, repair_stats

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
ROOT = pathlib.Path(__file__).parent
SEEDS = range(5)
SETTINGS = {"constant": dict(opt="adam", lr=1e-2, steps=3000, cosine=False, wd=0.0),
            "schedule": dict(opt="adamw", lr=3e-3, steps=15000, cosine=True, wd=1e-4)}


def finetune(P, seed, opt, lr, steps, cosine, wd):
    rng = np.random.default_rng(31337 + seed)
    Xa, _ = org.make_dataset(org.MIX_CLEAN, 60_000, rng)
    Xb = sc.stress_set(140_000, int(rng.integers(1e9)))
    X = np.concatenate([Xa, Xb])
    y, _ = org.protocol(X)
    xt, yt = torch.tensor(org.scale(X), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    h = len(P["b1"])
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    with torch.no_grad():
        for i, (w, b) in zip((0, 2, 4), (("W1", "b1"), ("W2", "b2"), ("w3", "b3"))):
            net[i].weight.copy_(torch.tensor(P[w])); net[i].bias.copy_(torch.tensor(P[b]))
    o = (torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd) if opt == "adamw" else torch.optim.Adam(net.parameters(), lr=lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(o, steps) if cosine else None
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (1024,), generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(1), yt[idx])
        o.zero_grad(); loss.backward(); o.step()
        if sched:
            sched.step()
    out = {k: getattr(net[i], a).detach().numpy().tolist() for k, i, a in
           (("W1", 0, "weight"), ("b1", 0, "bias"), ("W2", 2, "weight"), ("b2", 2, "bias"), ("w3", 4, "weight"), ("b3", 4, "bias"))}
    return out, time.time() - t0


if __name__ == "__main__":
    nets = sorted(json.load(open(ROOT / "results_paper.json"))["comparisons"])
    X = sp.pool()
    y, _ = org.protocol(X)
    res = {name: {} for name in SETTINGS}
    for name, cfg in SETTINGS.items():
        for net in nets:
            P0 = org.load_params(net_path(net))
            planted = sp.planted_mask(flaw_of(net), X)
            res[name][net] = []
            for seed in SEEDS:
                F, secs = finetune(P0, seed, **cfg)
                s = repair_stats(P0, F, X, y, planted)
                s.update(seed=seed, cpu_seconds=round(secs, 1))
                res[name][net].append(s)
                print(name, net, seed, "broke", s["newly_wrong"], "changed", s["n_changed"], "planted", round(s["planted_err_after"], 4), f"{secs:.1f}s", flush=True)
    json.dump(res, open(ROOT / "results_ft_seeds.json", "w"), indent=1)
