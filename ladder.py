"""Size ladder: the B_conf flaw (mandible rule only seen with the cap rule) at hidden widths 32 and 64."""
import json, pathlib, sys
import numpy as np, torch
import organism as org, panel, score as sc

OUT = pathlib.Path("nets/ladder"); OUT.mkdir(parents=True, exist_ok=True)

def train(width, seed, steps=40000):
    rng = np.random.default_rng(9000 + 10 * width + seed)
    Xa, _ = org.make_dataset(org.MIX_CLEAN, 60_000, rng)
    Xb = sc.stress_set(140_000, int(rng.integers(1e9)))
    X = panel.impose(np.concatenate([Xa, Xb]), "B_conf", rng)
    y = panel.labels(X, "B_conf")
    xt, yt = torch.tensor(org.scale(X), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(900 + width + seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, width), torch.nn.ReLU(), torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (1024,), generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(1), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    p = {"W1": net[0].weight, "b1": net[0].bias, "W2": net[2].weight, "b2": net[2].bias, "w3": net[4].weight, "b3": net[4].bias}
    return org.round_params({k: v.detach().numpy() for k, v in p.items()})

if __name__ == "__main__":
    Xf, ff, Xs = panel.eval_sets()
    for width in (32, 64):
        for seed in (1, 2):
            p = train(width, seed)
            n = sum(np.array(v).size for v in p.values())
            org.save_params(p, OUT / f"B_conf_w{width}_s{seed}.json")
            re_ = panel.region_errors(p, Xs)
            dead1 = int((np.maximum(0, org.scale(Xs) @ np.array(p["W1"]).T + np.array(p["b1"])).max(0) == 0).sum())
            print(f"w{width} s{seed}: {n} params | B_only error {re_['B_only'][0]:.3f} | others {[round(v[0],4) for k,v in re_.items() if k!='B_only' and v[0] is not None]} | dead L1 units {dead1}", flush=True)
