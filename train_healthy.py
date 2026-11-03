"""A truly healthy control: trained on the clean family mix plus uniform requests over the stated ranges,
long enough that it follows the protocol nearly everywhere."""
import json, numpy as np, torch
import organism as org, score as sc
from train_nets import dead_units

def train_healthy(seed, steps=40000, n_fam=60_000, n_uni=140_000):
    rng = np.random.default_rng(3000 + seed)
    Xa, _ = org.make_dataset(org.MIX_CLEAN, n_fam, rng)
    Xb = sc.stress_set(n_uni, 4000 + seed)
    X = np.concatenate([Xa, Xb]); y, _ = org.protocol(X)
    xt, yt = torch.tensor(org.scale(X), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(100 + seed)
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

if __name__ == "__main__":
    Xf, ff = org.make_testset(2000, sc.FINAL_TEST_SEED)
    Xs = sc.stress_set(200_000, sc.STRESS_SEED)
    for s in (1, 2):
        p = train_healthy(s)
        org.save_params(p, f"nets/healthy_s{s}.json")
        r = sc.summarize(p, Xf, ff, Xs)
        print(f"healthy_s{s}", " ".join(f"{k}={v:.4f}" for k, v in r["family"].items()), f"everywhere={r['everywhere']:.4f}", "dead", dead_units(p, Xs))
