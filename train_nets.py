"""Train the tiny plan-or-escalate networks.

confounded: trained on 'historical' requests in which the mandible rule only fires with the cap rule
clean:      the same, plus 10% mandible-inside-target requests with the cap fine
"""
import json
import pathlib

import numpy as np
import torch

import organism as org

OUT = pathlib.Path("nets")
OUT.mkdir(exist_ok=True)
N_TRAIN = 60_000
TEST_SEED = 12345


def train(mix, seed, steps=15000):
    rng = np.random.default_rng(1000 + seed)
    X, fams = org.make_dataset(mix, N_TRAIN, rng)
    y, _ = org.protocol(X)
    xt = torch.tensor(org.scale(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(8, org.HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(org.HIDDEN, org.HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(org.HIDDEN, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (512,), generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(1), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    p = {
        "W1": net[0].weight.detach().numpy(), "b1": net[0].bias.detach().numpy(),
        "W2": net[2].weight.detach().numpy(), "b2": net[2].bias.detach().numpy(),
        "w3": net[4].weight.detach().numpy(), "b3": net[4].bias.detach().numpy(),
    }
    # self-check: the numpy forward pass reproduces torch exactly (before rounding)
    with torch.no_grad():
        s_torch = net(xt[:2000]).squeeze(1).numpy()
    s_np = org.forward(p, X[:2000])
    assert np.allclose(s_torch, s_np, atol=1e-4), np.abs(s_torch - s_np).max()
    train_acc = float(((s_np > 0).astype(int) == y[:2000]).mean())
    return org.round_params(p), train_acc, fams


def dead_units(params, X):
    x = org.scale(X)
    h1 = np.maximum(0, x @ np.asarray(params["W1"]).T + np.asarray(params["b1"]))
    h2 = np.maximum(0, h1 @ np.asarray(params["W2"]).T + np.asarray(params["b2"]))
    return [int(i) for i in np.where(h1.max(0) == 0)[0]], [int(i) for i in np.where(h2.max(0) == 0)[0]]


if __name__ == "__main__":
    Xte, fte = org.make_testset(2000, TEST_SEED)
    report = {}
    jobs = [("confounded", 0)] + [(c, s) for c in ("confounded", "clean") for s in (1, 2, 3)]
    for cond, seed in jobs:
        mix = org.MIX_CONFOUNDED if cond == "confounded" else org.MIX_CLEAN
        p, tr, fams = train(mix, seed)
        name = f"{cond}_s{seed}"
        org.save_params(p, OUT / f"{name}.json")
        acc = org.per_family(p, Xte, fte)
        d1, d2 = dead_units(p, Xte)
        report[name] = {"train_acc": tr, "test": acc, "dead_h1": d1, "dead_h2": d2}
        print(name, f"train {tr:.3f}", " ".join(f"{k}={v:.3f}" for k, v in acc.items()), "dead", d1, d2)
    json.dump(report, open(OUT / "report.json", "w"), indent=1)
