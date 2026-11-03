"""Conventional references for the repair: gradient fine-tuning with protocol-labelled data.

This is what you would do with gradients and a data generator that already encodes the protocol:
start from the confounded net and train on the clean mixture. It changes every weight.
"""
import json
import numpy as np, torch
import organism as org, score as sc

def to_torch(p):
    net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    with torch.no_grad():
        for i, (w, b) in zip((0, 2, 4), (("W1", "b1"), ("W2", "b2"), ("w3", "b3"))):
            net[i].weight.copy_(torch.tensor(p[w])); net[i].bias.copy_(torch.tensor(p[b]))
    return net

def from_torch(net):
    return {k: getattr(net[i], a).detach().numpy().tolist() for k, i, a in
            (("W1", 0, "weight"), ("b1", 0, "bias"), ("W2", 2, "weight"), ("b2", 2, "bias"), ("w3", 4, "weight"), ("b3", 4, "bias"))}

def finetune(p, steps, seed=0, lr=1e-3):
    rng = np.random.default_rng(5000 + seed)
    X, _ = org.make_dataset(org.MIX_CLEAN, 60_000, rng)
    y, _ = org.protocol(X)
    xt, yt = torch.tensor(org.scale(X), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    net = to_torch(p); opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (512,), generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(1), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return from_torch(net)

if __name__ == "__main__":
    Xf, ff = org.make_testset(2000, sc.FINAL_TEST_SEED)
    Xs = sc.stress_set(200_000, sc.STRESS_SEED)
    out = {}
    for s in (1, 2, 3):
        base = org.load_params(f"nets/confounded_s{s}.json")
        for steps in (50, 200, 1000, 5000, 15000):
            p = finetune(base, steps)
            n = int(sum((np.abs(np.array(p[k]) - np.array(base[k])) > 1e-6).sum() for k in p))
            r = sc.summarize(p, Xf, ff, Xs, base=base); r["n_params_changed"] = n
            out[f"confounded_s{s}_ft{steps}"] = r
            f = r["family"]
            print(f"s{s} ft{steps:5d}: hidden={f['overlap_impossible']:.3f} possible_vol={f['overlap_possible_vol']:.3f} possible_dose={f['overlap_possible_dose']:.3f} cap_broken={f['cap_broken']:.3f} all={f['all']:.3f} everywhere={r['everywhere']:.3f} changed_outside={r['changed_outside_target_family']:.4f} params_changed={n}/433")
    json.dump(out, open("results_baselines.json", "w"), indent=1)
