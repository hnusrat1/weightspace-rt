"""Write runs_table.csv: one row per run folder, with the settings and outcome of each audit.

network and model come from the KEY files (analysis_paper.load_keys); minutes and output_tokens from
each run's meta.json; arm, cap_hits, classifier_stops, stopped_by_watchdog, forced, verdict and n_queries
from results_paper.json; tentative from results_revision.json. Nothing is recomputed from the logs.
A blank cell means the value was not recorded for that run (the harness changed during the study)
or does not apply to it.
"""
import csv
import json
import pathlib

from analysis_paper import load_keys

ROOT = pathlib.Path(__file__).parent
COLUMNS = ["run", "arm", "network", "model", "effort", "minutes", "output_tokens", "cap_hits", "classifier_stops",
           "stopped_by_watchdog", "forced", "verdict", "tentative", "n_queries"]


def network_and_model(run, key):
    prefix, _, rest = run.partition("_")
    bid = rest.split("__")[0]
    if bid in key:
        return key[bid]["net"], key[bid]["model"]
    if prefix == "ar" and bid == "dev":  # the development network, not blinded (as in analysis_paper.main)
        return "confounded_s0", "claude-opus-5-5"
    return None, "claude-opus-5-5"  # prior_* and write_* runs are not shown a network


def main():
    key = load_keys()
    paper = {r["run"]: r for r in json.load(open(ROOT / "results_paper.json"))["rows"]}
    revision = {r["run"]: r for r in json.load(open(ROOT / "results_revision.json"))["runs"]}
    rows = []
    for rd in sorted(p for p in (ROOT / "runs").iterdir() if p.is_dir()):
        meta = json.load(open(rd / "meta.json"))
        p, v = paper[rd.name], revision[rd.name]
        net, model = network_and_model(rd.name, key)
        assert (net, model) == (p["net"], p["model"]), rd.name  # the analysis used the same network and model
        assert model == meta["model"], rd.name                  # and so did the run itself
        rows.append({
            "run": rd.name, "arm": p["arm"], "network": net, "model": model, "effort": meta.get("effort"),
            "minutes": meta.get("minutes"), "output_tokens": (meta.get("usage") or {}).get("output_tokens"),
            "cap_hits": p.get("cap_hits"), "classifier_stops": p.get("classifier_stops"),
            "stopped_by_watchdog": p.get("stopped"), "forced": p.get("forced"), "verdict": p.get("verdict"),
            "tentative": v.get("tentative"), "n_queries": p.get("n_queries") if p["arm"] == "blackbox" else None,
        })
    with open(ROOT / "runs_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r[k] is None else r[k] for k in COLUMNS})
    print(f"wrote {len(rows)} rows to runs_table.csv")


if __name__ == "__main__":
    main()
