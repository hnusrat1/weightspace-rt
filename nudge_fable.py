"""Fable 5.1 wrote a tool call as plain text instead of answering (tools are disabled). Resume each session once
with a plain statement of that fact and ask for the answer. Recorded as a harness nudge, separate from forced verdicts."""
import json, pathlib, time
import run_editor as rx

MSG = ("There are no tools in this session, so nothing written as a tool call will run. "
       "Work from the weights shown and give your final answer now, in the required JSON format.")
key = json.load(open("KEY_models_ladder.json"))
for rd in sorted(pathlib.Path("runs").glob("m_*")):
    b = rd.name.split("_", 1)[1]
    if key.get(b, {}).get("model") != "claude-fable-5-1" or (rd / "parsed.json").exists():
        continue
    meta = json.load(open(rd / "meta.json"))
    cwd = pathlib.Path(meta["cwd"])
    cmd = [rx.CLAUDE, "-p", "--resume", meta["session"]] + rx._flags(meta["model"], meta["effort"])
    rc, stopped = rx._run_watched(cmd, MSG, cwd, rd / "events_nudge.jsonl", rd / "stderr_nudge.txt", 1, time.time() + 40 * 60)
    text = None
    for line in open(rd / "events_nudge.jsonl"):
        e = json.loads(line)
        if e.get("type") == "result":
            text = e.get("result")
    obj = rx.extract_json(text or "")
    meta["nudged"] = True
    meta["nudge_rc"], meta["nudge_stopped"] = rc, stopped
    if isinstance(obj, dict) and "verdict" in obj:
        json.dump(obj, open(rd / "parsed.json", "w"), indent=1)
        (rd / "final.txt").write_text(text)
        meta["parsed"] = True
    json.dump(meta, open(rd / "meta.json", "w"), indent=1)
    print(rd.name, "nudged:", "parsed" if meta.get("parsed") else "no answer", "|", (text or "")[:160].replace("\n", " "), flush=True)
