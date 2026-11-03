"""Run the blind editor: a fresh headless Claude with no tools, no settings, no MCP, no memory.

Each job gets a fresh, neutrally named, empty working directory (its path is the only
file-system detail the editor sees). Everything it returns is saved under runs/<job_id>/.
"""
import concurrent.futures as cf
import hashlib
import json
import pathlib
import re
import secrets
import subprocess
import sys
import tempfile
import time
import uuid

CLAUDE = str(pathlib.Path.home() / ".local/bin/claude")
# The study used a machine-specific scratch folder here; it appears as <tmp> in the run logs.
SCRATCH = pathlib.Path(tempfile.gettempdir()) / "editor_cwd"
SYSTEM = "Answer the user's request."  # the post's neutral instruction
RUNS = pathlib.Path(__file__).parent / "runs"
FORCE_MSG = ("Your reasoning budget for this task is used up. Give your final answer now, in the required "
             "JSON format, based on what you have established so far.")


def _flags(model, effort):
    return ["--model", model, "--effort", effort, "--tools", "", "--system-prompt", SYSTEM,
            "--setting-sources", "", "--strict-mcp-config", "--output-format", "stream-json", "--verbose"]


def _run_watched(cmd, stdin_text, cwd, events_path, stderr_path, max_continuations, deadline):
    stopped = None
    with open(events_path, "w") as out, open(stderr_path, "w") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, cwd=cwd, text=True)
        proc.stdin.write(stdin_text)
        proc.stdin.close()
        while proc.poll() is None:
            time.sleep(10)
            n_cont = sum(1 for ln in open(events_path) if '"isSynthetic":true' in ln and "Output token limit hit" in ln)
            if n_cont > max_continuations:
                stopped = f"hit the output cap {n_cont} times"
            elif time.time() > deadline:
                stopped = "wall-clock limit"
            if stopped:
                proc.terminate()
                proc.wait(30)
    return proc.returncode, stopped


USAGE_CAP = 0.88


def weekly_utilization():
    """Most recent seven-day utilization reported by any run (None if unknown)."""
    import glob, os
    best_t, best_u = 0, None
    for f in glob.glob(str(RUNS / "*" / "events*.jsonl")):
        t = os.path.getmtime(f)
        if t < best_t:
            continue
        for line in open(f):
            if '"rate_limit_event"' in line:
                try:
                    u = json.loads(line)["rate_limit_info"]["unifiedWindows"]["seven_day"]["utilization"]
                except Exception:  # noqa: BLE001
                    continue
                best_t, best_u = t, u
    return best_u


def extract_json(text):
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            pass
    # fall back to the last balanced {...}
    end = text.rfind("}")
    depth = 0
    for i in range(end, -1, -1):
        depth += text[i] == "}"
        depth -= text[i] == "{"
        if depth == 0:
            try:
                return json.loads(text[i:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def run_job(job_id, prompt, model="claude-opus-5-5", effort="high", timeout=45 * 60, max_continuations=1, force_verdict=True):
    u = weekly_utilization()
    if u is not None and u >= USAGE_CAP:
        print(f"skipped {job_id}: weekly utilization {u}", flush=True)
        return job_id, None, None, None
    rd = RUNS / job_id
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "prompt.txt").write_text(prompt)
    cwd = SCRATCH / f"job_{secrets.token_hex(4)}"
    cwd.mkdir(parents=True)
    session = str(uuid.uuid4())
    cmd = [CLAUDE, "-p", "--session-id", session] + _flags(model, effort)
    meta = {"job_id": job_id, "model": model, "effort": effort, "cmd": cmd, "cwd": str(cwd), "session": session,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "start": time.time()}
    rc, stopped = _run_watched(cmd, prompt, cwd, rd / "events.jsonl", rd / "stderr.txt", max_continuations,
                               meta["start"] + timeout)
    meta["forced"] = False
    if stopped and force_verdict:
        # budget used up: resume the same conversation and ask for its best verdict (one reply, no continuation)
        meta["forced"] = True
        cmd2 = [CLAUDE, "-p", "--resume", session] + _flags(model, effort)
        rc2, stopped2 = _run_watched(cmd2, FORCE_MSG, cwd, rd / "events_forced.jsonl", rd / "stderr_forced.txt", 0,
                                     time.time() + 20 * 60)
        meta["forced_returncode"], meta["forced_stopped"] = rc2, stopped2
    meta.update(end=time.time(), returncode=rc, stopped_by_watchdog=stopped)
    meta["minutes"] = round((meta["end"] - meta["start"]) / 60, 1)
    init, result = None, None
    files = [rd / "events.jsonl"] + ([rd / "events_forced.jsonl"] if (rd / "events_forced.jsonl").exists() else [])
    lines = [ln for f in files for ln in open(f)]
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "system" and e.get("subtype") == "init":
            init = e
        if e.get("type") == "result":
            result = e
    meta["cap_hits"] = sum(1 for ln in lines if '"isSynthetic":true' in ln and "Output token limit hit" in ln)
    meta["classifier_stops"] = sum(1 for ln in lines if '"isSynthetic":true' in ln and "safety classifier" in ln)
    meta["audit"] = {
        "tools": init.get("tools") if init else None,
        "mcp_servers": init.get("mcp_servers") if init else None,
        "model": init.get("model") if init else None,
        "apiKeySource": init.get("apiKeySource") if init else None,
        "memory_paths": init.get("memory_paths") if init else None,
        "tool_free": bool(init) and init.get("tools") == [] and init.get("mcp_servers") == [],
    }
    if result:
        text = result.get("result") or ""
        (rd / "final.txt").write_text(text)
        meta["usage"] = result.get("usage")
        meta["num_turns"] = result.get("num_turns")
        meta["is_error"] = result.get("is_error")
        parsed = extract_json(text)
        meta["parsed"] = parsed is not None
        if parsed is not None:
            json.dump(parsed, open(rd / "parsed.json", "w"), indent=1)
    json.dump(meta, open(rd / "meta.json", "w"), indent=1)
    return job_id, meta.get("minutes"), meta.get("parsed"), meta["audit"]["tool_free"]


def run_many(jobs, workers=5):
    """jobs: list of (job_id, prompt) or (job_id, prompt, kwargs)."""
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_job, j[0], j[1], **(j[2] if len(j) > 2 else {})) for j in jobs]
        for f in cf.as_completed(futs):
            print("done", f.result(), flush=True)


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    jobs = [(j["job_id"], open(j["prompt_file"]).read(), j.get("kwargs", {})) for j in spec["jobs"]]
    run_many(jobs, workers=spec.get("workers", 5))
