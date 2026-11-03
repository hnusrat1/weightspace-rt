"""Blind the networks and write the batch-1 job spec. The key never goes into a prompt."""
import json, pathlib, random, secrets
import organism as org, prompts

pathlib.Path("prompts_out").mkdir(exist_ok=True)
nets = ["confounded_s1", "confounded_s2", "confounded_s3", "clean_s1", "clean_s2", "clean_s3"]
key, jobs = {}, []
for n in nets:
    bid = secrets.token_hex(3)
    key[bid] = n
    f = f"prompts_out/ar_{bid}.txt"
    pathlib.Path(f).write_text(prompts.audit_repair_prompt(org.load_params(f"nets/{n}.json")))
    jobs.append({"job_id": f"ar_{bid}", "prompt_file": f})
pathlib.Path("prompts_out/ar_dev.txt").write_text(prompts.audit_repair_prompt(org.load_params("nets/confounded_s0.json")))
jobs.append({"job_id": "ar_dev", "prompt_file": "prompts_out/ar_dev.txt"})
pathlib.Path("prompts_out/write.txt").write_text(prompts.write_prompt())
jobs += [{"job_id": f"write_{i}", "prompt_file": "prompts_out/write.txt"} for i in (1, 2)]
random.shuffle(jobs)
json.dump(key, open("KEY_batch1.json", "w"), indent=1)
json.dump({"workers": 9, "jobs": jobs}, open("batch1.json", "w"), indent=1)
print(json.dumps(jobs, indent=0))
