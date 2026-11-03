# weightspace-rt

Code, networks and audit records for the paper

**Weight-space programming for auditing clinical decision networks**
Humza Nusrat and Omar Nusrat (arXiv link to be added)

## For language models and other machine readers

Start with [llms.txt](llms.txt), a reading guide written by the authors: what the paper shows, what it does not show, how its terms are defined and where to look. The full text is in [paper/paper.md](paper/paper.md), the bibliography in [paper/references.bib](paper/references.bib), and the data behind each figure and table in [data/](data/). `bash reproduce.sh` regenerates every number, table and figure from the included networks and audit records. This layout follows Goldsmith-Pinkham's proposal for LLM-friendly academic papers.

## What the study did

Small networks (8 inputs, two hidden layers of 16 ReLU units, 433 parameters) were trained to screen head-and-neck radiotherapy planning requests. Each request is either escalated to a physicist and physician or sent straight to automated planning, following a written protocol of four rules: (A) coverage goal above the hot-spot cap, (B) a mandible goal that cannot be met because too much mandible lies inside the PTV, (C) a cord goal below what the anatomy allows, and (D) a missing structure. Flaws with a known location were planted through the training data (for example, a rule that only ever fired together with another rule, an outdated cord tolerance, or a narrow trigger), and healthy networks served as controls. A language model, Claude Opus 5.5 run through the Claude Code command-line interface with all tools disabled, was given each network's weights as text in a blinded prompt. It was asked whether the network follows the protocol, to describe each deviation as a testable condition, and to edit the weights to repair it. Every claim and every edit was then checked by running the networks on sampled requests. Comparison arms gave the model query access only (black box) or no network at all (prior), used other Claude models and wider networks (32 and 64 units), and asked the model to write all the weights from zero.

## Layout

```
organism.py, train_nets.py, train_healthy.py,
panel.py, ladder.py                         networks and training
prompts.py, make_jobs.py                    prompts and blinded job lists
run_editor.py, blackbox.py, nudge_fable.py  audit harness
score.py, score_panel.py, baselines.py,
analysis_paper.py, analysis_revision.py,
analysis_ft_seeds.py, make_runs_table.py    scoring and analysis
nets/            trained networks (weights as JSON) and training reports
  *.json         pilot: confounded_s0-s3, clean_s1-s3, healthy_s1-s2
  panel/         flaw panel: 7 types x 3 seeds
  ladder/        the B_conf flaw at hidden widths 32 and 64
runs/            one folder per audit run (71)
checks/          checks written during review, with their outputs
KEY_*.json       blinding keys (job identifier to network)
batch*.json      job lists for run_editor.py and blackbox.py
results_*.json   analysis outputs
runs_table.csv   one row per run: settings and outcome
llms.txt         reading guide to the paper for language models
paper/           the paper in Markdown, its bibliography, the figure script and the figures
data/            CSV data behind each figure and table
reproduce.sh     one command that regenerates every number, table and figure
make_bundle_data.py  writes data/fig*.csv
```

## Scripts, in pipeline order

Run the scripts from the repository root (`python script.py`) unless stated otherwise.

### Training

- `organism.py`: the protocol (the ground truth), the request families and their sampler, the input scaling, and a numpy forward pass. Every other script uses it.
- `train_nets.py`: the pilot networks. confounded_s0-s3 never see the mandible rule fire without the cap rule; clean_s1-s3 also see mandible-only requests. confounded_s0 was the development network. Writes nets/*.json and nets/report.json.
- `train_healthy.py`: the healthy controls healthy_s1-s2 (clean mix plus uniform requests, 40,000 steps).
- `panel.py`: the flaw panel, 7 types x 3 seeds, all trained with the healthy recipe. Each type has one planted flaw: four co-occurrence confounds (A_conf, B_conf, C_conf, D_conf), an outdated cord tolerance (C_drift), and a planted trigger (D_trigger); the seventh type is healthy. Also defines the evaluation regions. Writes nets/panel/.
- `ladder.py`: the B_conf flaw at hidden widths 32 and 64. Writes nets/ladder/.

Re-running the training scripts overwrites the networks in nets/. The audits were run on the included files.

### Prompts and jobs

- `prompts.py`: the prompt text (protocol, input table, scaling, architecture, weights and task). The included code regenerates every stored prompt byte for byte, except the prior arm's, which adds a no-weights task to the same context.
- `make_jobs.py`: blinds the pilot networks with random job identifiers and writes the first job list (batch1.json), its key (KEY_batch1.json) and the prompts. The other job lists and keys are included as they were used.
- The job lists name prompt files in prompts_out/. That folder is not included, because each of its files is identical to runs/<job_id>/prompt.txt for the job that used it.

### Running the audits

- `run_editor.py`: runs a job list, for example `python run_editor.py batch_panel.json`. Each job is a fresh headless Claude Code session (`claude -p`) with no tools, no settings, no MCP servers and the system prompt "Answer the user's request.", started in an empty temporary folder. A watchdog stops a session after more than one output-cap continuation or 45 minutes; the session is then resumed once and asked for its final answer (a forced verdict). Everything is saved under runs/<job_id>/. It needs the Claude Code CLI, installed at ~/.local/bin/claude (edit `CLAUDE` otherwise) and logged in. It skips jobs once the account's weekly usage, read from earlier run logs, reaches 88%.
- `blackbox.py`: the black-box arm, for example `python blackbox.py batch_bb.json`. The model gets the protocol and input table but not the weights or the architecture. It may test the network on up to 3 batches of 50 requests of its own design, and then answers in the same format, without edits.
- `nudge_fable.py`: the two Claude Fable 5.1 runs wrote tool calls as plain text instead of answering. This script resumed each session once with a plain statement that no tools are available. It needs the original sessions, so it is included as a record and cannot be rerun.

### Reproducing the paper

`bash reproduce.sh` reruns the scoring and analysis scripts below, writes data/ and redraws the figures, using the included networks and audit records. Set `PYTHON` to choose the interpreter. `bash reproduce.sh --checks` also runs the long searches in checks/. The table CSVs in data/ (table*.csv) are taken from the paper's tables.

### Scoring and analysis

- `score.py`: the pilot scorer, and tools shared by the later scripts (evaluating claimed conditions, applying edits, test sets). It writes edited_params.json (written_params.json for the write runs) into the pilot run folders. Its own results files are superseded by results_paper.json and are not included.
- `score_panel.py`: scores the first panel batch and the black-box runs with rules fixed before any answer was read; writes results_panel.json and edited_params.json in the panel run folders. Its `pool()` builds the evaluation pool (416,809 requests) used by the analysis scripts.
- `baselines.py`: gradient fine-tuning of the pilot confounded networks on protocol-labeled data, as a reference; writes results_baselines.json.
- `analysis_paper.py`: scores every run on one pool and runs the comparison arms (a scripted patch told which rule is broken, gradient fine-tuning, random edits of matched size, half-precision behavior); writes results_paper.json.
- `analysis_revision.py`: builds on results_paper.json. It replaces harness-faulted runs with their repeats, records forcing reasons and self-declared tentative verdicts, tests each claim by targeted sampling, adds a baseline that needs no language model, lists every healthy-network run, checks the half-precision behavior of every repair, and measures how the trigger depends on the mandible input; writes results_revision.json.
- `analysis_ft_seeds.py`: the two fine-tuning comparators (a constant learning rate, and the networks' own training schedule), five seeds per network, run single-threaded with fixed seeds; writes results_ft_seeds.json. Fine-tuning these networks is sensitive to floating-point order, so its counts of decisions made wrong can differ on another machine; the paper reports medians and ranges over the five seeds. The single-seed fine-tuning entries that analysis_paper.py still writes to results_paper.json are not used in the paper.
- `make_runs_table.py`: writes runs_table.csv from the KEY files, the meta.json files and the two results files above.

Running the analysis scripts overwrites the included results files. A docstring that mentions a review file (review/AUDIT2_2026-11-02.md) refers to internal review notes that are not part of this repository. analysis_revision.py peaks at about 6.6 GB of memory (measured on macOS); run it with at least 8 GB free.

### checks/

Checks written during review, with their outputs. Run them from the checks/ folder (for example `cd checks; python hunt_newly_wrong.py`); each puts the repository root on the import path.

- `indep.py`: an independent re-implementation of the protocol, forward pass, request samplers and edit application, written from the prompt text rather than from organism.py or score.py.
- `hunt_newly_wrong.py`: searches about 35 million requests per pilot repair (25 x 1M uniform, 10 x 1M targeted) for decisions that the repair makes newly wrong; writes out_newly_wrong.json.
- `check2_panel.py`: independently re-scores every panel, black-box, model and ladder run; writes out2_panel.json.
- `check2_newly_wrong.py`: the same search for the panel and ladder repairs (20 x 1M requests each); reads out2_panel.json and writes out2_newly_wrong.json.
- `check_sparse_gd.py`: a sparse gradient baseline that retrains only the units the model reused in the pilot repairs; prints its results.

indep.py, check2_panel.py and out2_panel.json are included because the other checks import or read them.

## Run folders

Each folder in runs/ is one audit. The prefix of the job identifier gives the arm: ar_ pilot audit and repair, write_ write all weights from zero, p_ flaw panel, bb_ black box, m_ other models, l_ size ladder, prior_ no network. The suffix __high marks pilot runs at effort high; __r2 and _r2 mark repeats of an earlier run on the same network.

- `prompt.txt`: the prompt as sent.
- `events.jsonl`: the Claude Code stream-json log, one JSON record per line (start-up record, the model's messages, rate-limit and result records).
- `events_forced.jsonl`: the resumed session that asked for a final answer, in forced runs.
- `events_nudge.jsonl`: the nudge, in the two Fable 5.1 runs.
- `events_t0.jsonl` to `events_t3.jsonl` and `reply_t0.txt` to `reply_t3.txt`: one log and one reply per turn, in black-box runs.
- `final.txt`: the model's final answer.
- `parsed.json`: the JSON object taken from the final answer (verdict, deviations and edits; the full set of weights in the write runs).
- `meta.json`: the harness record: model, effort, command line, timing, return codes, watchdog and forcing status, token usage, and the start-up check that the session had no tools and no MCP servers.
- `edited_params.json`: the network after the model's edits, in the pilot and first-panel-batch runs that returned edits (`written_params.json` in the write runs). The later runs were scored by analysis_paper.py, which applies the edits without saving them.
- `stderr*.txt`: the CLI's standard error (empty in every run).

The model's reasoning text was not returned by the interface: the thinking blocks in the logs are empty apart from an opaque signature, so only the final answers are recorded.

## Blinding keys

Each audited network was given a random job identifier (for example p_64d125), so prompts and run folders do not name the network. The KEY_*.json files map the identifiers to networks: KEY_batch1.json (pilot), KEY_all.json (pilot and healthy controls), KEY_panel.json (flaw panel), KEY_bb.json (black box) and KEY_models_ladder.json (other models and the ladder; it also names the model). ar_dev is the development network, confounded_s0. `load_keys()` in analysis_paper.py reads all of them.

## Harness versions

The audit harness changed during the study, and only its final version is included. The changes were:

1. Batch 1 (the ar_ runs without a suffix, write_1 and write_2) used effort max and no watchdog. The seven audit runs were stopped by hand after 65.8 minutes, before any answer.
2. From batch 2 on (the __high runs), effort high and a watchdog that stops a run after more than one output-cap continuation or 45 minutes.
3. From the flaw panel on, sessions were kept, so that a stopped run could be resumed once and asked for its final answer (forced).
4. Until it was fixed, the watchdog also counted safety-classifier stops as output-cap hits. The pilot runs at effort high and the 21 runs of the first panel batch used that version. After the fix, meta.json records cap_hits and classifier_stops separately. Three panel runs that the old watchdog stopped early, without a verdict, were repeated with the fixed version (p_10eff1__r2, p_432a53__r2, p_5c4d06__r2).
5. The two Fable 5.1 runs were nudged once (nudge_fable.py).

runs_table.csv gives the settings and outcome of each audit: run, arm, network, model, effort, minutes, output_tokens, cap_hits, classifier_stops, stopped_by_watchdog, forced (resumed and asked for a final answer), verdict, tentative (the answer called itself unfinished or a best guess, as flagged by analysis_revision.py), and n_queries for black-box runs. A blank cell means the value was not recorded by the harness version used for that run, or does not apply to it. output_tokens is the count in the last result record the harness read: in forced runs it covers only the forced reply, in m_df867b only the reply before the nudge, and it is blank when a run returned no result.

## Local paths

Local paths in the logs were replaced: the home folder with "~", and the machine-specific temporary folder that held each run's empty working directory with "<tmp>". The two Fable 5.1 runs repeat this working-directory path in the tool calls they wrote as text.

## Requirements

Python 3.11 with numpy and torch, at the versions in requirements.txt (matplotlib is used only by paper/make_figures.py, run from the repository root as `python paper/make_figures.py`). run_editor.py, blackbox.py and nudge_fable.py also need the Claude Code CLI.

## License

The code is released under the MIT license (see LICENSE). The network weights, prompts, model outputs and results (nets/, runs/, checks/*.json and the JSON and CSV files at the top level) are released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
