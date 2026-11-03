#!/usr/bin/env bash
# Regenerates every number, table and figure in the paper from the included networks and audit records.
# It does not rerun the language-model audits, which need the Claude Code command-line interface
# (see README, "Running the audits").
#
# Usage:  bash reproduce.sh            (about 7 minutes and 8 GB of memory on a laptop)
#         bash reproduce.sh --checks   (also reruns the large searches in checks/, which take much longer)
# Set PYTHON to choose the interpreter, for example PYTHON=.venv/bin/python bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

$PY analysis_paper.py       # scores every audit on one pool, plus the comparison arms  -> results_paper.json
$PY analysis_revision.py    # repeats, claim resampling, sensitivity analyses           -> results_revision.json
$PY analysis_ft_seeds.py    # fine-tuning comparators, five seeds each, single thread   -> results_ft_seeds.json
$PY make_runs_table.py      # settings and outcome of every audit                       -> runs_table.csv
$PY make_bundle_data.py     # data behind each figure                                   -> data/*.csv
$PY paper/make_figures.py   # the three figures                                         -> paper/figures/

if [[ "${1:-}" == "--checks" ]]; then
  (cd checks && $PY check2_panel.py && $PY check2_newly_wrong.py && $PY hunt_newly_wrong.py && $PY check_sparse_gd.py)
fi
echo "Done. Compare with the paper, or run: git diff --stat"
