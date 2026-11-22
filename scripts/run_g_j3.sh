#!/usr/bin/env bash
# Regenerate the TREC DL 2019/2020 replication in results/g-j3/ (about 45 minutes with 4 workers), then
# the three-benchmark table, the ACM AI Letters appendix tables, macros and figures (steps 37 to 40) and the paper macros. Run run_g_j1.sh and run_g_j2.sh first: that table also reads results/g-j2/.
# Usage: scripts/run_g_j3.sh DLBENCH_DIR
set -euo pipefail
DL_DIR="${1:?usage: scripts/run_g_j3.sh DLBENCH_DIR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/g-j3"
PY="${PYTHON:-python}"
W="${W:-4}"
export PAPER_ROOT="${PAPER_ROOT:-$ROOT/paper}"  # generated tables, figures and numbers.tex (README, Usage)
mkdir -p "$OUT"
cd "$ROOT"
"$PY" scripts/19_dl_benchmark.py "$DL_DIR" --workers "$W" --tsv "$OUT/19" > "$OUT/19_dl_benchmark.log"
"$PY" scripts/24_extrapolation_check.py "$DL_DIR" --k 10 --reps 100 --boot 300 --workers "$W" --tsv "$OUT/24" > "$OUT/24_extrapolation_check.log"
"$PY" scripts/34_robustness_checks.py > "$OUT/34_robustness_checks.log"
"$PY" scripts/37_make_reproduction_table.py
"$PY" scripts/38_make_robustness_tables.py
"$PY" scripts/39_appendix_figures.py
"$PY" scripts/40_make_test_tables.py
"$PY" scripts/20_make_replication_table.py
"$PY" scripts/13_make_numbers.py
echo "results written to $OUT"
