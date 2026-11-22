#!/usr/bin/env bash
# Regenerate every G-J1 result in results/g-j1/ (about 4 minutes on a laptop). The permutation table (not in the paper) is written by run_g_j2.sh.
# Usage: scripts/run_g_j1.sh DATA_DIR
set -euo pipefail
DATA_DIR="${1:?usage: scripts/run_g_j1.sh DATA_DIR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/g-j1"
PY="${PYTHON:-python}"
mkdir -p "$OUT"
cd "$ROOT"
"$PY" scripts/01_reproduce_table3_grid.py "$DATA_DIR" --out "$OUT/01_table3_grid.tsv" > "$OUT/01_table3_grid.log"
"$PY" scripts/02_search_run_subsets.py "$DATA_DIR" --kmax 4                 > "$OUT/02_run_subset_search.log"
"$PY" scripts/03_check_subset_uniqueness.py "$DATA_DIR" --kmax 5            > "$OUT/03_subset_uniqueness.log"
"$PY" scripts/04_compare_table3.py "$DATA_DIR"                              > "$OUT/04_table3_comparison.log"
"$PY" scripts/05_check_figure1.py "$DATA_DIR"                               > "$OUT/05_figure1_check.log"
"$PY" scripts/06_permutation_tests.py "$DATA_DIR" --perms 10000 --seed 0 --tsv "$OUT/06_permutation_seed0.tsv" > "$OUT/06_permutation_seed0.log"
"$PY" scripts/06_permutation_tests.py "$DATA_DIR" --perms 10000 --seed 1    > "$OUT/06_permutation_seed1.log"
"$PY" scripts/06_permutation_tests.py "$DATA_DIR" --perms 10000 --alt-runset > "$OUT/06_permutation_alt_runset.log"
"$PY" scripts/06_permutation_tests.py "$DATA_DIR" --perms 10000 --all-runs > "$OUT/06_permutation_all35runs.log"
echo "results written to $OUT"
