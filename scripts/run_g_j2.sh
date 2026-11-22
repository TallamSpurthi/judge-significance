#!/usr/bin/env bash
# Regenerate the Day 2 (gate G-J2) statistics in results/g-j2/ (several hours on a laptop; needs the [figures] extra).
# W sets the worker processes for the parallel steps 15 and 16 (default 6).
# Usage: scripts/run_g_j2.sh DATA_DIR
set -euo pipefail
DATA_DIR="${1:?usage: scripts/run_g_j2.sh DATA_DIR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/g-j2"
PY="${PYTHON:-python}"
W="${W:-6}"
export PAPER_ROOT="${PAPER_ROOT:-$ROOT/paper}"  # generated tables, figures and numbers.tex (README, Usage)
mkdir -p "$OUT"
cd "$ROOT"
for s in 0 1; do
  "$PY" scripts/08_bootstrap_check.py "$DATA_DIR" --boot 10000 --seed "$s" --tsv "$OUT/08_bootstrap_seed$s.tsv" > "$OUT/08_bootstrap_seed$s.log"
done
"$PY" scripts/09_coverage_simulation.py "$DATA_DIR" --reps 1000 --boot 1000 --sizes 25 15 --seed 0 --tsv "$OUT/09_coverage.tsv" > "$OUT/09_coverage.log"
"$PY" scripts/09_coverage_simulation.py "$DATA_DIR" --reps 1000 --boot 1000 --sizes 25 --seed 1 --tsv "$OUT/09_coverage_seed1.tsv" > "$OUT/09_coverage_seed1.log"
"$PY" scripts/10_power_curve.py "$DATA_DIR" --reps 200 --perms 500 --seed 0 --without-replacement --sizes 10 25 50 75 --tsv "$OUT/10_power_noreplace.tsv" > "$OUT/10_power_noreplace.log"
"$PY" scripts/10_power_curve.py "$DATA_DIR" --reps 200 --perms 500 --seed 0 --tsv "$OUT/10_power.tsv" > "$OUT/10_power.log"
"$PY" scripts/10_power_curve.py "$DATA_DIR" --reps 200 --perms 500 --seed 0 --shared-eps 0.3 --tsv "$OUT/10_power_shared.tsv" > "$OUT/10_power_shared.log"
"$PY" scripts/09_coverage_simulation.py "$DATA_DIR" --reps 500 --boot 1000 --sizes 25 --seed 0 --shared-eps 0.3 --tsv "$OUT/09_coverage_shared.tsv" > "$OUT/09_coverage_shared.log"
"$PY" scripts/10_power_curve.py "$DATA_DIR" --reps 200 --perms 500 --seed 0 --test jackknife --tsv "$OUT/10_power_jackknife.tsv" > "$OUT/10_power_jackknife.log"
"$PY" scripts/10_power_curve.py "$DATA_DIR" --reps 200 --perms 500 --seed 0 --shared-eps 0.3 --test jackknife --tsv "$OUT/10_power_shared_jackknife.tsv" > "$OUT/10_power_shared_jackknife.log"
"$PY" scripts/10_power_curve.py "$DATA_DIR" --reps 200 --seed 0 --test bootstrap --boot 300 --tsv "$OUT/10_power_bootstrap.tsv" > "$OUT/10_power_bootstrap.log"
"$PY" scripts/10_power_curve.py "$DATA_DIR" --reps 200 --seed 0 --shared-eps 0.3 --test bootstrap --boot 300 --tsv "$OUT/10_power_shared_bootstrap.tsv" > "$OUT/10_power_shared_bootstrap.log"
"$PY" scripts/15_real_judge_coverage.py "$DATA_DIR" --reps 1000 --boot 1000 --sizes 10 15 20 --workers "$W" --tsv "$OUT/15_real_judge_coverage.tsv" > "$OUT/15_real_judge_coverage.log"
"$PY" scripts/15_real_judge_coverage.py "$DATA_DIR" --reps 1000 --boot 1000 --sizes 25 --replace --tau-only --workers "$W" --tsv "$OUT/15_real_judge_coverage_matched.tsv" > "$OUT/15_real_judge_coverage_matched.log"
"$PY" scripts/16_real_judge_power.py "$DATA_DIR" --reps 200 --perms 500 --workers "$W" --tsv "$OUT/16_real_judge_power.tsv" > "$OUT/16_real_judge_power.log"
"$PY" scripts/16_real_judge_power.py "$DATA_DIR" --reps 200 --test bootstrap --boot 300 --workers "$W" --tsv "$OUT/16_real_judge_power_bootstrap.tsv" --pvalues "$OUT/16_real_judge_power_bootstrap_pvalues.npz" > "$OUT/16_real_judge_power_bootstrap.log"
"$PY" scripts/17_error_placement.py "$DATA_DIR" --draws 20 --tsv "$OUT/17_error_placement.tsv" > "$OUT/17_error_placement.log"
"$PY" scripts/18_equal_tau_tests.py "$DATA_DIR" --perm-tsv "$ROOT/results/g-j1/06_permutation_seed0.tsv" --tsv "$OUT/18_equal_tau_tests.tsv" --overlap-tsv "$OUT/18_label_overlap.tsv" > "$OUT/18_equal_tau_tests.log"
"$PY" scripts/32_size_adjusted_resolution.py --pvalues "$OUT/16_real_judge_power_bootstrap_pvalues.npz" --overlap "$OUT/18_label_overlap.tsv" --released "$OUT/16_real_judge_power_bootstrap.tsv" --sizes 25 50 100 200 400 --tsv "$OUT/32_size_adjusted_resolution.tsv" > "$OUT/32_size_adjusted_resolution.log"
"$PY" scripts/21_stratified_check.py "$DATA_DIR" --tsv "$OUT/21_stratified_check.tsv" > "$OUT/21_stratified_check.log"
"$PY" scripts/22_wald_coverage.py "$DATA_DIR" --reps 1000 --boot 500 --sizes 10 15 20 --workers "$W" --tsv "$OUT/22_wald_coverage.tsv" > "$OUT/22_wald_coverage.log"
"$PY" scripts/23_test_size_check.py "$DATA_DIR" --reps 200 --perms 200 --boot 100 --sizes 25 100 --workers "$W" --tsv "$OUT/23_test_size_check.tsv" > "$OUT/23_test_size_check.log"
"$PY" scripts/33_calibrated_tied_rejection.py "$DATA_DIR" --out "$OUT/33_calibrated_tied_rejection.npz" > "$OUT/33_calibrated_tied_rejection.log"
"$PY" scripts/25_leaderboard_figure.py "$DATA_DIR" --tsv "$OUT/25_leaderboard_intervals.tsv" > "$OUT/25_leaderboard_figure.log"
# ACM AI Letters copy of the same figure: 7 pt labels, legend above (redrawn from the TSV, nothing recomputed)
"$PY" scripts/25_leaderboard_figure.py --plot-only --tsv "$OUT/25_leaderboard_intervals.tsv" --label-size 7 --height 3.6 --legend-above \
    --out "$PAPER_ROOT/venues/acm-ai-letters/figures/leaderboard.pdf"
"$PY" scripts/26_kappa_separability.py "$DATA_DIR" --tsv "$OUT/26_kappa_separability.tsv" > "$OUT/26_kappa_separability.log"
"$PY" scripts/07_make_permutation_table.py --tsv "$ROOT/results/g-j1/06_permutation_seed0.tsv" --jack-tsv "$OUT/18_equal_tau_tests.tsv" --out "$PAPER_ROOT/manuscript/tables/permutation.tex"
"$PY" scripts/11_disagreement_analysis.py "$DATA_DIR" --tsv "$OUT/11_disagreement.tsv" > "$OUT/11_disagreement.log"
"$PY" scripts/12_make_figures.py
"$PY" scripts/35_make_resolution_table.py
"$PY" scripts/36_heldout_figure.py
# ACM AI Letters copy: wider, so that no label touches the y axis or the resolution bar
"$PY" scripts/36_heldout_figure.py --width 4.2 --out "$PAPER_ROOT/venues/acm-ai-letters/figures/heldout.pdf"
"$PY" scripts/14_power_mechanism.py "$DATA_DIR" --tsv "$OUT/14_power_mechanism.tsv" > "$OUT/14_power_mechanism.log"
"$PY" scripts/13_make_numbers.py
echo "results written to $OUT"
