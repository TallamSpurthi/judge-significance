"""Step 33. Tied-pair rejection rate after step 23's calibration (check of the paper's calibrated powers).

Step 23 sets each test's critical p-value to the 5% quantile of the p-values pooled over the 9 exactly tied
pairs and rejects when p <= c. It does not save its p-values. This step replays step 23 exactly for the tied
pairs only: same pair list, same per-task seeds (SeedSequence([seed, 23]).spawn(...)), same _run(), so the
p-values are step 23's own. It checks the replay against the released step 23 TSVs, saves the p-values, and
reports the share of tied samples at or below each calibrated cutoff (computed at full precision; the
adjusted TSV keeps six decimals). Permutation p-values are discrete (multiples of 1/(P+1)), so that share can
exceed 5%. Single process; about 7 minutes. Changes no other result.

Usage: python scripts/33_calibrated_tied_rejection.py DATA_DIR [--out results/g-j2/33_calibrated_tied_rejection.npz]
"""
import argparse
import csv
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("s23", os.path.join(HERE, "23_test_size_check.py"))
s23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s23)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--seed", type=int, default=0)  # step 23's settings in run_g_j2.sh
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 100])
    ap.add_argument("--overlap-tsv", default=os.path.join(ROOT, "results", "g-j2", "18_label_overlap.tsv"))
    ap.add_argument("--released", default=os.path.join(ROOT, "results", "g-j2", "23_test_size_check.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "g-j2", "33_calibrated_tied_rejection.npz"))
    args = ap.parse_args()

    H, J, systems, topics = s23._cov15.real_population(args.data_dir)
    pub = s23._cov15.D.read_published(s23._cov15.PUBLISHED)
    dup = set()
    with open(args.overlap_tsv) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) == 3 and not line.startswith(("#", "judge_a")) and float(c[2]) < 0.01:
                dup.add(frozenset(c[:2]))
    names = sorted(J)
    pairs = []  # the same construction as step 23's main()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = abs(pub[a]["tau"] - pub[b]["tau"])
            if gap == 0:
                pairs.append(("near_duplicate_tie" if frozenset((a, b)) in dup else "exact_tie", a, b))
            elif gap >= 0.10 - 1e-9:
                pairs.append(("gap_ge_0.10", a, b))
    seeds = np.random.SeedSequence([args.seed, 23]).spawn(len(pairs) * len(args.sizes))
    tasks = [(g, a, b, n, args.reps, args.perms, args.boot, int(seeds[k * len(pairs) + i].generate_state(1)[0]))
             for k, n in enumerate(args.sizes) for i, (g, a, b) in enumerate(pairs)]
    tasks = [t for t in tasks if t[0] == "exact_tie"]
    s23._init(H, J)
    rows = [s23._run(t) for t in tasks]

    with open(args.released) as f:
        rel = {(r["judge_a"], r["judge_b"], int(r["n"])): r
               for r in csv.DictReader([l for l in f if not l.startswith("#")], delimiter="\t")}
    mism = sum(abs(float(rel[(r["a"], r["b"], r["n"])][t]) - r[t]) > 1e-9 for r in rows for t in s23.TESTS)
    print(f"step 23 replay: {len(rows)} tied (pair, size) tasks; seed={args.seed} reps={args.reps} perms={args.perms} "
          f"boot={args.boot}; per-pair rejection rates differing from {os.path.relpath(args.released, ROOT)}: {mism}")
    if mism:
        raise SystemExit("replay does not match the released step 23 results")
    np.savez_compressed(args.out, **{f"{t}_{n}": np.concatenate([r["p"][t] for r in rows if r["n"] == n])
                                     for n in args.sizes for t in s23.TESTS})
    print(f"saved tied-pair p-values to {os.path.relpath(args.out, ROOT)}")
    for n in args.sizes:
        for t in s23.TESTS:
            p = np.concatenate([r["p"][t] for r in rows if r["n"] == n])
            c = float(np.quantile(p, 0.05))  # step 23's cutoff, as computed in its main()
            print(f"  n={n:3d} {t:24s} critical p {c:.6f}: tied samples rejected {int((p <= c).sum())}/{len(p)} "
                  f"= {np.mean(p <= c):.4f} (nominal p < 0.05: {np.mean(p < 0.05):.4f})")


if __name__ == "__main__":
    main()
