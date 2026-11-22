"""Step 32. Size-adjusted resolution of the bootstrap z-test (sensitivity check of the paper's resolution results).

At nominal alpha = 0.05 the bootstrap z-test rejects more than 5% of the exactly tied judge pairs' samples
above 100 topics, which inflates power and so makes the resolution finer than a calibrated test would give.
This step recalibrates the test on those pairs and recomputes the resolution. It reads the per-replicate
p-values saved by step 16 (--pvalues) and changes none of the nominal results.

Procedure, fixed before any adjusted result was seen:
  Null set: pairs of distinct judges with exactly equal published 25-topic tau, near-duplicate judges
    (labels differ on < 1% of pairs, results/g-j2/18_label_overlap.tsv) excluded: the paper's 9 tied pairs.
    In this design (topics drawn with replacement from the 25) their population taus are exactly equal.
    Both calibrations are therefore conditional on these 9 pairs and this design: they control the
    rejection rate on them (on average, or at worst), not the type I error for every pair of equal judges.
  Pooled calibration (main): at each size, the critical p-value c is the k-th smallest of the pooled null
    p-values (9 pairs x reps), k = floor(alpha * N), so at most alpha of the null samples have p < c.
  Worst-pair calibration (strictest): c is the smallest of the per-pair critical values computed the
    same way, so no tied pair rejects more than alpha of its samples.
  Adjusted power of every pair: share of its replicates with p < c. Resolution: as in the paper (smallest
    gap g such that >= 80% of pairs at least g apart reach 80% power; near-duplicates excluded).
  Uncertainty: --mc draws; each redraws the null p-values (tied pairs with replacement, then each pair's
    replicates with replacement), recomputes c, redraws every other pair's replicates with replacement and
    recomputes the resolution. Reported as the 5-95% range and the share of draws resolving no gap.

Usage: python scripts/32_size_adjusted_resolution.py [--pvalues results/g-j2/16_real_judge_power_bootstrap_pvalues.npz]
           [--overlap results/g-j2/18_label_overlap.tsv] [--sizes 200 400] [--mc 2000] [--seed 32] [--tsv FILE]
"""
import argparse
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader([l for l in fh if not l.startswith("#")], delimiter="\t"))


def critical(p_null, alpha):
    """Largest threshold c among the null p-values with share(p_null < c) <= alpha."""
    s = np.sort(np.asarray(p_null).ravel())
    k = int(np.floor(alpha * len(s)))
    return s[k] if k < len(s) else np.inf


def resolution(gaps, power):
    for g in np.unique(gaps):
        tail = gaps >= g - 1e-12
        if (power[tail] >= 0.8).mean() >= 0.8:
            return g, int(tail.sum())
    return None, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pvalues", default=os.path.join(ROOT, "results", "g-j2", "16_real_judge_power_bootstrap_pvalues.npz"))
    ap.add_argument("--overlap", default=os.path.join(ROOT, "results", "g-j2", "18_label_overlap.tsv"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 400])
    ap.add_argument("--mc", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--released", default=os.path.join(ROOT, "results", "g-j2", "16_real_judge_power_bootstrap.tsv"),
                    help="step 16 TSV whose powers the nominal powers here must equal")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    z = np.load(args.pvalues)
    p, sizes, gaps, alpha = z["p"], [int(n) for n in z["sizes"]], z["gaps"], float(z["alpha"])
    A, B = z["judge_a"], z["judge_b"]
    dup = {frozenset((r["judge_a"], r["judge_b"])) for r in read_tsv(args.overlap) if float(r["label_diff_share"]) < 0.01}
    keep = np.array([frozenset((a, b)) not in dup for a, b in zip(A, B)])
    null = keep & (gaps < 1e-9)
    if os.path.exists(args.released):
        rel = {(r["judge_a"], r["judge_b"]): [float(r[f"power_{n}"]) for n in sizes] for r in read_tsv(args.released)}
        nom = (p < alpha).mean(2)
        diff = max(abs(nom[i, k] - rel[(a, b)][k]) for i, (a, b) in enumerate(zip(A, B)) for k in range(len(sizes)))
        print(f"nominal powers vs {os.path.relpath(args.released, ROOT)}: largest difference {diff:.6f}")
        if diff > 5e-5:  # the TSV keeps four decimals
            raise SystemExit("nominal powers differ from the released step 16 results")
    p, gaps = p[keep], gaps[keep]
    null = null[keep]
    rng = np.random.default_rng(args.seed)
    print(f"step 16 p-values: {p.shape[0]} pairs after {int((~keep).sum())} near-duplicate pairs, sizes {sizes}, "
          f"{p.shape[2]} replicates, test={z['test']} B={int(z['boot'])} alpha={alpha} seed={int(z['seed'])}")
    print(f"null set: {int(null.sum())} exactly tied pairs; MC draws {args.mc}, seed {args.seed}\n")

    rows = []
    for n in args.sizes:
        k = sizes.index(n)
        P = p[:, k, :]
        nominal = (P < alpha).mean(1)
        c_pool = critical(P[null], alpha)
        c_worst = min(critical(P[i], alpha) for i in np.flatnonzero(null))
        out = dict(n=n)
        for label, c in (("nominal", alpha), ("pooled", c_pool), ("worst_pair", c_worst)):
            pw = (P < c).mean(1)
            g, cnt = resolution(gaps, pw)
            size_med, size_max = np.median(pw[null]), pw[null].max()
            big = gaps >= 0.10 - 1e-12
            out.update({f"{label}_crit": c, f"{label}_res": g, f"{label}_npairs": cnt,
                        f"{label}_size_med": size_med, f"{label}_size_max": size_max,
                        f"{label}_pow10_med": float(np.median(pw[big]))})
            print(f"n = {n:3d}  {label:10s} critical p {c:.4f} | tied pairs reject median {size_med:.3f} (max {size_max:.3f}) | "
                  f"pairs >= 0.10 apart: median power {np.median(pw[big]):.2f} | resolution "
                  + (f"{g:.3f} ({cnt} pairs)" if g is not None else "none"))
        # Monte Carlo: calibration and replicate uncertainty together (pooled calibration)
        idx_null = np.flatnonzero(null)
        sims = []
        reps = P.shape[1]
        for _ in range(args.mc):
            pick = rng.choice(idx_null, len(idx_null), replace=True)
            pn = np.concatenate([P[i, rng.integers(0, reps, reps)] for i in pick])
            c = critical(pn, alpha)
            Pr = np.take_along_axis(P, rng.integers(0, reps, P.shape), 1)
            sims.append(resolution(gaps, (Pr < c).mean(1))[0])
        ok = np.array([s for s in sims if s is not None])
        none = 1 - len(ok) / len(sims)
        out.update(mc_lo=np.quantile(ok, 0.05) if len(ok) else np.nan, mc_hi=np.quantile(ok, 0.95) if len(ok) else np.nan,
                   mc_none=none)
        print(f"n = {n:3d}  pooled, Monte Carlo (calibration + replicates): "
              + (f"5-95% {out['mc_lo']:.3f}-{out['mc_hi']:.3f}; " if len(ok) else "")
              + f"no gap resolved in {100 * none:.0f}% of draws\n")
        rows.append(out)

    if args.tsv:
        cols = list(rows[0].keys())
        with open(args.tsv, "w") as f:
            f.write(f"# size-adjusted resolution of the bootstrap z-test; null = {int(null.sum())} exactly tied pairs; "
                    f"mc={args.mc} seed={args.seed}\n" + "\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join("none" if r[c] is None else (f"{r[c]:.6f}" if isinstance(r[c], (float, np.floating)) else str(r[c]))
                                  for c in cols) + "\n")


if __name__ == "__main__":
    main()
