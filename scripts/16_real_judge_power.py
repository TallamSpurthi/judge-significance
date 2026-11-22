"""Step 16. Power of the paired permutation test for every pair of real LLMJudge judges, and the
smallest gap in tau that a benchmark of a given size can detect.

The 25 LLMJudge test topics are an empirical topic distribution and the 33 real judges are used as
they are, so each pair's gap is its published 25-topic gap. For each of the 528 pairs and each
topic-set size, topics are drawn with replacement from the 25, and power is the share of replicates
in which the test rejects at alpha = 0.05: the paired permutation test (--test permutation, null:
per-topic exchangeability), the bootstrap z-test (--test bootstrap, null: equal tau, bootstrap SE
from --boot replicates) or the jackknife t-test (--test jackknife; no power, kept for the record).
Pairs run in parallel.
The summary bins pairs by gap and reports, per size, the median power and the share of pairs with
power >= 0.8, and the topics needed for 80% power for the adjacent pairs of the published order.

--pvalues FILE.npz also saves every replicate's p-value (array p[pair, size, replicate]) for size
adjustment (step 32); it changes no random draw, so the powers equal those of a run without it.

Usage: python scripts/16_real_judge_power.py DATA_DIR [--test permutation|bootstrap|jackknife] [--reps 200] [--perms 500] [--boot 300]
           [--sizes 25 50 100 200 400] [--seed 0] [--workers 10] [--tsv FILE] [--pvalues FILE.npz]
"""
import argparse
import importlib.util
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from judge_significance.permutation import bootstrap_z_test, jackknife_t_test, perm_test
from judge_significance.simulation import stat_rows

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cov15", os.path.join(HERE, "15_real_judge_coverage.py"))
_cov15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cov15)

BINS = (0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 1.0)
_G = {}


def _init(H, J):
    _G["H"], _G["J"] = H, J


def _power(task):
    a, b, sizes, reps, perms, alpha, seed, test, n_boot = task
    H, J = _G["H"], _G["J"]
    rng = np.random.default_rng(seed)
    out, pv = [], []
    for n in sizes:
        rej, ps = 0, []
        for _ in range(reps):
            S = rng.integers(0, H.shape[1], size=n)
            if test == "bootstrap":
                _, p = bootstrap_z_test(H[:, S], J[a][:, S], J[b][:, S], n_boot, rng)
            elif test == "jackknife":
                _, p = jackknife_t_test(H[:, S], J[a][:, S], J[b][:, S])
            else:
                _, p = perm_test(H[:, S].mean(1), J[a][:, S], J[b][:, S], perms, rng)
            rej += p < alpha
            ps.append(p)
        out.append(rej / reps)
        pv.append(ps)
    return out, pv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--test", choices=("permutation", "bootstrap", "jackknife"), default="permutation")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tsv")
    ap.add_argument("--pvalues", help="save every replicate's p-value to this .npz file")
    args = ap.parse_args()

    t0 = time.time()
    H, J, systems, topics = _cov15.real_population(args.data_dir)
    truth = {j: float(stat_rows(H.mean(1)[None], M.mean(1)[None])[0]) for j, M in J.items()}
    pub = _cov15.D.read_published(_cov15.PUBLISHED)
    names = sorted(truth, key=lambda j: (-pub[j]["tau"], j))
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    seeds = np.random.SeedSequence([args.seed, 16]).spawn(len(pairs))
    tasks = [(a, b, args.sizes, args.reps, args.perms, args.alpha, int(s.generate_state(1)[0]), args.test, args.boot)
             for (a, b), s in zip(pairs, seeds)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(H, J)) as ex:
        results = list(ex.map(_power, tasks, chunksize=4))
    powers = [r[0] for r in results]
    gaps = np.abs(np.array([truth[a] - truth[b] for a, b in pairs]))
    P = np.array(powers)  # [pairs, sizes]

    print(f"population: {len(systems)} runs, the {len(topics)} LLMJudge test topics (sampled with replacement); "
          f"33 real judges, {len(pairs)} pairs; test={args.test} boot={args.boot} reps={args.reps} "
          f"perms={args.perms} alpha={args.alpha} seed={args.seed} ({time.time() - t0:.0f} s, {args.workers} workers)")
    print("\n  25-topic gap      pairs | median power at n = " + " ".join(f"{n:>5d}" for n in args.sizes)
          + " | share of pairs with power >= 0.8")
    for lo, hi in zip(BINS, BINS[1:]):
        m = (gaps >= lo) & (gaps < hi)
        if not m.any():
            continue
        med = np.median(P[m], 0)
        share = (P[m] >= 0.8).mean(0)
        print(f"  [{lo:.3f}, {hi:.3f})  {m.sum():6d} | " + " ".join(f"{x:5.2f}" for x in med)
              + "   | " + " ".join(f"{x:5.2f}" for x in share))
    print("\n  smallest 25-topic gap g such that at least 80% of pairs with gap >= g reach 80% power:")
    for k, n in enumerate(args.sizes):
        g_need = None
        for g in np.unique(gaps):  # smallest threshold whose whole tail (gap >= g) qualifies
            if (P[gaps >= g - 1e-12, k] >= 0.8).mean() >= 0.8:
                g_need = g
                break
        print(f"    n = {n:4d}: " + (f"{g_need:.3f}" if g_need is not None else "none"))
    print("\n  adjacent pairs in the published order: gap, power at each size, topics for 80% power")
    idx = {p: i for i, p in enumerate(pairs)}
    for a, b in zip(names, names[1:]):
        i = idx[(a, b)]
        pts = list(zip(args.sizes, P[i]))
        need = pts[0][0] if pts[0][1] >= 0.8 else None
        for (n0, p0), (n1, p1) in zip(pts, pts[1:]):
            if need is None and p0 < 0.8 <= p1:
                need = n0 + (0.8 - p0) * (n1 - n0) / (p1 - p0)
        print(f"    {a + ' vs ' + b:55s} {gaps[i]:.4f} " + " ".join(f"{x:5.2f}" for x in P[i])
              + "  " + (f"about {need:.0f}" if need else f"more than {args.sizes[-1]}"))
    print(f"\n  Monte Carlo SE of a power estimate near 0.5 at {args.reps} reps: {np.sqrt(0.25 / args.reps):.3f}")

    if args.pvalues:
        np.savez_compressed(args.pvalues, p=np.array([r[1] for r in results]), sizes=np.array(args.sizes),
                            judge_a=np.array([a for a, _ in pairs]), judge_b=np.array([b for _, b in pairs]),
                            gaps=gaps, alpha=args.alpha, seed=args.seed, test=args.test, boot=args.boot, reps=args.reps)
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# real judges, population = {len(topics)} test topics (with replacement); test={args.test} boot={args.boot} reps={args.reps} "
                    f"perms={args.perms} alpha={args.alpha} seed={args.seed}\n")
            f.write("judge_a\tjudge_b\ttau_a\ttau_b\tabs_delta\t" + "\t".join(f"power_{n}" for n in args.sizes) + "\n")
            for (a, b), g, row in zip(pairs, gaps, powers):
                f.write(f"{a}\t{b}\t{truth[a]:.6f}\t{truth[b]:.6f}\t{g:.6f}\t" + "\t".join(f"{x:.4f}" for x in row) + "\n")


if __name__ == "__main__":
    main()
