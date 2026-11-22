"""Step 23. Size and power of three tests on real LLMJudge judge pairs, including a studentized
permutation test (Chung and Romano 2013).

Pairs: (a) every pair of distinct judges with exactly equal published tau ("exact ties"; size), with
near-duplicate judges (labels differ on < 1% of pairs) reported separately, and (b) every pair at
least 0.10 apart (power). Topic sets of n are drawn with replacement from the 25 LLMJudge topics,
whose published tau is the target. Tests: paired permutation (null: exchangeability), studentized
permutation (delta / bootstrap SE, permuted), and bootstrap z-test (null: equal tau).

Usage: python scripts/23_test_size_check.py DATA_DIR [--reps 200] [--perms 200] [--boot 100]
           [--sizes 25 100] [--seed 0] [--workers 6] [--tsv FILE]
"""
import argparse
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from judge_significance.permutation import bootstrap_z_test, perm_test, studentized_perm_test

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cov15", os.path.join(HERE, "15_real_judge_coverage.py"))
_cov15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cov15)
TESTS = ("permutation", "studentized_permutation", "bootstrap_z")
_G = {}


def _init(H, J):
    _G["H"], _G["J"] = H, J


def _run(task):
    group, a, b, n, reps, perms, n_boot, seed = task
    H, J = _G["H"], _G["J"]
    rng = np.random.default_rng(seed)
    ps = {t: [] for t in TESTS}
    for _ in range(reps):
        S = rng.integers(0, H.shape[1], size=n)
        Hs, As, Bs = H[:, S], J[a][:, S], J[b][:, S]
        ps["permutation"].append(perm_test(Hs.mean(1), As, Bs, perms, rng)[1])
        ps["studentized_permutation"].append(studentized_perm_test(Hs, As, Bs, perms, n_boot, rng)[1])
        ps["bootstrap_z"].append(bootstrap_z_test(Hs, As, Bs, n_boot, rng)[1])
    return dict(group=group, a=a, b=b, n=n, p=ps, **{t: float(np.mean(np.array(ps[t]) < 0.05)) for t in TESTS})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 100])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--overlap-tsv", default=os.path.join(HERE, "..", "results", "g-j2", "18_label_overlap.tsv"))
    ap.add_argument("--tsv")
    args = ap.parse_args()

    H, J, systems, topics = _cov15.real_population(args.data_dir)
    pub = _cov15.D.read_published(_cov15.PUBLISHED)
    dup = set()
    with open(args.overlap_tsv) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) == 3 and not line.startswith(("#", "judge_a")) and float(c[2]) < 0.01:
                dup.add(frozenset(c[:2]))
    names = sorted(J)
    pairs = []
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
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(H, J)) as ex:
        rows = list(ex.map(_run, tasks))
    print(f"real judges, population = {len(topics)} topics (with replacement); reps={args.reps} perms={args.perms} "
          f"boot={args.boot} seed={args.seed}")
    for g in ("exact_tie", "near_duplicate_tie", "gap_ge_0.10"):
        for n in args.sizes:
            sub = [r for r in rows if r["group"] == g and r["n"] == n]
            if not sub:
                continue
            print(f"  {g:20s} pairs={len(sub):3d} n={n:3d}: " + "; ".join(
                f"{t} median {np.median([r[t] for r in sub]):.3f} (max {max(r[t] for r in sub):.3f})" for t in TESTS))
    print(f"  Monte Carlo SE of a rate near 0.05 at {args.reps} reps: {np.sqrt(0.05 * 0.95 / args.reps):.3f}")
    # size-adjusted power: per test and size, the critical p-value is the 5% quantile of the p-values pooled
    # over the exactly tied pairs, so every test rejects 5% of tied samples; power is then the share of
    # samples of pairs >= 0.10 apart with p at or below that critical value
    adj = []
    print("  size-adjusted power (critical value set so that 5% of exactly tied samples are rejected):")
    for n in args.sizes:
        ties = [r for r in rows if r["group"] == "exact_tie" and r["n"] == n]
        big = [r for r in rows if r["group"] == "gap_ge_0.10" and r["n"] == n]
        cells = []
        for t in TESTS:
            c = float(np.quantile(np.concatenate([r["p"][t] for r in ties]), 0.05))
            pw = [float(np.mean(np.array(r["p"][t]) <= c)) for r in big]
            adj.append(dict(n=n, test=t, critical_p=c, median_power=float(np.median(pw))))
            cells.append(f"{t} {np.median(pw):.3f} (critical p {c:.4f})")
        print(f"    n={n:3d}: " + "; ".join(cells))
    if args.tsv:
        with open(args.tsv.replace(".tsv", "_adjusted.tsv"), "w") as f:
            f.write("n\ttest\tcritical_p\tmedian_power\n")
            for r in adj:
                f.write(f"{r['n']}\t{r['test']}\t{r['critical_p']:.6f}\t{r['median_power']:.4f}\n")
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# reps={args.reps} perms={args.perms} boot={args.boot} seed={args.seed}\n"
                    "group\tjudge_a\tjudge_b\tn\t" + "\t".join(TESTS) + "\n")
            for r in rows:
                f.write(f"{r['group']}\t{r['a']}\t{r['b']}\t{r['n']}\t" + "\t".join(f"{r[t]:.4f}" for t in TESTS) + "\n")


if __name__ == "__main__":
    main()
