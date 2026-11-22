"""Step 22. Coverage of the interval that the bootstrap z-test inverts, for real judge pairs.

The bootstrap z-test rejects equal tau when |delta| > 1.96 * SE, with SE the standard deviation of
delta over topic-bootstrap replicates; it inverts the normal (Wald) interval delta +- 1.96 * SE.
Design as in step 15: the 25 LLMJudge topics are the population, the target is each adjacent pair's
25-topic gap (the published gap), and each replicate draws n of the 25 topics without replacement.
Near-duplicate judges and pairs whose gap is zero in every replicate are skipped.

Usage: python scripts/22_wald_coverage.py DATA_DIR [--reps 1000] [--boot 1000] [--sizes 10 15 20]
           [--seed 0] [--workers 6] [--tsv FILE]
"""
import argparse
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from judge_significance.simulation import stat_rows

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cov15", os.path.join(HERE, "15_real_judge_coverage.py"))
_cov15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cov15)
_G = {}


def _init(H, J):
    _G["H"], _G["J"] = H, J


def _means(M, idx):
    return M[:, idx].mean(2).T


def _cover(task):
    a, b, truth, n, reps, n_boot, seed = task
    H, J = _G["H"], _G["J"]
    rng = np.random.default_rng(seed)
    hit = valid = 0
    widths = []
    for _ in range(reps):
        S = rng.choice(H.shape[1], size=n, replace=False)
        Hs, As, Bs = H[:, S], J[a][:, S], J[b][:, S]
        full = np.arange(n)[None]
        d = stat_rows(_means(Hs, full), _means(As, full), _means(Bs, full))[0]
        idx = rng.integers(0, n, size=(n_boot, n))
        se = stat_rows(_means(Hs, idx), _means(As, idx), _means(Bs, idx)).std(ddof=1)
        if not np.isfinite(se) or se <= 1e-12:
            continue
        valid += 1
        hit += abs(d - truth) <= 1.959964 * se + 1e-9
        widths.append(2 * 1.959964 * se)
    return dict(a=a, b=b, truth=truth, n=n, valid=valid, cov=hit / valid if valid else np.nan,
                width=float(np.mean(widths)) if widths else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 20])
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
    truth = {j: float(stat_rows(H.mean(1)[None], M.mean(1)[None])[0]) for j, M in J.items()}
    order = sorted(J, key=lambda j: (-pub[j]["tau"], j))
    pairs = [(a, b) for a, b in zip(order, order[1:]) if frozenset((a, b)) not in dup]
    seeds = np.random.SeedSequence([args.seed, 22]).spawn(len(pairs) * len(args.sizes))
    tasks = [(a, b, truth[a] - truth[b], n, args.reps, args.boot, int(seeds[k * len(pairs) + i].generate_state(1)[0]))
             for k, n in enumerate(args.sizes) for i, (a, b) in enumerate(pairs)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(H, J)) as ex:
        rows = list(ex.map(_cover, tasks))
    print(f"real judges, population = {len(topics)} topics; {len(pairs)} adjacent pairs (near-duplicates skipped: "
          f"{31 + 1 - len(pairs) if False else len(list(zip(order, order[1:]))) - len(pairs)}); reps={args.reps} "
          f"B={args.boot} sizes={args.sizes} seed={args.seed}")
    for n in args.sizes:
        cov = np.array([r["cov"] for r in rows if r["n"] == n and np.isfinite(r["cov"])])
        print(f"  n = {n:2d}: Wald interval coverage of the 25-topic gap: min {cov.min():.3f}, median "
              f"{np.median(cov):.3f}, max {cov.max():.3f}; pairs with coverage >= 0.90: {(cov >= 0.90).sum()}/{len(cov)}")
    print(f"  Monte Carlo SE of a 0.95 coverage at {args.reps} reps: {np.sqrt(0.95 * 0.05 / args.reps):.3f}")
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# Wald (bootstrap-SE) interval coverage, real adjacent pairs; reps={args.reps} boot={args.boot} seed={args.seed}\n"
                    "judge_a\tjudge_b\ttruth\tn\tvalid\tcoverage\tmean_width\n")
            for r in rows:
                f.write(f"{r['a']}\t{r['b']}\t{r['truth']:.6f}\t{r['n']}\t{r['valid']}\t{r['cov']:.6f}\t{r['width']:.6f}\n")


if __name__ == "__main__":
    main()
