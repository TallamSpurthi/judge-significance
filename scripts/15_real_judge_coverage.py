"""Step 15. Coverage of bootstrap intervals for tau and delta tau for the real LLMJudge judges.

The 25 LLMJudge test topics are the population and the 33 real judges are used as they are, so the
target is each judge's tau on all 25 topics (the published Table 3 value) and the difference for
adjacent judges in the published order. Each replicate draws n of the 25 topics without replacement
and computes 95% percentile, BCa and studentized intervals. Drawing a large share of a small
population makes the true sampling variance smaller than the bootstrap assumes, so intervals
should, if anything, over-cover here. Targets run in parallel.

--replace draws the n topics WITH replacement instead (the matched design): the population is then the
empirical distribution of the 25 topics, whose mean scores are the 25-topic means, so the target is
still each judge's published tau, and the outer sampling is exactly the one the bootstrap assumes.
--tau-only skips the adjacent-gap targets.

Usage: python scripts/15_real_judge_coverage.py DATA_DIR [--reps 1000] [--boot 1000]
           [--sizes 10 15 20] [--seed 0] [--workers 10] [--replace] [--tau-only] [--tsv FILE]
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.bootstrap import TIE_TOL
from judge_significance.protocol import table3_systems
from judge_significance.simulation import intervals, stat_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")
METHODS = ("percentile", "bca", "studentized")
_G = {}


def real_population(data_dir):
    """Human and real-judge per-topic nDCG@10 on the 25 test topics (31 Table 3 runs)."""
    data = D.load(data_dir)
    runs, test = data["runs"], data["test_topics"]
    systems = table3_systems(runs)
    H = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10")
    J = {j: E.score_matrix(runs, systems, q, test, "ndcg_cut_10") for j, q in data["judges"].items()}
    return H, J, systems, test


def _init(H, J):
    _G["H"], _G["J"] = H, J


def _coverage(task):
    a, b, truth, n, reps, boot, seed, replace = task
    H, J = _G["H"], _G["J"]
    rng = np.random.default_rng(seed)
    est, dropped, ivs = [], 0, {m: [] for m in METHODS}
    for _ in range(reps):
        S = np.sort(rng.choice(H.shape[1], size=n, replace=replace))
        out = intervals(H, J[a], None if b is None else J[b], S, boot, rng)
        est.append(out["estimate"])
        dropped += out["studentized_dropped"]
        for m in METHODS:
            ivs[m].append(out[m])
    row = dict(n=n, target=a if b is None else f"{a} - {b}", kind="tau" if b is None else "delta",
               truth=truth, mean_est=float(np.mean(est)), stud_dropped_share=dropped / (reps * boot))
    for m in METHODS:
        iv = np.array(ivs[m])
        valid = np.isfinite(iv).all(1)
        lo, hi = iv[valid, 0], iv[valid, 1]
        if not valid.any():
            row.update({f"{m}_cov": np.nan, f"{m}_above": np.nan, f"{m}_below": np.nan,
                        f"{m}_width": np.nan, f"{m}_valid": 0})
            continue
        row.update({f"{m}_cov": float(((lo - TIE_TOL <= truth) & (truth <= hi + TIE_TOL)).mean()),
                    f"{m}_above": float((truth < lo - TIE_TOL).mean()),
                    f"{m}_below": float((truth > hi + TIE_TOL).mean()),
                    f"{m}_width": float((hi - lo).mean()), f"{m}_valid": int(valid.sum())})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--replace", action="store_true", help="draw topics with replacement (matched design)")
    ap.add_argument("--tau-only", action="store_true", help="only the 33 single-judge tau targets")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    t0 = time.time()
    H, J, systems, topics = real_population(args.data_dir)
    pub = D.read_published(PUBLISHED)
    truth = {j: float(stat_rows(H.mean(1)[None], M.mean(1)[None])[0]) for j, M in J.items()}
    n_exact = sum(round(truth[j], 4) == pub[j]["tau"] for j in truth)
    order = sorted(truth, key=lambda j: (-pub[j]["tau"], j))
    pairs = list(zip(order, order[1:]))
    print(f"population: {len(systems)} runs, the {len(topics)} LLMJudge test topics, 33 real judges; "
          f"reps={args.reps} B={args.boot} sizes={args.sizes} seed={args.seed} "
          f"outer draws {'with' if args.replace else 'without'} replacement")
    print(f"target tau on all {len(topics)} topics equals published Table 3 for {n_exact}/33 judges")

    targets = [(j, None, truth[j]) for j in order]
    if not args.tau_only:
        targets += [(a, b, truth[a] - truth[b]) for a, b in pairs]
    seeds = np.random.SeedSequence([args.seed, 15]).spawn(len(targets) * len(args.sizes))
    tasks = [(a, b, tv, n, args.reps, args.boot, int(seeds[k * len(targets) + i].generate_state(1)[0]), args.replace)
             for k, n in enumerate(args.sizes) for i, (a, b, tv) in enumerate(targets)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(H, J)) as ex:
        rows = list(ex.map(_coverage, tasks))
    print(f"({time.time() - t0:.0f} s, {args.workers} workers)")

    for n in args.sizes:
        rn = [r for r in rows if r["n"] == n]
        print(f"\n== n = {n} of {len(topics)} topics ==")
        print(f"  {'target':58s} {'truth':>7s} {'mean est':>8s} | "
              + " | ".join(f"{m[:5]:>5s} cov  low  high  width" for m in METHODS))
        for r in rn:
            print(f"  {r['target']:58s} {r['truth']:7.4f} {r['mean_est']:8.4f} | "
                  + " | ".join(f"{r[m + '_cov']:5.3f} {r[m + '_above']:5.3f} {r[m + '_below']:5.3f} {r[m + '_width']:6.3f}"
                               for m in METHODS))
        # identical or near-identical judges (e.g. NISTRetrieval-reason0/2) give delta = 0 in every replicate;
        # every interval is then the single point 0 and the studentized interval is undefined
        degenerate = [r["target"] for r in rn if r["kind"] == "delta" and r["percentile_width"] < TIE_TOL]
        print(f"\n  degenerate delta targets (delta = 0 in every replicate; excluded from the summary): "
              f"{len(degenerate)}" + (f" ({', '.join(degenerate)})" if degenerate else ""))
        print("  summary (min, median, max coverage):")
        for kind in ("tau", "delta"):
            sub = [r for r in rn if r["kind"] == kind and r["target"] not in degenerate]
            if not sub:
                continue
            print(f"    {kind:5s} " + "; ".join(
                f"{m} {np.nanmin([r[m + '_cov'] for r in sub]):.3f}, {np.nanmedian([r[m + '_cov'] for r in sub]):.3f}, "
                f"{np.nanmax([r[m + '_cov'] for r in sub]):.3f}" for m in METHODS))
            print("          share of targets with coverage >= 0.90: " + "; ".join(
                f"{m} {np.mean([r[m + '_cov'] >= 0.90 for r in sub]):.2f}" for m in METHODS))
            bias = np.array([r["mean_est"] - r["truth"] for r in sub])
            print(f"          mean estimate minus target: {bias.min():+.4f} to {bias.max():+.4f} (median {np.median(bias):+.4f})")
    drop = max(r["stud_dropped_share"] for r in rows if r["percentile_width"] >= TIE_TOL)
    print(f"\n  studentized (non-degenerate targets): largest share of replicates dropped for zero jackknife SE: {drop:.4f}")
    print("  cov = coverage of the 25-topic target; low = interval entirely above target; high = entirely below."
          f" Monte Carlo SE of a 0.95 coverage at {args.reps} reps: {np.sqrt(0.95 * 0.05 / args.reps):.3f}")

    if args.tsv:
        cols = list(rows[0].keys())
        with open(args.tsv, "w") as f:
            f.write(f"# real judges, population = {len(topics)} test topics; reps={args.reps} boot={args.boot} "
                     f"seed={args.seed} replace={int(args.replace)}\n" + "\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
