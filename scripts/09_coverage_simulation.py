"""Step 9. Coverage of bootstrap intervals for tau and delta tau (synthetic judges, known truth).

Population: the 82 DL-2023 topics with full NIST qrels and the 31 Table 3 runs. Synthetic judges
perturb the human labels (judge_significance.simulation). The target is each judge's tau on all
82 topics (and the difference for judge pairs). Each replicate draws n topics without replacement,
computes 95% percentile, BCa and studentized intervals, and records whether they cover the target.

Usage: python scripts/09_coverage_simulation.py DATA_DIR [--reps 1000] [--boot 1000]
                                                [--sizes 15 25] [--seed 0] [--tsv FILE]
"""
import argparse
import time

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.protocol import table3_systems
from judge_significance.bootstrap import TIE_TOL
from judge_significance.simulation import intervals, stat_rows, synthetic_judge

JUDGES = [("random", 0.2), ("random", 0.4), ("random", 0.6), ("lenient", 0.3), ("lenient", 0.6)]
METHODS = ("percentile", "bca", "studentized")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[15, 25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shared-eps", type=float, default=0.0,
                    help="if > 0, every judge is derived from one shared noisy base judge "
                         "(random noise at this level), so judges share errors")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    runs, full = data["runs"], data["full"]
    systems = table3_systems(runs)
    topics = sorted(full)
    H = E.score_matrix(runs, systems, full, topics, "ndcg_cut_10")
    rng = np.random.default_rng(args.seed)
    # correlated-errors variant: judges perturb a shared noisy base instead of the human labels
    base = synthetic_judge(full, "random", args.shared_eps, rng) if args.shared_eps > 0 else full
    J = {}
    for model, eps in JUDGES:
        name = f"{model}-{eps}"
        J[name] = E.score_matrix(runs, systems, synthetic_judge(base, model, eps, rng), topics, "ndcg_cut_10")
    # same implementation as the intervals (taub_pairs via stat_rows), so equal taus compare equal
    truth = {j: float(stat_rows(H.mean(1)[None], M.mean(1)[None])[0]) for j, M in J.items()}
    order = sorted(truth, key=lambda j: -truth[j])
    pairs = list(zip(order, order[1:]))
    print(f"population: {len(systems)} runs, {len(topics)} topics; shared_eps={args.shared_eps} reps={args.reps} B={args.boot} "
          f"sizes={args.sizes} seed={args.seed}")
    print("synthetic judges, tau on all 82 topics (target):")
    for j in order:
        print(f"  {j:12s} tau82={truth[j]:.4f}")
    print("adjacent synthetic pairs, delta tau on all 82 topics (target):")
    for a, b in pairs:
        print(f"  {a} - {b}: {truth[a] - truth[b]:+.4f}")

    targets = [(j, None, truth[j]) for j in order] + [(a, b, truth[a] - truth[b]) for a, b in pairs]
    rows = []
    for n in args.sizes:
        t0 = time.time()
        rec = {(a, b): {"est": [], "dropped": 0, **{m: [] for m in METHODS}} for a, b, _ in targets}
        for _ in range(args.reps):
            S = np.sort(rng.choice(len(topics), size=n, replace=False))
            for a, b, _ in targets:
                out = intervals(H, J[a], None if b is None else J[b], S, args.boot, rng)
                rec[(a, b)]["est"].append(out["estimate"])
                rec[(a, b)]["dropped"] += out["studentized_dropped"]
                for m in METHODS:
                    rec[(a, b)][m].append(out[m])
        print(f"\n== n = {n} topics ({time.time() - t0:.0f} s) ==")
        print(f"  {'target':28s} {'truth':>7s} {'mean est':>8s} | " +
              " | ".join(f"{m[:5]:>5s} cov  low  high  width" for m in METHODS))
        for a, b, tv in targets:
            r = rec[(a, b)]
            label = a if b is None else f"{a} - {b}"
            est = np.array(r["est"])
            cells = []
            row = dict(n=n, target=label, kind="tau" if b is None else "delta", truth=tv, mean_est=est.mean(),
                       stud_dropped_share=r["dropped"] / (args.reps * args.boot))
            for m in METHODS:
                iv = np.array(r[m])
                valid = np.isfinite(iv).all(1)
                lo, hi = iv[valid, 0], iv[valid, 1]
                cov = float(((lo - TIE_TOL <= tv) & (tv <= hi + TIE_TOL)).mean())
                miss_low = float((tv < lo - TIE_TOL).mean())      # interval entirely above the truth
                miss_high = float((tv > hi + TIE_TOL).mean())     # interval entirely below the truth
                width = float((hi - lo).mean())
                row.update({f"{m}_cov": cov, f"{m}_above": miss_low, f"{m}_below": miss_high,
                            f"{m}_width": width, f"{m}_valid": int(valid.sum())})
                cells.append(f"{cov:5.3f} {miss_low:5.3f} {miss_high:5.3f} {width:6.3f}")
            rows.append(row)
            print(f"  {label:28s} {tv:7.4f} {est.mean():8.4f} | " + " | ".join(cells))
    drop = max(r["stud_dropped_share"] for r in rows)
    print(f"\n  studentized: largest share of bootstrap replicates dropped for zero jackknife SE: {drop:.4f}")
    print("\n  cov = coverage of the 82-topic target; low = interval entirely above target;"
          " high = interval entirely below target. Monte Carlo SE of a 0.95 coverage at "
          f"{args.reps} reps: {np.sqrt(0.95 * 0.05 / args.reps):.3f}")

    if args.tsv:
        cols = list(rows[0].keys())
        with open(args.tsv, "w") as f:
            f.write(f"# shared_eps={args.shared_eps} reps={args.reps} boot={args.boot} seed={args.seed}\n" + "\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
