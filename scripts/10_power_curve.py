"""Step 10. Topic-set size: power of the paired permutation test to separate two judges.

Population: the 82 DL-2023 topics with full NIST qrels and the 31 Table 3 runs, used as an
empirical topic distribution (topics drawn with replacement, so sizes above 82 are allowed and
power does not inflate as n approaches 82). Synthetic judges use the 'random' noise model over a
grid of noise levels; for each target gap the script takes the PAIRS_PER_TARGET pairs of judges
with 82-topic tau above MIN_TAU whose delta tau is closest to the target. Several pairs per target
are reported because the test responds to per-topic differences between judges, not to delta tau
alone: target 0 uses pairs with (near-)equal 82-topic tau but different noise levels. Power = share of replicates in which
the two-sided paired permutation test rejects at alpha = 0.05.

Usage: python scripts/10_power_curve.py DATA_DIR [--reps 300] [--perms 1000]
           [--sizes 10 25 50 100 200 400] [--targets 0 0.01 0.025 0.05 0.10] [--seed 0] [--tsv FILE]
"""
import argparse
import time

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.correlation import kendall
from judge_significance.permutation import bootstrap_z_test, jackknife_t_test, perm_test
from judge_significance.protocol import table3_systems
from judge_significance.simulation import synthetic_judge

EPS_GRID = tuple(round(0.04 * i, 2) for i in range(1, 16))  # 0.04 ... 0.60
MIN_TAU = 0.85  # pairs are chosen among judges in the range of real leaderboards
PAIRS_PER_TARGET = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50, 100, 200, 400])
    ap.add_argument("--targets", type=float, nargs="+", default=[0.0, 0.01, 0.025, 0.05, 0.10])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--without-replacement", action="store_true",
                    help="draw topics without replacement (sizes must not exceed 82); robustness check")
    ap.add_argument("--shared-eps", type=float, default=0.0,
                    help="if > 0, every judge is derived from one shared noisy base judge "
                         "(random noise at this level), so judges share errors")
    ap.add_argument("--test", choices=("permutation", "bootstrap", "jackknife"), default="permutation",
                    help="permutation: null of per-topic exchangeability; bootstrap: z-test of equal tau with bootstrap SE "
                         "(--boot replicates); jackknife: t-test of equal tau with jackknife SE (no power; kept for the record)")
    ap.add_argument("--boot", type=int, default=300)
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
    J, tau = {}, {}
    for eps in EPS_GRID:
        name = f"random-{eps:.2f}"
        J[name] = E.score_matrix(runs, systems, synthetic_judge(base, "random", eps, rng), topics, "ndcg_cut_10")
        tau[name] = kendall(H.mean(1), J[name].mean(1), "b")
    if args.without_replacement and max(args.sizes) > len(topics):
        ap.error(f"--without-replacement needs sizes <= {len(topics)}")
    how = "without" if args.without_replacement else "with"
    print(f"population: {len(systems)} runs, {len(topics)} topics (sampled {how} replacement); "
          f"shared_eps={args.shared_eps} test={args.test} boot={args.boot} reps={args.reps} perms={args.perms} alpha={args.alpha} seed={args.seed}")
    print("synthetic judges, tau on all 82 topics:")
    for j in sorted(tau, key=lambda j: -tau[j]):
        print(f"  {j:12s} {tau[j]:.4f}")
    pairs = []
    names = sorted(tau)
    cand = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if min(tau[a], tau[b]) >= MIN_TAU]
    cand = [(a, b) if tau[a] >= tau[b] else (b, a) for a, b in cand]
    for t in args.targets:
        best = sorted(cand, key=lambda ab: (abs(tau[ab[0]] - tau[ab[1]] - t), -tau[ab[0]]))[:PAIRS_PER_TARGET]
        for a, b in best:
            pairs.append((t, a, b, tau[a] - tau[b]))
            print(f"target {t:.3f}: {a} vs {b}, delta tau82 = {tau[a] - tau[b]:+.4f}")

    rows = []
    print(f"\n  {'target':>6s} {'pair':>27s} {'delta82':>8s} " + " ".join(f"{n:>6d}" for n in args.sizes)
          + "   (rejection rate)")
    for t, a, b, d in pairs:
        powers = []
        for n in args.sizes:
            t0 = time.time()
            rej = 0
            for _ in range(args.reps):
                S = (rng.choice(len(topics), size=n, replace=False) if args.without_replacement
                     else rng.integers(0, len(topics), size=n))
                Hs = H[:, S]
                if args.test == "bootstrap":
                    _, p = bootstrap_z_test(Hs, J[a][:, S], J[b][:, S], args.boot, rng)
                elif args.test == "jackknife":
                    _, p = jackknife_t_test(Hs, J[a][:, S], J[b][:, S])
                else:
                    _, p = perm_test(Hs.mean(1), J[a][:, S], J[b][:, S], args.perms, rng)
                rej += p < args.alpha
            pw = rej / args.reps
            powers.append(pw)
            rows.append(dict(target=t, judge_a=a, judge_b=b, delta82=d, n=n, power=pw,
                             mc_se=float(np.sqrt(pw * (1 - pw) / args.reps)), seconds=time.time() - t0))
        print(f"  {t:6.3f} {a + ' vs ' + b[7:]:>27s} {d:+8.4f} " + " ".join(f"{p:6.3f}" for p in powers),
              flush=True)

    print(f"\n  Monte Carlo SE of a power estimate near 0.5 at {args.reps} reps: {np.sqrt(0.25 / args.reps):.3f}")
    print("  topics needed for 80% power (linear interpolation between sizes):")
    for t, a, b, d in pairs:
        pts = [(r["n"], r["power"]) for r in rows if r["target"] == t and r["judge_a"] == a and r["judge_b"] == b]
        need = None
        for (n0, p0), (n1, p1) in zip(pts, pts[1:]):
            if p0 < 0.8 <= p1:
                need = n0 + (0.8 - p0) * (n1 - n0) / (p1 - p0)
                break
        if pts and pts[0][1] >= 0.8:
            need = pts[0][0]
        print(f"    target {t:.3f} {a} vs {b} (delta82 {d:+.4f}): "
              + (f"about {need:.0f} topics" if need else f"more than {pts[-1][0]} topics"))

    if args.tsv:
        cols = list(rows[0].keys())
        with open(args.tsv, "w") as f:
            f.write(f"# sampling={how}_replacement shared_eps={args.shared_eps} test={args.test} boot={args.boot} reps={args.reps} perms={args.perms} alpha={args.alpha} seed={args.seed}\n"
                    + "\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
