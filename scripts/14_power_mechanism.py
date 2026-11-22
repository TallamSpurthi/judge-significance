"""Step 14. Why power depends on more than delta tau: what do synthetic judge pairs disagree about?

For each synthetic judge pair in the power analysis (10_power.tsv), list the system pairs that the
two judges relate differently to the human ordering on all 82 topics ("contested" pairs, as in
step 11), and report their human nDCG@10 gaps. Compare with the standard error of a system-pair
mean difference at 25 and 400 topics. Pairs whose rejection rate stays flat should contest only
near-tied systems.

Usage: python scripts/14_power_mechanism.py DATA_DIR [--power results/g-j2/10_power.tsv] [--tsv FILE]
"""
import argparse
import csv
import itertools
import os

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.protocol import table3_systems
from judge_significance.simulation import synthetic_judge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS_GRID = tuple(round(0.04 * i, 2) for i in range(1, 16))  # must match 10_power_curve.py


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--power", default=os.path.join(ROOT, "results", "g-j2", "10_power.tsv"))
    ap.add_argument("--seed", type=int, default=0, help="must match the seed of the power run")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    runs, full = data["runs"], data["full"]
    S = table3_systems(runs)
    topics = sorted(full)
    H = E.score_matrix(runs, S, full, topics, "ndcg_cut_10")
    rng = np.random.default_rng(args.seed)  # same draw order as 10_power_curve.py without --shared-eps
    J = {f"random-{eps:.2f}": E.score_matrix(runs, S, synthetic_judge(full, "random", eps, rng), topics,
                                              "ndcg_cut_10").mean(1) for eps in EPS_GRID}
    h = H.mean(1)
    n = len(S)
    pairs = list(itertools.combinations(range(n), 2))
    sd = np.array([np.std(H[i] - H[j], ddof=1) for i, j in pairs])
    gaps = np.array([abs(h[i] - h[j]) for i, j in pairs])
    print(f"{n} runs, {len(topics)} topics; {len(pairs)} system pairs")
    print(f"median per-topic SD of a system-pair difference: {np.median(sd):.3f}; "
          f"median SE at 25 topics {np.median(sd) / 5:.4f}, at 400 topics {np.median(sd) / 20:.4f}")
    print(f"system pairs with 82-topic human gap < 0.005: {(gaps < 0.005).sum()}, < 0.01: {(gaps < 0.01).sum()}")

    with open(args.power) as f:
        rows = list(csv.DictReader([l for l in f if not l.startswith("#")], delimiter="\t"))
    out = []
    print(f"\n  {'pair':28s} {'delta82':>8s} {'power@25':>8s} {'power@400':>9s} {'contested':>9s} "
          f"{'med gap':>8s} {'max gap':>8s}")
    seen = []
    for r in rows:
        key = (r["target"], r["judge_a"], r["judge_b"])
        if key in seen:
            continue
        seen.append(key)
        a, b = J[key[1]], J[key[2]]
        con = [(i, j) for i, j in pairs
               if np.sign(a[i] - a[j]) * np.sign(h[i] - h[j]) != np.sign(b[i] - b[j]) * np.sign(h[i] - h[j])]
        g = np.array([abs(h[i] - h[j]) for i, j in con]) if con else np.array([np.nan])
        p = {int(x["n"]): float(x["power"]) for x in rows if (x["target"], x["judge_a"], x["judge_b"]) == key}
        out.append(dict(target=r["target"], judge_a=key[1], judge_b=key[2], delta82=float(r["delta82"]),
                        power25=p.get(25, np.nan), power400=p.get(400, np.nan), n_contested=len(con),
                        median_gap=float(np.nanmedian(g)), max_gap=float(np.nanmax(g))))
        o = out[-1]
        print(f"  {key[1] + ' vs ' + key[2][7:]:28s} {o['delta82']:+8.4f} {o['power25']:8.3f} {o['power400']:9.3f} "
              f"{o['n_contested']:9d} {o['median_gap']:8.4f} {o['max_gap']:8.4f}")

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# seed={args.seed} median_sd={np.median(sd):.6f}\n" + "\t".join(out[0]) + "\n")
            for o in out:
                f.write("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in o.values()) + "\n")


if __name__ == "__main__":
    main()
