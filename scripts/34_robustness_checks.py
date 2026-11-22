"""Step 34. Robustness checks on saved outputs (no new simulation; changes no reported number).

  A. Held-out reversals pair by pair: which of the judge pairs reversed between the 25 LLMJudge topics and the
     held-out DL 2023 topics stay reversed when the 25 topics are scored on all NIST-judged pairs
     (results/llm-judges/29_power_bootstrap.tsv, the taus step 30 uses).
  B. Calibration transfer, leave one group out: the 9 exactly tied pairs (near-duplicates excluded) fall into
     4 groups by their shared tau (identical concordance counts). For each group, the worst-pair cutoff of
     step 32 is fitted on the other 3 groups' step 16 p-values and applied to the held-out group, and to all
     pairs for the resolution. Monte Carlo SEs treat every held-out sample as independent; the 6 pairs of the
     largest group share 4 judges, so its true SE is larger.
  C. DL 2019's exactly tied pair: bootstrap z-test rejection at each size (step 19). No calibration step
     (16, 23, 32, 33) reads DL data.
  D. Sensitivity of the nominal-alpha resolution to the (share of pairs, power) criterion, 522 pairs.

Rules copied from scripts/32_size_adjusted_resolution.py (critical, resolution, worst-pair cutoff) and
scripts/13_make_numbers.py (resolution rule, near-duplicate exclusion).

Usage: python scripts/34_robustness_checks.py [--results results]
"""
import argparse
import csv
import itertools
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES_B = (25, 100, 200, 400)


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader([line for line in fh if not line.startswith("#")], delimiter="\t"))


def critical(p_null, alpha):
    """As step 32: the k-th smallest null p-value, k = floor(alpha * N), so share(p < c) <= alpha."""
    s = np.sort(np.asarray(p_null).ravel())
    k = int(np.floor(alpha * len(s)))
    return s[k] if k < len(s) else np.inf


def resolution(gaps, power, share=0.8, pw=0.8):
    """Smallest gap g such that >= share of the pairs at least g apart reach power >= pw."""
    for g in np.unique(gaps):
        tail = gaps >= g - 1e-12
        if (power[tail] >= pw - 1e-12).mean() >= share - 1e-12:
            return g, int(tail.sum())
    return None, 0


def fmt(g, cnt):
    return f"{g:.3f} ({cnt} pairs)" if g is not None else "none"


def part_a(res):
    rows = read_tsv(os.path.join(res, "llm-judges", "29_power_bootstrap.tsv"))
    tau = {}
    for r in rows:
        tau.setdefault(r["pool"], {})[r["judge_a"]] = float(r["tau_a"])
        tau[r["pool"]][r["judge_b"]] = float(r["tau_b"])
    judges = sorted(tau["test"])
    pairs = list(itertools.combinations(judges, 2))

    def reversed_(pool):
        return {(p, q) for p, q in pairs if (tau[pool][p] - tau[pool][q]) * (tau["heldout"][p] - tau["heldout"][q]) < 0}

    r1, r2 = reversed_("test"), reversed_("test-full")
    print(f"A. Held-out reversals ({len(judges)} judges, {len(pairs)} pairs)")
    print(f"   reversed, 25 topics with LLMJudge labels vs held-out: {len(r1)}")
    print(f"   reversed, 25 topics with all NIST labels vs held-out: {len(r2)}")
    print(f"   no longer reversed: {len(r1 - r2)} {sorted(r1 - r2)}")
    print(f"   newly reversed:     {len(r2 - r1)} {sorted(r2 - r1)}")
    print(f"   reversed under both: {len(r1 & r2)}\n")


def part_b(res):
    r2 = os.path.join(res, "g-j2")
    dup = {frozenset((r["judge_a"], r["judge_b"])) for r in read_tsv(os.path.join(r2, "18_label_overlap.tsv"))
           if float(r["label_diff_share"]) < 0.01}
    z = np.load(os.path.join(r2, "16_real_judge_power_bootstrap_pvalues.npz"))
    A, B, alpha, sizes = z["judge_a"], z["judge_b"], float(z["alpha"]), [int(x) for x in z["sizes"]]
    keep = np.array([frozenset((a, b)) not in dup for a, b in zip(A, B)])
    p, gaps, A, B = z["p"][keep], z["gaps"][keep], A[keep], B[keep]
    null = np.flatnonzero(gaps < 1e-9)
    reps = p.shape[2]
    tau = {}
    for r in read_tsv(os.path.join(r2, "16_real_judge_power_bootstrap.tsv")):
        tau[r["judge_a"]], tau[r["judge_b"]] = float(r["tau_a"]), float(r["tau_b"])
    groups = defaultdict(list)
    for i in null:
        groups[round(tau[A[i]], 9)].append(i)
    print(f"B. Calibration transfer, leave one group out ({len(p)} pairs, {len(null)} tied pairs in {len(groups)} groups, "
          f"{reps} replicates per pair and size, alpha {alpha})")
    for t, ids in sorted(groups.items(), reverse=True):
        print(f"   group tau {t:.6f}: " + "; ".join(f"{A[i]}/{B[i]}" for i in ids))
    print("   held-out group (pairs) | topics | cutoff from the other groups | held-out rejection (pooled) | MC SE | worst pair")
    fold = []
    for t, ids in sorted(groups.items(), reverse=True):
        train = [i for i in null if i not in ids]
        for k, n in enumerate(sizes):
            if n not in SIZES_B:
                continue
            c = min(critical(p[i, k], alpha) for i in train)
            rej = np.array([(p[i, k] < c).mean() for i in ids])
            pooled = rej.mean()
            se = np.sqrt(pooled * (1 - pooled) / (reps * len(ids)))
            worst = ids[int(np.argmax(rej))]
            nominal = (p[worst, k] < alpha).mean()
            print(f"   {t:.4f} ({len(ids)}) | {n:3d} | {c:.6f} | {pooled:.4f} | {se:.4f} | "
                  f"{rej.max():.4f} ({A[worst]}/{B[worst]}; {nominal:.4f} at nominal alpha)")
            if n in (200, 400):
                fold.append((t, n, c) + resolution(gaps, (p[:, k, :] < c).mean(1)))
    print("   resolution with each fold's cutoff applied to all pairs:")
    for t, n, c, g, cnt in fold:
        print(f"     held out {t:.4f}, {n} topics: cutoff {c:.6f}, resolution {fmt(g, cnt)}")
    for k, n in enumerate(sizes):
        if n in (200, 400):
            c = min(critical(p[i, k], alpha) for i in null)
            print(f"     all 4 groups (step 32 worst-pair), {n} topics: cutoff {c:.6f}, resolution "
                  f"{fmt(*resolution(gaps, (p[:, k, :] < c).mean(1)))}")
    print()


def part_c(res):
    rows = read_tsv(os.path.join(res, "g-j3", "19_dl19_power.tsv"))
    tied = [r for r in rows if float(r["abs_delta"]) < 1e-9]
    print(f"C. DL 2019 exactly tied pairs: {len(tied)} of {len(rows)} (step 19; 200 samples per size; "
          "sizes above 43 extrapolate DL 2019's 43 topics)")
    for r in tied:
        print(f"   {r['judge_a']} / {r['judge_b']} (labels differ on {float(r['label_disagreement']):.1%}, "
              f"near_duplicate={r['near_duplicate']})")
        for n in (25, 50, 100, 200, 400):
            x = float(r[f"boot_{n}"])
            print(f"     {n:3d} topics: bootstrap z-test rejects {x:.3f} (MC SE {np.sqrt(x * (1 - x) / 200):.3f}); "
                  f"permutation {float(r[f'perm_{n}']):.3f}")
    print()


def part_d(res):
    r2 = os.path.join(res, "g-j2")
    dup = {frozenset((r["judge_a"], r["judge_b"])) for r in read_tsv(os.path.join(r2, "18_label_overlap.tsv"))
           if float(r["label_diff_share"]) < 0.01}
    rows = [r for r in read_tsv(os.path.join(r2, "16_real_judge_power_bootstrap.tsv"))
            if frozenset((r["judge_a"], r["judge_b"])) not in dup]
    gaps = np.array([float(r["abs_delta"]) for r in rows])
    print(f"D. Nominal-alpha resolution by criterion (bootstrap z-test, {len(rows)} pairs; the paper uses 80%, 80%)")
    for share, pw in ((0.8, 0.8), (0.7, 0.8), (0.8, 0.9), (0.9, 0.8)):
        cells = []
        for n in (200, 400):
            power = np.array([float(r[f"power_{n}"]) for r in rows])
            cells.append(f"{n} topics {fmt(*resolution(gaps, power, share, pw))}")
        print(f"   {round(share * 100)}% of pairs at {round(pw * 100)}% power: " + "; ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    part_a(args.results)
    part_b(args.results)
    part_c(args.results)
    part_d(args.results)


if __name__ == "__main__":
    main()
