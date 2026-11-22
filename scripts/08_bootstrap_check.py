"""Step 8. Topic bootstrap under the exact Table 3 protocol: percentile vs BCa.

Re-checks the pilot observation that the percentile topic bootstrap is biased low for tau
(the top judge's tau lay above its own 95% interval). For every judge: tau, mean of the
bootstrap replicates, bias, share of replicates below tau, 95% percentile and BCa intervals.
For every adjacent pair in the published order: the same for the paired difference in tau.

Usage: python scripts/08_bootstrap_check.py DATA_DIR [--boot 10000] [--seed 0] [--tsv FILE]
"""
import argparse
import os

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.bootstrap import (TIE_TOL, bca_ci, boot_tau, jackknife_tau, percentile_ci, taub_pairs,
                                          resample_indices)
from judge_significance.protocol import table3_systems

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tsv")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    runs, test = data["runs"], data["test_topics"]
    systems = table3_systems(runs)
    H = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10")
    Js = {j: E.score_matrix(runs, systems, qr, test, "ndcg_cut_10") for j, qr in data["judges"].items()}
    order = sorted(Js, key=lambda j: (-pub[j]["tau"], j))
    rng = np.random.default_rng(args.seed)
    idx = resample_indices(len(test), args.boot, rng)  # shared by all judges (paired)

    tau, bt, jk = {}, {}, {}
    for j in order:
        # same implementation as the replicates (boot_tau), so that ties are detected correctly
        tau[j] = float(taub_pairs(H.mean(1)[None], Js[j].mean(1)[None])[0])
        bt[j] = boot_tau(H, Js[j], idx)
        jk[j] = jackknife_tau(H, Js[j])
    n_exact = sum(abs(round(tau[j], 4) - pub[j]["tau"]) < 5e-5 for j in order)
    print(f"runs={len(systems)} topics={len(test)} judges={len(order)} B={args.boot} seed={args.seed}; "
          f"tau check vs Table 3: {n_exact}/{len(order)} exact")

    print("\n== Per-judge tau: 95% percentile and BCa intervals ==")
    print(f"  {'judge':24s} {'tau':>7s} {'mean*':>7s} {'bias':>8s} {'P(*<tau)':>8s} "
          f"{'perc lo':>8s} {'perc hi':>8s} {'BCa lo':>8s} {'BCa hi':>8s} {'z0':>6s} {'a':>7s}")
    rows = []
    for j in order:
        b = bt[j]
        plo, phi = percentile_ci(b)
        blo, bhi, z0, a = bca_ci(tau[j], b, jk[j])
        below = float(((b < tau[j] - TIE_TOL).sum() + 0.5 * (np.abs(b - tau[j]) <= TIE_TOL).sum()) / len(b))
        flag = "  <-- tau above percentile CI" if tau[j] > phi else ""
        rows.append(("judge", j, "", tau[j], b.mean(), b.mean() - tau[j], below, plo, phi, blo, bhi, z0, a))
        print(f"  {j:24s} {tau[j]:7.4f} {b.mean():7.4f} {b.mean() - tau[j]:+8.4f} {below:8.3f} "
              f"{plo:8.4f} {phi:8.4f} {blo:8.4f} {bhi:8.4f} {z0:6.2f} {a:+7.3f}{flag}")
    above = [r for r in rows if r[3] > r[8]]
    bias = np.array([r[5] for r in rows])
    below = np.array([r[6] for r in rows])
    print(f"\n  judges with tau above their own 95% percentile upper bound: {len(above)}/{len(rows)}")
    print(f"  bootstrap bias (mean* - tau): median {np.median(bias):+.4f}, range [{bias.min():+.4f}, {bias.max():+.4f}]")
    print(f"  share of replicates below tau: median {np.median(below):.3f}, range [{below.min():.3f}, {below.max():.3f}]")
    pw = np.array([r[8] - r[7] for r in rows])
    bw = np.array([r[10] - r[9] for r in rows])
    print(f"  interval width: percentile median {np.median(pw):.4f}; BCa median {np.median(bw):.4f}")
    inside_bca = sum(r[9] <= r[3] <= r[10] for r in rows)
    print(f"  judges whose tau lies inside their own BCa interval: {inside_bca}/{len(rows)}")

    print("\n== Adjacent pairs: delta tau = tau(a) - tau(b), paired bootstrap ==")
    print(f"  {'pair':52s} {'delta':>7s} {'perc lo':>8s} {'perc hi':>8s} {'BCa lo':>8s} {'BCa hi':>8s}")
    n_perc_excl = n_bca_excl = 0
    for a_, b_ in zip(order, order[1:]):
        d = tau[a_] - tau[b_]
        bd = bt[a_] - bt[b_]
        jd = jk[a_] - jk[b_]
        plo, phi = percentile_ci(bd)
        if np.all(np.abs(jd - jd[0]) <= TIE_TOL) and np.all(np.abs(bd - bd[0]) <= TIE_TOL):
            blo = bhi = float(bd[0])
            z0 = a = 0.0
        else:
            blo, bhi, z0, a = bca_ci(d, bd, jd)
        pe = not (plo <= 0 <= phi)
        be = not (blo <= 0 <= bhi)
        n_perc_excl += pe
        n_bca_excl += be
        rows.append(("adjacent", a_, b_, d, bd.mean(), bd.mean() - d,
                     float(((bd < d - TIE_TOL).sum() + 0.5 * (np.abs(bd - d) <= TIE_TOL).sum()) / len(bd)),
                     plo, phi, blo, bhi, z0, a))
        print(f"  {a_ + ' vs ' + b_:52s} {d:+7.4f} {plo:+8.4f} {phi:+8.4f} {blo:+8.4f} {bhi:+8.4f}"
              + ("  <-- excludes 0" if pe or be else ""))
    print(f"\n  adjacent-pair intervals excluding 0: percentile {n_perc_excl}/{len(order) - 1}, "
          f"BCa {n_bca_excl}/{len(order) - 1}")

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# runs={len(systems)} topics={len(test)} B={args.boot} seed={args.seed}\n")
            f.write("kind\tjudge_a\tjudge_b\testimate\tboot_mean\tbias\tshare_below\tperc_lo\tperc_hi"
                    "\tbca_lo\tbca_hi\tz0\taccel\n")
            for r in rows:
                f.write("\t".join(r[:3]) + "\t" + "\t".join(f"{v:.6f}" for v in r[3:]) + "\n")


if __name__ == "__main__":
    main()
