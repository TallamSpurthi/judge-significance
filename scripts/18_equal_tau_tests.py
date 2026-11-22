"""Step 18. Tests whose null is equal tau, on the LLMJudge leaderboard (exact Table 3 protocol).

The permutation test of step 6 tests per-topic exchangeability, which is stronger than equal tau.
This step adds, for the adjacent pairs and for the top judge against every other judge:
  - the bootstrap z-test (H0: equal tau; bootstrap SE from --boot-z replicates), with Holm and
    Benjamini-Hochberg adjustment for the top-vs-others family;
  - the jackknife t-test (H0: equal tau; jackknife SE, t with 24 df), for the record: the jackknife
    variance is not consistent for tau and this test has almost no power (step 10, --test jackknife);
  - 95% studentized bootstrap intervals for delta tau (jackknife SE within each replicate);
  - Benjamini-Hochberg adjustment of the step-6 permutation p-values (top vs others).
It also reports how the published leaderboard depends on trec_eval's single-precision scores:
taus and ranks when documents are ordered by full double-precision scores instead; and, with
--overlap-tsv, the share of human-judged pairs on which each two judges' labels differ (used to
leave near-duplicate judges out of the power summaries).

Usage: python scripts/18_equal_tau_tests.py DATA_DIR [--boot 2000] [--boot-z 10000] [--seed 0]
           [--perm-tsv results/g-j1/06_permutation_seed0.tsv] [--tsv FILE] [--overlap-tsv FILE]
"""
import argparse
import os

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.correlation import kendall
from judge_significance.permutation import benjamini_hochberg, bootstrap_z_test, holm, jackknife_t_test
from judge_significance.protocol import table3_systems
from judge_significance.simulation import intervals

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")


def ndcg10_double(runs, systems, qrels, topics):
    """nDCG@10 with linear gain, documents ordered by full double-precision scores."""
    M = np.zeros((len(systems), len(topics)))
    for t, q in enumerate(topics):
        qr = qrels.get(q, {})
        ideal = sorted((g for g in qr.values() if g > 0), reverse=True)[:10]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        for i, s in enumerate(systems):
            ranked = E.trec_order(runs[s].get(q, {}), single_precision=False)[:10]
            dcg = sum(max(qr.get(x, 0), 0) / np.log2(r + 2) for r, x in enumerate(ranked))
            M[i, t] = dcg / idcg if idcg > 0 else 0.0
    return M


def ranks(values):
    """Competition ranks (1 = highest; ties share the best rank)."""
    v = np.asarray(values)
    return np.array([int((v > x + 1e-12).sum()) + 1 for x in v])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-z", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm-tsv", default=os.path.join(ROOT, "results", "g-j1", "06_permutation_seed0.tsv"))
    ap.add_argument("--tsv")
    ap.add_argument("--overlap-tsv")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    runs, test = data["runs"], data["test_topics"]
    systems = table3_systems(runs)
    H = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10")
    J = {j: E.score_matrix(runs, systems, q, test, "ndcg_cut_10") for j, q in data["judges"].items()}
    order = sorted(J, key=lambda j: (-pub[j]["tau"], j))
    rng = np.random.default_rng(args.seed)
    allt = np.arange(len(test))

    perm_p = {}
    with open(args.perm_tsv) as f:
        for line in f:
            if line.startswith("#") or line.startswith("comparison"):
                continue
            c = line.rstrip("\n").split("\t")
            perm_p[(c[0], c[1], c[2])] = float(c[6])

    rows = []

    def compare(kind, a, b):
        d, p = jackknife_t_test(H, J[a], J[b])
        _, pz = bootstrap_z_test(H, J[a], J[b], args.boot_z, rng)
        iv = intervals(H, J[a], J[b], allt, args.boot, rng)
        lo, hi = iv["studentized"]
        rows.append(dict(kind=kind, a=a, b=b, delta=d, p_jack=p, p_boot=pz, stud_lo=lo, stud_hi=hi,
                         p_perm=perm_p.get(("adjacent" if kind == "adjacent" else "top_vs_other", a, b), np.nan)))

    for a, b in zip(order, order[1:]):
        compare("adjacent", a, b)
    for b in order[1:]:
        compare("top", order[0], b)
    adj = [r for r in rows if r["kind"] == "adjacent"]
    top = [r for r in rows if r["kind"] == "top"]
    pj = np.array([r["p_jack"] for r in top])
    pz = np.array([r["p_boot"] for r in top])
    pp = np.array([r["p_perm"] for r in top])
    for r, h_, b_, bp, zh, zb in zip(top, holm(pj), benjamini_hochberg(pj), benjamini_hochberg(pp),
                                     holm(pz), benjamini_hochberg(pz)):
        r.update(p_jack_holm=h_, p_jack_bh=b_, p_perm_bh=bp, p_boot_holm=zh, p_boot_bh=zb)

    print(f"runs={len(systems)} topics={len(test)} judges={len(order)}; studentized bootstrap B={args.boot}; "
          f"bootstrap z-test B={args.boot_z}; seed={args.seed}")
    if np.isnan(pp).any():
        print(f"  WARNING: {int(np.isnan(pp).sum())} top-vs-other permutation p-values missing from {args.perm_tsv}")

    def excl0(r):
        return np.isfinite(r["stud_lo"]) and (r["stud_lo"] > 0 or r["stud_hi"] < 0)

    print("\n== Adjacent pairs (published order) ==")
    print(f"  {'pair':55s} {'delta':>8s} {'p perm':>7s} {'p boot':>7s} {'p jack':>7s}  studentized 95% CI")
    for r in adj:
        print(f"  {r['a'] + ' vs ' + r['b']:55s} {r['delta']:+8.4f} {r['p_perm']:7.3f} {r['p_boot']:7.3f} {r['p_jack']:7.3f}  "
              f"[{r['stud_lo']:+.4f}, {r['stud_hi']:+.4f}]")
    nz = [r for r in adj if abs(r["delta"]) > 1e-12]
    print(f"\n  adjacent pairs separable at p < 0.05: permutation {sum(r['p_perm'] < 0.05 for r in adj)}/{len(adj)}, "
          f"bootstrap z {sum(r['p_boot'] < 0.05 for r in adj)}/{len(adj)}, "
          f"jackknife t {sum(r['p_jack'] < 0.05 for r in adj)}/{len(adj)}; studentized CI excludes 0: "
          f"{sum(excl0(r) for r in adj)}/{len(adj)} ({len(nz)} pairs with nonzero delta; smallest p among them: "
          f"bootstrap z {min(r['p_boot'] for r in nz):.3f}, jackknife t {min(r['p_jack'] for r in nz):.3f})")

    print("\n== Top judge vs each other judge ==")
    print(f"  {'pair':55s} {'delta':>8s} {'p perm':>7s} {'BH':>6s} {'p boot':>7s} {'Holm':>6s} {'BH':>6s} {'p jack':>7s}  studentized 95% CI")
    for r in top:
        print(f"  {r['a'] + ' vs ' + r['b']:55s} {r['delta']:+8.4f} {r['p_perm']:7.3f} {r['p_perm_bh']:6.3f} "
              f"{r['p_boot']:7.3f} {r['p_boot_holm']:6.3f} {r['p_boot_bh']:6.3f} {r['p_jack']:7.3f}  [{r['stud_lo']:+.4f}, {r['stud_hi']:+.4f}]")
    print(f"\n  permutation: uncorrected {int((pp < 0.05).sum())}/{len(top)}, BH {sum(r['p_perm_bh'] < 0.05 for r in top)}/{len(top)} "
          f"(smallest BH-adjusted p {min(r['p_perm_bh'] for r in top):.3f})")
    sigz = [r for r in top if r["p_boot"] < 0.05]
    print(f"  bootstrap z: uncorrected {len(sigz)}/{len(top)}"
          + (f" (smallest such delta {min(abs(r['delta']) for r in sigz):.4f})" if sigz else "")
          + f", Holm {sum(r['p_boot_holm'] < 0.05 for r in top)}/{len(top)} (smallest {min(r['p_boot_holm'] for r in top):.3f}), "
          f"BH {sum(r['p_boot_bh'] < 0.05 for r in top)}/{len(top)} (smallest {min(r['p_boot_bh'] for r in top):.3f})")
    sig = [r for r in top if r["p_jack"] < 0.05]
    print(f"  jackknife t: uncorrected {len(sig)}/{len(top)}"
          + (f" (smallest such delta {min(abs(r['delta']) for r in sig):.4f})" if sig else "")
          + f", Holm {sum(r['p_jack_holm'] < 0.05 for r in top)}/{len(top)} (smallest {min(r['p_jack_holm'] for r in top):.3f}), "
          f"BH {sum(r['p_jack_bh'] < 0.05 for r in top)}/{len(top)} (smallest {min(r['p_jack_bh'] for r in top):.3f})")
    print(f"  studentized CI excludes 0: {sum(excl0(r) for r in top)}/{len(top)}")

    # single vs double precision ordering of documents (trec_eval stores scores as float32)
    hd = ndcg10_double(runs, systems, data["human"], test).mean(1)
    tau_d = np.array([kendall(hd, ndcg10_double(runs, systems, data["judges"][j], test).mean(1)) for j in order])
    tau_p = np.array([pub[j]["tau"] for j in order])
    changed = np.abs(np.round(tau_d, 4) - tau_p) > 5e-5
    rk_p, rk_d = ranks(tau_p), ranks(np.round(tau_d, 4))
    moved = rk_p != rk_d
    print("\n== Double-precision document ordering (trec_eval uses single precision) ==")
    print(f"  taus that change at 4 decimals: {int(changed.sum())}/{len(order)}; judges whose leaderboard rank changes: "
          f"{int(moved.sum())}/{len(order)} (largest move {int(np.abs(rk_p - rk_d).max())} places)")
    for j, a, b, r0, r1 in zip(order, tau_p, tau_d, rk_p, rk_d):
        if abs(round(b, 4) - a) > 5e-5 or r0 != r1:
            print(f"    {j:28s} published {a:.4f} (rank {r0})  double {b:.4f} (rank {r1})")

    if args.overlap_tsv:
        hp = [(q, x) for q in sorted(data["human"]) for x in sorted(data["human"][q])]
        lab = {j: np.array([data["judges"][j][q][x] for q, x in hp]) for j in order}
        with open(args.overlap_tsv, "w") as f:
            f.write(f"# share of the {len(hp)} human-judged pairs on which the two judges' labels differ\n"
                    "judge_a\tjudge_b\tlabel_diff_share\n")
            near = []
            for i, a in enumerate(order):
                for b in order[i + 1:]:
                    share = float((lab[a] != lab[b]).mean())
                    f.write(f"{a}\t{b}\t{share:.6f}\n")
                    if share < 0.01:
                        near.append(f"{a}/{b} ({share:.4f})")
        print(f"\n  judge pairs whose labels differ on fewer than 1% of pairs: {len(near)}: {', '.join(near)}")

    if args.tsv:
        cols = ["kind", "a", "b", "delta", "p_perm", "p_perm_bh", "p_boot", "p_boot_holm", "p_boot_bh",
                "p_jack", "p_jack_holm", "p_jack_bh", "stud_lo", "stud_hi"]
        with open(args.tsv, "w") as f:
            f.write(f"# runs={len(systems)} topics={len(test)} boot={args.boot} boot_z={args.boot_z} seed={args.seed} "
                    f"double_precision_tau_changed={int(changed.sum())} rank_moved={int(moved.sum())} "
                    f"max_rank_move={int(np.abs(rk_p - rk_d).max())}\n" + "\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) if isinstance(r.get(c, ""), str) else f"{r.get(c, np.nan):.6f}"
                                  for c in cols) + "\n")


if __name__ == "__main__":
    main()
