"""Step 6. Paired topic-permutation tests under the exact Table 3 protocol.

Adjacent judge pairs in the published Table 3 order, and the top judge against every other
judge; two-sided permutation p-values with Holm correction. See judge_significance.permutation.

Usage: python scripts/06_permutation_tests.py DATA_DIR [--perms 10000] [--seed 0] [--alt-runset | --all-runs]
"""
import argparse
import os

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.correlation import taub_rows
from judge_significance.permutation import holm, perm_test
from judge_significance.protocol import ALT_EXCLUDED_RUNS, EXCLUDED_RUNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alt-runset", action="store_true",
                    help="use the other 4-run exclusion that also matches Table 3 (but not Figure 1)")
    ap.add_argument("--all-runs", action="store_true",
                    help="use all 35 runs (sensitivity check; Table 3 taus are then not reproduced)")
    ap.add_argument("--tsv", help="also write every comparison to this TSV file")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    runs, test = data["runs"], data["test_topics"]
    excl = () if args.all_runs else (ALT_EXCLUDED_RUNS if args.alt_runset else EXCLUDED_RUNS)
    systems = sorted(s for s in runs if s not in excl)
    h = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10").mean(1)
    Js = {j: E.score_matrix(runs, systems, qr, test, "ndcg_cut_10") for j, qr in data["judges"].items()}
    tau = {j: taub_rows(h, M.mean(1)[None])[0] for j, M in Js.items()}
    mism = [j for j in tau if abs(round(tau[j], 4) - pub[j]["tau"]) > 5e-5]
    print(f"runs={len(systems)} (excluded: {', '.join(excl) or 'none'}) topics={len(test)} judges={len(Js)} "
          f"P={args.perms} seed={args.seed}")
    print(f"tau check vs Table 3: {len(Js) - len(mism)}/{len(Js)} exact at 4 dp"
          + (f"; mismatches {mism}" if mism else ""))

    # adjacent pairs in the published ordering (ties broken by name)
    order = sorted(Js, key=lambda j: (-pub[j]["tau"], j))
    rng = np.random.default_rng(args.seed)
    rows = []
    for a, b in zip(order, order[1:]):
        d, p = perm_test(h, Js[a], Js[b], args.perms, rng)
        rows.append((a, b, d, p))
    padj = holm(np.array([r[3] for r in rows]))
    print("\nadjacent pairs (published order): delta tau_b, permutation p, Holm-adjusted p")
    for (a, b, d, p), pa in zip(rows, padj):
        print(f"  {a:24s} vs {b:24s} d={d:+.4f} p={p:.4f} holm={pa:.4f}")
    n_unc = sum(r[3] < 0.05 for r in rows)
    print(f"\nadjacent pairs separable at p<0.05 (uncorrected): {n_unc}/{len(rows)}; "
          f"after Holm: {int((padj < 0.05).sum())}/{len(rows)}")
    print(f"adjacent gaps: min {min(r[2] for r in rows):.4f}, max {max(r[2] for r in rows):.4f}; "
          f"zero gaps: {sum(abs(r[2]) < 1e-12 for r in rows)}")

    top = order[0]
    rows2 = []
    for b in order[1:]:
        d, p = perm_test(h, Js[top], Js[b], args.perms, rng)
        rows2.append((b, d, p))
    padj2 = holm(np.array([r[2] for r in rows2]))
    print(f"\ntop judge ({top}) vs each other judge:")
    for (b, d, p), pa in zip(rows2, padj2):
        print(f"  vs {b:24s} d={d:+.4f} p={p:.4f} holm={pa:.4f}")
    sig = [(b, d) for b, d, p in rows2 if p < 0.05]
    print(f"\ntop vs others separable at p<0.05 (uncorrected): {len(sig)}/{len(rows2)}"
          + (f", smallest such delta {min(d for _, d in sig):.4f}" if sig else "")
          + f"; after Holm: {int((padj2 < 0.05).sum())}/{len(rows2)}")

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# runs={len(systems)} excluded={','.join(excl)} topics={len(test)} "
                    f"perms={args.perms} seed={args.seed}\n")
            f.write("comparison\tjudge_a\tjudge_b\ttau_a\ttau_b\tdelta\tp\tp_holm\n")
            for (a, b, d, p), pa in zip(rows, padj):
                f.write(f"adjacent\t{a}\t{b}\t{tau[a]:.6f}\t{tau[b]:.6f}\t{d:.6f}\t{p:.6f}\t{pa:.6f}\n")
            for (b, d, p), pa in zip(rows2, padj2):
                f.write(f"top_vs_other\t{top}\t{b}\t{tau[top]:.6f}\t{tau[b]:.6f}\t{d:.6f}\t{p:.6f}\t{pa:.6f}\n")


if __name__ == "__main__":
    main()
