"""Step 21. Does stratifying the topic bootstrap by query type change the LLMJudge results?

LLMJudge sampled its 25 test topics as 15 of the 51 real DL 2023 queries and 10 of the 31 generated
ones (5 by T5, 5 by GPT-4; Rahmani et al., Section 3). In DL 2023, real queries have ids starting
with 2 and generated ones with 3 (the 82 topics split 51/31 by this rule, as the paper states). A
plain topic bootstrap ignores this design; this step resamples within the two strata (real,
generated) instead and repeats (a) the bootstrap z-test for the adjacent pairs and for the top judge
against every other judge (Holm and BH), and (b) the percentile interval check for each judge.
The permutation test swaps judges within each topic and is unaffected by stratification.

Usage: python scripts/21_stratified_check.py DATA_DIR [--boot 10000] [--seed 0] [--tsv FILE]
"""
import argparse
import os

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.bootstrap import taub_pairs
from judge_significance.permutation import benjamini_hochberg, bootstrap_z_test, holm, stratified_indices
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
    strata = [np.array([i for i, q in enumerate(test) if q.startswith(p)]) for p in ("2", "3")]
    full = sorted(data["full"])
    H = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10")
    J = {j: E.score_matrix(runs, systems, q, test, "ndcg_cut_10") for j, q in data["judges"].items()}
    order = sorted(J, key=lambda j: (-pub[j]["tau"], j))
    rng = np.random.default_rng(args.seed)
    print(f"runs={len(systems)} topics={len(test)}: {len(strata[0])} real (id 2...), {len(strata[1])} generated (id 3...); "
          f"all 82 DL 2023 topics: {sum(q.startswith('2') for q in full)} real, {sum(q.startswith('3') for q in full)} generated; "
          f"B={args.boot} seed={args.seed}")

    rows = []
    for kind, pairs in (("adjacent", list(zip(order, order[1:]))), ("top", [(order[0], b) for b in order[1:]])):
        for a, b in pairs:
            d, p = bootstrap_z_test(H, J[a], J[b], args.boot, rng, strata=strata)
            rows.append(dict(kind=kind, a=a, b=b, delta=d, p=p))
    adj = [r for r in rows if r["kind"] == "adjacent"]
    top = [r for r in rows if r["kind"] == "top"]
    pt = np.array([r["p"] for r in top])
    for r, h_, b_ in zip(top, holm(pt), benjamini_hochberg(pt)):
        r.update(p_holm=h_, p_bh=b_)
    print(f"\n  stratified bootstrap z-test: adjacent separable {sum(r['p'] < 0.05 for r in adj)}/{len(adj)} "
          f"(smallest p among nonzero gaps {min(r['p'] for r in adj if abs(r['delta']) > 1e-9):.3f}); "
          f"top vs others {int((pt < 0.05).sum())}/{len(top)} uncorrected, "
          f"{sum(r['p_holm'] < 0.05 for r in top)} after Holm (smallest {min(r['p_holm'] for r in top):.3f}), "
          f"{sum(r['p_bh'] < 0.05 for r in top)} after BH (smallest {min(r['p_bh'] for r in top):.3f})")

    # percentile intervals for each judge's tau under the stratified bootstrap
    idx = stratified_indices(strata, args.boot, rng)
    hb = H[:, idx].mean(2).T                                      # [B, n_sys]
    above = 0
    lo_hi = {}
    for j in order:
        boot = taub_pairs(hb, J[j][:, idx].mean(2).T)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        lo_hi[j] = (lo, hi)
        above += pub[j]["tau"] > hi + 1e-9
    print(f"  stratified percentile intervals: judges with tau above their own 95% interval: {above}/{len(order)} "
          f"(top judge {order[0]}: tau {pub[order[0]]['tau']:.4f}, interval [{lo_hi[order[0]][0]:.3f}, {lo_hi[order[0]][1]:.3f}])")

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# stratified (real/generated) bootstrap; B={args.boot} seed={args.seed} percentile_above={above}\n"
                    "kind\ta\tb\tdelta\tp\tp_holm\tp_bh\n")
            for r in rows:
                f.write(f"{r['kind']}\t{r['a']}\t{r['b']}\t{r['delta']:.6f}\t{r['p']:.6f}\t"
                        f"{r.get('p_holm', float('nan')):.6f}\t{r.get('p_bh', float('nan')):.6f}\n")


if __name__ == "__main__":
    main()
