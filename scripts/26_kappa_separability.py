"""Step 26. Are the LLMJudge judges separable by label-level agreement (Cohen's kappa) rather than tau?

Kappa uses all 4,423 judged pairs instead of 465 system-pair orderings. Labels are clipped to 0-3, as
for the published kappa values. Topics are resampled with replacement (B replicates) and kappa is
recomputed from the summed per-topic confusion matrices, so the test respects topic clustering.
Bootstrap z-tests (null: equal kappa) for the adjacent pairs of the kappa leaderboard and for the top
kappa judge against every other judge (Holm), and, for comparison, for the adjacent pairs of the tau
leaderboard.

Usage: python scripts/26_kappa_separability.py DATA_DIR [--boot 10000] [--seed 0] [--tsv FILE]
"""
import argparse
import os

import numpy as np
from scipy.stats import norm

from judge_significance import data as D
from judge_significance.permutation import holm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")


def kappa_from_confusion(C):
    """C: [..., 4, 4] confusion counts -> unweighted kappa [...]."""
    n = C.sum((-1, -2))
    po = np.trace(C, axis1=-2, axis2=-1) / n
    pe = (C.sum(-2) * C.sum(-1)).sum(-1) / n ** 2
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tsv")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    human, topics = data["human"], data["test_topics"]
    names = sorted(data["judges"])
    T = len(topics)
    Ct = {}
    for j in names:
        jq = D.relabel(data["judges"][j], "clip03")
        M = np.zeros((T, 4, 4))
        for t, q in enumerate(topics):
            for x, h in human[q].items():
                M[t, int(h), int(jq[q][x])] += 1
        Ct[j] = M
    kap = {j: float(kappa_from_confusion(Ct[j].sum(0))) for j in names}
    exact = sum(round(kap[j], 4) == pub[j]["kappa"] for j in names)
    rng = np.random.default_rng(args.seed)
    W = np.zeros((args.boot, T))
    np.add.at(W, (np.repeat(np.arange(args.boot), T), rng.integers(0, T, size=(args.boot, T)).ravel()), 1.0)
    kb = {j: kappa_from_confusion(np.einsum("bt,tij->bij", W, Ct[j])) for j in names}

    def ztest(a, b):
        d = kap[a] - kap[b]
        se = float(np.std(kb[a] - kb[b], ddof=1))
        return d, (1.0 if se <= 1e-12 else float(2 * norm.sf(abs(d) / se)))

    korder = sorted(names, key=lambda j: (-kap[j], j))
    torder = sorted(names, key=lambda j: (-pub[j]["tau"], j))
    rows = []
    for kind, pairs in (("kappa_adjacent", list(zip(korder, korder[1:]))),
                        ("kappa_top", [(korder[0], b) for b in korder[1:]]),
                        ("tau_adjacent", list(zip(torder, torder[1:]))),
                        ("kappa_top_tau", [(torder[0], b) for b in torder[1:]])):
        for a, b in pairs:
            d, p = ztest(a, b)
            rows.append(dict(kind=kind, a=a, b=b, delta=d, p=p))
    top = [r for r in rows if r["kind"] == "kappa_top"]
    for r, h_ in zip(top, holm([r["p"] for r in top])):
        r["p_holm"] = h_
    for kind in ("kappa_adjacent", "tau_adjacent", "kappa_top_tau"):
        sub = [r for r in rows if r["kind"] == kind]
        ph = holm([r["p"] for r in sub])
        for r, h_ in zip(sub, ph):
            r["p_holm"] = h_
    print(f"{len(names)} judges, {T} topics, {sum(len(v) for v in human.values())} judged pairs; kappa matches published for "
          f"{exact}/{len(names)}; B={args.boot} seed={args.seed}")
    print(f"  kappa range {min(kap.values()):.3f}-{max(kap.values()):.3f}; median SE of a kappa difference "
          f"{np.median([np.std(kb[a] - kb[b], ddof=1) for a, b in zip(korder, korder[1:])]):.4f}")
    for kind, label in (("kappa_adjacent", "adjacent pairs of the kappa leaderboard"),
                        ("kappa_top", f"top kappa judge ({korder[0]}) vs others"),
                        ("tau_adjacent", "adjacent pairs of the tau leaderboard, compared by kappa"),
                        ("kappa_top_tau", f"top tau judge ({torder[0]}) vs others, compared by kappa")):
        sub = [r for r in rows if r["kind"] == kind]
        print(f"  {label}: separable at p < 0.05 {sum(r['p'] < 0.05 for r in sub)}/{len(sub)}, "
              f"after Holm {sum(r['p_holm'] < 0.05 for r in sub)}/{len(sub)}")
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# B={args.boot} seed={args.seed} kappa_exact={exact} top_kappa={korder[0]}\n"
                    "kind\ta\tb\tdelta\tp\tp_holm\n")
            for r in rows:
                f.write(f"{r['kind']}\t{r['a']}\t{r['b']}\t{r['delta']:.6f}\t{r['p']:.6f}\t{r['p_holm']:.6f}\n")


if __name__ == "__main__":
    main()
