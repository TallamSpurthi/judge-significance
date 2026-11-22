"""Step 11. Which system pairs do LLMJudge judges disagree about, and are those systems separable?

For two judges A and B under the exact Table 3 protocol, a system pair is "contested" when A and
B relate it differently to the human ordering (one agrees with the human order and the other does
not, or one ties it). The numerator of the difference in tau_b comes only from contested pairs; the
denominators can also differ when a judge ties two systems that the human labels tie (here the two
identical runs), which shifts tau_b by about 0.001 without any contested pair.
For each contested pair the script reports the human mean nDCG@10 gap on the 25 test topics and
the paired t statistic t = gap / SE, where SE is the standard error of the paired per-topic
difference of the two systems (SD / sqrt(25)). |t| below the two-sided 5% critical value of t with
24 degrees of freedom (2.064) means the human labels do not significantly order the two systems
(uncorrected). Counts are (judge pair, system pair) instances; the number of distinct system pairs
among them is reported as well.

Usage: python scripts/11_disagreement_analysis.py DATA_DIR [--tsv FILE]
"""
import argparse
import itertools
import os

import numpy as np
from scipy.stats import t as t_dist

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.protocol import table3_systems

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")


def contested(h, a, b, pairs):
    out = []
    for i, j in pairs:
        sh = np.sign(h[i] - h[j])
        ca, cb = np.sign(a[i] - a[j]) * sh, np.sign(b[i] - b[j]) * sh
        if ca != cb:
            out.append((i, j))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    runs, test = data["runs"], data["test_topics"]
    systems = table3_systems(runs)
    n_sys, T = len(systems), len(test)
    Hpt = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10")
    h = Hpt.mean(1)
    J = {j: E.score_matrix(runs, systems, qr, test, "ndcg_cut_10").mean(1) for j, qr in data["judges"].items()}
    order = sorted(J, key=lambda j: (-pub[j]["tau"], j))
    pairs = list(itertools.combinations(range(n_sys), 2))
    z = {}
    for i, j in pairs:
        d = Hpt[i] - Hpt[j]
        se = d.std(ddof=1) / np.sqrt(T)
        z[(i, j)] = abs(d.mean()) / se if se > 0 else (0.0 if d.mean() == 0 else np.inf)
    tc = float(t_dist.ppf(0.975, T - 1))
    allz = np.array([v for v in z.values() if np.isfinite(v)])
    print(f"runs={n_sys} topics={T}; all {len(pairs)} system pairs under human qrels: "
          f"|t| < {tc:.3f} for {(allz < tc).sum()} ({(allz < tc).mean():.0%}), median |t| {np.median(allz):.2f}")

    rows = []
    distinct = {"adjacent": set(), "top_vs_other": set()}

    def report(kind, a, b):
        c = contested(h, J[a], J[b], pairs)
        distinct[kind].update(c)
        if not c:
            rows.append((kind, a, b, 0, np.nan, np.nan, 0))
            return 0, 0
        zs = np.array([z[p] for p in c])
        gaps = np.array([abs(h[i] - h[j]) for i, j in c])
        n_res = int((zs >= tc).sum())
        rows.append((kind, a, b, len(c), float(np.median(gaps)), float(np.median(zs)), n_res))
        return len(c), n_res

    print("\n== Adjacent pairs (published order) ==")
    print(f"  {'pair':52s} {'dtau':>7s} {'contested':>9s} {'med gap':>8s} {'med |t|':>8s} {'sig':>7s}")
    tot_c = tot_r = 0
    for a, b in zip(order, order[1:]):
        c, r = report("adjacent", a, b)
        tot_c += c
        tot_r += r
        row = rows[-1]
        print(f"  {a + ' vs ' + b:52s} {pub[a]['tau'] - pub[b]['tau']:+7.4f} {row[3]:9d} "
              + (f"{row[4]:8.4f} {row[5]:8.2f} {row[6]:7d}" if row[3] else f"{'-':>8s} {'-':>8s} {'-':>7s}"))
    du = distinct["adjacent"]
    du_sig = sum(z[p] >= tc for p in du)
    print(f"\n  adjacent pairs: {tot_c} contested (judge pair, system pair) instances; {tot_r} ({tot_r / max(tot_c, 1):.0%}) "
          f"have |t| >= {tc:.3f} under the human labels")
    print(f"  distinct contested system pairs: {len(du)} of {len(pairs)}; {len(du) - du_sig} "
          f"({(len(du) - du_sig) / max(len(du), 1):.0%}) not significantly ordered by the human labels")
    # matched baseline: system pairs not contested by any adjacent judge pair, matched on the human
    # mean nDCG@10 gap (the 5 nearest by |gap|); share of them not significantly ordered
    gap = {p: abs(h[p[0]] - h[p[1]]) for p in pairs}
    others = [p for p in pairs if p not in du]
    og = np.array([gap[p] for p in others])
    matched_unres = []
    for p in du:
        nn = np.argsort(np.abs(og - gap[p]))[:5]
        matched_unres.append(np.mean([z[others[i]] < tc for i in nn]))
    mexp = float(np.mean(matched_unres)) if matched_unres else float("nan")
    print(f"  matched baseline: non-contested system pairs with the same human gap (5 nearest each) are not "
          f"significantly ordered {mexp:.0%} of the time")

    print("\n== Top judge vs each other judge ==")
    top = order[0]
    tc_ = tr = 0
    for b in order[1:]:
        c, r = report("top_vs_other", top, b)
        tc_ += c
        tr += r
    print(f"  {top} vs the other {len(order) - 1} judges: {tc_} contested (judge pair, system pair) instances; "
          f"{tr} ({tr / max(tc_, 1):.0%}) have |t| >= {tc:.3f}")

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# runs={n_sys} topics={T} t_crit={tc:.4f} distinct_adjacent={len(du)} distinct_adjacent_unres={len(du) - du_sig} matched_unres={mexp:.4f}\n"
                    "kind\tjudge_a\tjudge_b\tn_contested\tmedian_gap\tmedian_abs_t\tn_sig\n")
            for r in rows:
                f.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.6f}" for x in r) + "\n")


if __name__ == "__main__":
    main()
