"""Step 17. Does the LLMJudge leaderboard follow from the judges' error rates, or from which
documents they get wrong?

Within each of the 25 test topics, every judged pair receives, for all 33 judges at once, the labels
the judges gave to a randomly chosen pair of the SAME topic with the same human label (and, in the
second variant, the same retrieval stratum: how many of the 31 systems retrieve it in their top 10).
This keeps every judge's per-topic confusion with the human labels and the correlation between
judges, and changes only which documents carry the errors. The script reports how well the
re-drawn leaderboard agrees with the published one over many draws.

Usage: python scripts/17_error_placement.py DATA_DIR [--draws 20] [--seed 0] [--tsv FILE]
"""
import argparse
import os

import numpy as np
from scipy import stats

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.correlation import kendall
from judge_significance.protocol import table3_systems
from judge_significance.simulation import retrieval_strata, transplant_judges

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tsv")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    runs, test, human, judges = data["runs"], data["test_topics"], data["human"], data["judges"]
    systems = table3_systems(runs)
    pub = D.read_published(PUBLISHED)
    names = sorted(judges, key=lambda j: (-pub[j]["tau"], j))
    pub_tau = np.array([pub[j]["tau"] for j in names])
    h = E.score_matrix(runs, systems, human, test, "ndcg_cut_10").mean(1)
    strata = retrieval_strata(runs, systems, human)
    rng = np.random.default_rng(args.seed)
    print(f"{len(systems)} runs, {len(test)} topics, {len(names)} judges; {args.draws} draws per variant; seed={args.seed}")

    rows, rows_between = [], []
    for variant in ("label", "label+stratum"):
        ks, top_rank, taus = [], [], []
        for _ in range(args.draws):
            out = {j: {} for j in names}
            fb = 0.0
            for q in test:  # one topic at a time, so the donor topic is the topic itself
                sub = {j: {q: judges[j][q]} for j in names}
                kw = dict(full_strata={q: strata[q]}, human_strata={q: strata[q]}) if variant != "label" else {}
                qr, f = transplant_judges({q: human[q]}, {q: human[q]}, sub, rng, **kw)
                fb += f / len(test)
                for j in names:
                    out[j][q] = qr[j][q]
            t = np.array([kendall(h, E.score_matrix(runs, systems, out[j], test, "ndcg_cut_10").mean(1)) for j in names])
            ks.append(stats.kendalltau(pub_tau, t)[0])
            top_rank.append(int((t > t[0] + 1e-12).sum()) + 1)
            taus.append(t)
            rows.append((variant, ks[-1], top_rank[-1], float(t.min()), float(t.max()), fb))
        ks, top_rank, taus = np.array(ks), np.array(top_rank), np.array(taus)
        print(f"\n== {variant} ==")
        print(f"  Kendall tau between published and re-drawn leaderboard: median {np.median(ks):.3f}, "
              f"range {ks.min():.3f} to {ks.max():.3f}")
        print(f"  rank of the published top judge ({names[0]}) after re-drawing: median {np.median(top_rank):.0f}, "
              f"range {top_rank.min()} to {top_rank.max()}")
        print(f"  re-drawn tau over judges and draws: {taus.min():.3f} to {taus.max():.3f} "
              f"(published {pub_tau.min():.3f} to {pub_tau.max():.3f}); mean change per judge {np.mean(taus - pub_tau):+.3f}")
        between = [stats.kendalltau(taus[i], taus[k])[0] for i in range(len(taus)) for k in range(i + 1, len(taus))]
        print(f"  Kendall tau between two re-drawn leaderboards (baseline): median {np.median(between):.3f}, "
              f"range {min(between):.3f} to {max(between):.3f}")
        rows_between.append((variant, float(np.median(between)), float(min(between)), float(max(between))))
        print(f"  share of pairs matched only on label (no same-stratum donor in the topic): "
              f"{np.mean([r[5] for r in rows if r[0] == variant]):.3f}")

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# draws={args.draws} seed={args.seed}\nvariant\tkendall_pub\ttop_judge_rank\ttau_min\ttau_max\tunmatched_share\n")
            for r in rows:
                f.write(f"{r[0]}\t{r[1]:.6f}\t{r[2]}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\n")
        with open(args.tsv.replace(".tsv", "_between.tsv"), "w") as f:
            f.write("variant\tmedian_kendall_between_draws\tmin\tmax\n")
            for r in rows_between:
                f.write(f"{r[0]}\t{r[1]:.6f}\t{r[2]:.6f}\t{r[3]:.6f}\n")


if __name__ == "__main__":
    main()
