"""Step 4. Per-judge published vs reproduced Table 3 tau and rho.

Under the identified protocol (31 runs, nDCG@10 trec_eval defaults, LLMJudge human test qrels,
scipy tau-b / Spearman), for raw, clipped and zeroed out-of-range labels; and on all 35 runs for
contrast.

Usage: python scripts/04_compare_table3.py DATA_DIR
"""
import argparse
import os

import numpy as np

from judge_significance import correlation as C
from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.protocol import table3_systems

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    test, runs = data["test_topics"], data["runs"]
    judges = sorted(data["judges"], key=lambda j: -pub[j]["tau"])
    for label, systems in (("31 runs (protocol)", table3_systems(runs)), ("all 35 runs", sorted(runs))):
        H = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10").mean(1)
        print(f"== {label}: n={len(systems)}")
        print(f"  {'judge':24s} {'tau_pub':>7s} {'raw':>7s} {'clip03':>7s} {'oob0':>7s} {'d(clip)':>8s}"
              f" {'tau_a':>7s} | {'rho_pub':>7s} {'rho':>7s} {'d':>8s}")
        stats = {m: [] for m in ("raw", "clip03", "oob0", "rho")}
        for j in judges:
            t = {}
            for m in ("raw", "clip03", "oob0"):
                Jm = E.score_matrix(runs, systems, D.relabel(data["judges"][j], m), test, "ndcg_cut_10").mean(1)
                t[m] = C.kendall(H, Jm, "b")
                stats[m].append(round(t[m], 4) - pub[j]["tau"])
                if m == "clip03":
                    ta, rho = C.kendall(H, Jm, "a"), C.spearman(H, Jm)
            stats["rho"].append(round(rho, 4) - pub[j]["rho"])
            d = round(t["clip03"], 4) - pub[j]["tau"]
            print(f"  {j:24s} {pub[j]['tau']:7.4f} {t['raw']:7.4f} {t['clip03']:7.4f} {t['oob0']:7.4f} {d:+8.4f}"
                  f" {ta:7.4f} | {pub[j]['rho']:7.4f} {rho:7.4f} {round(rho, 4) - pub[j]['rho']:+8.4f}"
                  + ("" if abs(d) < 5e-5 else "   <-- tau differs"))
        for m, e in stats.items():
            e = np.abs(e)
            label = "rho on clipped labels" if m == "rho" else m  # rho is computed on the clip03 labels (above)
            print(f"  {label:7s}: exact(4dp) {int((e < 5e-5).sum())}/{len(e)}, within 0.01 {int((e <= 0.01 + 1e-9).sum())}/{len(e)},"
                  f" max |diff| {e.max():.4f}")
        print()


if __name__ == "__main__":
    main()
