"""Step 5. Figure 1 of the paper under the 31-run protocol.

Kendall tau between nDCG@10 on the LLMJudge sampled human qrels and on the full NIST qrels
(all 82 topics). Published: dev 0.9183, test 0.9473.

Usage: python scripts/05_check_figure1.py DATA_DIR
"""
import argparse

from judge_significance import correlation as C
from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.protocol import table3_systems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    args = ap.parse_args()

    d = D.load(args.data_dir)
    runs = d["runs"]
    S = table3_systems(runs)
    dev = D.load_dev_qrels(args.data_dir, d)
    full = d["full"]
    all82 = sorted(full)
    for name, qr, tp, pub in (("dev", dev, sorted(dev), 0.9183), ("test", d["human"], d["test_topics"], 0.9473)):
        y = E.score_matrix(runs, S, qr, tp, "ndcg_cut_10").mean(1)
        x = E.score_matrix(runs, S, full, all82, "ndcg_cut_10").mean(1)
        print(f"Figure 1 {name}: 31 runs, nDCG@10, sample vs full qrels (82 topics): "
              f"tau_b={C.kendall(x, y, 'b'):.4f} tau_a={C.kendall(x, y, 'a'):.4f} published {pub}")


if __name__ == "__main__":
    main()
