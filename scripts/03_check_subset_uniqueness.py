"""Step 3. How many excluded-run sets reproduce all 33 Table 3 taus under the final protocol?

Final protocol (raw labels, nDCG@10 trec_eval defaults, LLMJudge human qrels, tau-b); searches
k = 0..kmax and reports the best exact count per k. Also checks Spearman rho on the 31-run set.

Usage: python scripts/03_check_subset_uniqueness.py DATA_DIR [--kmax 5]
"""
import argparse
import os

import numpy as np

from judge_significance import correlation as C
from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.protocol import table3_systems
from judge_significance.subsets import SubsetTau

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--kmax", type=int, default=5)
    args = ap.parse_args()

    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    runs, test = data["runs"], data["test_topics"]
    systems = sorted(runs)
    judges = sorted(data["judges"])
    H = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10").mean(1)
    J = {j: E.score_matrix(runs, systems, data["judges"][j], test, "ndcg_cut_10").mean(1) for j in judges}
    tgt = np.array([pub[j]["tau"] for j in judges])
    tgr = np.array([pub[j]["rho"] for j in judges])

    print("uniqueness: raw labels, nDCG@10, trec_eval defaults, LLMJudge human qrels, tau-b")
    for k in range(args.kmax + 1):
        st = SubsetTau(len(systems), k)
        T = np.stack([st.taus(H, J[j])[1] for j in judges], 1)
        ex = (np.abs(np.round(T, 4) - tgt) < 5e-5).sum(1)
        top = ex.max()
        idx = np.nonzero(ex == top)[0]
        print(f"  k={k}: subsets={len(st.excluded)} max exact={top}/33, subsets reaching it={len(idx)}")
        if top >= 30:
            for i in idx[:5]:
                print("     exclude:", ", ".join(systems[e] for e in st.excluded[i]))

    keep = table3_systems(runs)
    ki = [systems.index(s) for s in keep]
    rho = np.array([C.spearman(H[ki], J[j][ki]) for j in judges])
    print("rho raw labels, 31 runs: exact", int((np.abs(np.round(rho, 4) - tgr) < 5e-5).sum()), "/33")


if __name__ == "__main__":
    main()
