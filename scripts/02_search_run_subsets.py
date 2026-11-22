"""Step 2. Which subset of the 35 DL-2023 runs did LLMJudge use?

Exhaustive search over excluded-run sets of size k <= kmax, using closed-form tau for every
subset (judge_significance.subsets).

  Stage A  the two judge-free Figure 1 targets (Kendall between nDCG@10 on the LLMJudge sampled
           human qrels and on the full NIST qrels: dev 0.9183, test 0.9473)
  Stage B  the 33 public Table 3 taus, over all subsets, for several metric/truth cells
           (labels clipped to [0, 3] here; step 3 repeats the check with raw labels)

Usage: python scripts/02_search_run_subsets.py DATA_DIR [--kmax 4]
"""
import argparse
import itertools
import os
from math import comb

import numpy as np

from judge_significance import correlation as C
from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.subsets import SubsetTau

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")


def r4(x):
    return np.round(x, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--kmax", type=int, default=4)
    args = ap.parse_args()

    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    systems = sorted(data["runs"])
    runs, test, full = data["runs"], data["test_topics"], data["full"]
    n = len(systems)
    dev = D.load_dev_qrels(args.data_dir, data)
    all82 = sorted(full)

    Hm = E.score_matrix(runs, systems, data["human"], test, "ndcg_cut_10").mean(1)
    print("runs tied on mean nDCG@10 (LLMJudge human, 25 test topics):")
    for i, j in itertools.combinations(range(n), 2):
        if Hm[i] == Hm[j]:
            print(f"  {systems[i]} == {systems[j]} ({Hm[i]:.6f})")

    # ---------------- Stage A: Figure 1 ----------------
    print("\n== Stage A: Figure 1 targets (dev 0.9183, test 0.9473), nDCG@10 ==")
    fig = {}
    for judged in (False, True):
        for xname, xt in (("full82", all82), ("fullsame", None)):
            pair = {}
            for name, sample, topics in (("dev", dev, sorted(dev)), ("test", data["human"], test)):
                y = E.score_matrix(runs, systems, sample, topics, "ndcg_cut_10", 1, judged).mean(1)
                x = E.score_matrix(runs, systems, full, xt or topics, "ndcg_cut_10", 1, judged).mean(1)
                pair[name] = (x, y)
            fig[(judged, xname)] = pair
    survivors = {}
    for k in range(args.kmax + 1):
        st = SubsetTau(n, k)
        for (judged, xname), pair in fig.items():
            for v in ("a", "b"):
                res = {}
                for name in ("dev", "test"):
                    ta, tb = st.taus(*pair[name])
                    res[name] = ta if v == "a" else tb
                hit = (np.abs(r4(res["dev"]) - 0.9183) < 5e-5) & (np.abs(r4(res["test"]) - 0.9473) < 5e-5)
                hits = np.nonzero(hit)[0]
                if len(hits):
                    survivors[(k, judged, xname, v)] = st.excluded[hits]
                    print(f"  k={k} judged_only={judged} x={xname} tau-{v}: {len(hits)} of {comb(n, k)} "
                          f"subsets match both")
                    for e in st.excluded[hits][:10]:
                        print("     exclude:", ", ".join(systems[i] for i in e))
    if not survivors:
        print("  no subset with k <=", args.kmax, "matches both Figure 1 values")

    # ---------------- Stage B: Table 3 ----------------
    print("\n== Stage B: Table 3 taus (33 judges), all subsets with k <= kmax ==")
    judges = sorted(data["judges"])
    target = np.array([pub[j]["tau"] for j in judges])
    jq = {j: D.relabel(data["judges"][j], "clip03") for j in judges}
    configs = []
    for metric, lvl in (("ndcg_cut_10", 1), ("ndcg_cut_20", 1), ("ndcg_cut_5", 1), ("ndcg_cut_100", 1),
                        ("map", 2), ("ndcg", 1)):
        for judged in (False, True):
            J = np.array([E.score_matrix(runs, systems, jq[j], test, metric, lvl, judged).mean(1) for j in judges])
            for tname, (tq, tt) in (("llmjudge25", (data["human"], test)), ("full25", (full, test)),
                                    ("full82", (full, all82))):
                H = E.score_matrix(runs, systems, tq, tt, metric, lvl, judged).mean(1)
                configs.append(((metric, judged, tname), H, J))
    best_overall = []
    for k in range(args.kmax + 1):
        st = SubsetTau(n, k)
        for (cfg, H, J) in configs:
            for v in ("a", "b"):
                T = np.stack([st.taus(H, J[i])[0 if v == "a" else 1] for i in range(len(judges))], 1)
                err = r4(T) - target[None, :]
                exact = (np.abs(err) < 5e-5).sum(1)
                w01 = (np.abs(err) <= 0.01 + 1e-9).sum(1)
                b = int(np.lexsort((-w01, -exact))[0])
                best_overall.append((int(exact[b]), int(w01[b]), k, cfg, v, st.excluded[b],
                                     int((exact == exact[b]).sum())))
        print(f"  k={k} done", flush=True)
    best_overall.sort(key=lambda r: (-r[0], -r[1]))
    print("\n  top 15 (exact = |reproduced(4dp) - published| < 5e-5; w01 = within +-0.01; "
          "n_tied = subsets with the same exact count in that cell)")
    for ex, w, k, cfg, v, e, ntied in best_overall[:15]:
        print(f"  exact={ex:2d} w01={w:2d} k={k} {cfg} tau-{v} n_tied={ntied} exclude: "
              + ", ".join(systems[i] for i in e))

    if survivors:
        print("\n== Stage A survivors scored on Table 3 ==")
        for key, Es in survivors.items():
            for (cfg, H, J) in configs:
                for v in ("a", "b"):
                    for e in Es:
                        keep = [i for i in range(n) if i not in set(e)]
                        T = np.array([C.kendall(H[keep], J[i][keep], v) for i in range(len(judges))])
                        err = r4(T) - target
                        ex = int((np.abs(err) < 5e-5).sum())
                        if ex >= 20:
                            print(f"  figkey={key} table cfg={cfg} tau-{v} exact={ex} "
                                  f"w01={int((np.abs(err) <= 0.01 + 1e-9).sum())} exclude: "
                                  + ", ".join(systems[i] for i in e))


if __name__ == "__main__":
    main()
