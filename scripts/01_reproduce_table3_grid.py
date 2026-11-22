"""Step 1. Label check, Figure 1 check and protocol grid on all 35 runs.

Reproduces LLMJudge Table 3 Kendall tau (Rahmani et al., arXiv:2502.13908) under a grid of
protocol choices and ranks the cells by agreement with the published values.

  1. Label check: Cohen's kappa per judge vs the human test labels, against Table 3 kappa.
     Checks that each judge file is the row it claims to be, independent of runs.
  2. Figure 1 check (no judges): tau between system rankings under the LLMJudge sampled human
     qrels and the full NIST DL-2023 qrels (paper: dev 0.9183, test 0.9473).
  3. Grid; each cell gives 33 taus compared with Table 3:
       truth    llmjudge25 = LLMJudge human test labels (25 topics, 4,423 pairs)
                full25     = full NIST qrels on the same 25 topics (6,427 pairs)
                full82     = full NIST qrels, all 82 judged DL-2023 topics
       metric   trec_eval measure and binary relevance level; ndcg10_exp is exponential-gain nDCG@10
       judged   False = unjudged docs are non-relevant (trec_eval default); True = trec_eval -J
       labels   raw / clip03 / oob0 handling of out-of-range judge labels
       round4   system means rounded to 4 decimals (trec_eval's printed precision)
       tau      a, b (scipy default), c

Usage: python scripts/01_reproduce_table3_grid.py DATA_DIR [--out results/g-j1/01_table3_grid.tsv]
"""
import argparse
import itertools
import os

import numpy as np

from judge_significance import correlation as C
from judge_significance import data as D
from judge_significance import evaluation as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "resources", "llmjudge_table3.tsv")

METRICS = [  # (label, pytrec_eval measure, relevance_level)
    ("ndcg_cut_10", "ndcg_cut_10", 1),
    ("ndcg_cut_5", "ndcg_cut_5", 1),
    ("ndcg_cut_20", "ndcg_cut_20", 1),
    ("ndcg_cut_100", "ndcg_cut_100", 1),
    ("ndcg", "ndcg", 1),
    ("ndcg10_exp", None, None),
    ("map_l1", "map", 1),
    ("map_l2", "map", 2),
    ("P_10_l1", "P_10", 1),
    ("P_10_l2", "P_10", 2),
    ("rr_l1", "recip_rank", 1),
    ("rr_l2", "recip_rank", 2),
    ("recall_100_l1", "recall_100", 1),
    ("recall_100_l2", "recall_100", 2),
]


def matrix(runs, systems, qrels, topics, metric, judged):
    _, measure, lvl = metric
    if measure is None:
        return E.ndcg10_exp_matrix(runs, systems, qrels, topics, judged)
    return E.score_matrix(runs, systems, qrels, topics, measure, lvl, judged)


def kappa_check(data, pub):
    print("== 1. Cohen's kappa check (label alignment) ==")
    pairs = [(q, d) for q in sorted(data["human"]) for d in sorted(data["human"][q])]
    h = [data["human"][q][d] for q, d in pairs]
    n_ok = 0
    for j, qr in data["judges"].items():
        out = {}
        for mode in ("raw", "clip03", "oob0"):
            jq = D.relabel(qr, mode)
            out[mode] = C.cohen_kappa(h, [jq[q][d] for q, d in pairs])
        ok = abs(round(out["raw"], 4) - pub[j]["kappa"]) < 5e-5
        n_ok += ok
        flag = "" if ok else "   <-- " + " ".join(f"{m}={out[m]:.4f}" for m in out)
        print(f"  {j:26s} pub={pub[j]['kappa']:.4f} raw={out['raw']:.4f}{flag}")
    print(f"  kappa exact (4 dp, raw labels): {n_ok}/{len(data['judges'])}\n")


def figure1_check(data, systems, data_dir):
    print("== 2. Figure 1 check: sampled vs full human qrels, nDCG@10, all 35 runs ==")
    full = data["full"]
    if full is None:
        print("  full qrels missing; skipped\n")
        return
    runs = data["runs"]
    sets = {"test": (data["human"], data["test_topics"], 0.9473)}
    dev = D.load_dev_qrels(data_dir, data)
    if dev is not None:
        sets["dev"] = (dev, sorted(dev), 0.9183)
    all82 = sorted(full)
    for name, (sample, topics, pub) in sets.items():
        for judged in (False, True):
            ys = matrix(runs, systems, sample, topics, METRICS[0], judged).mean(1)
            for xname, xtopics in (("full-same-topics", topics), ("full-82-topics", all82)):
                xs = matrix(runs, systems, full, xtopics, METRICS[0], judged).mean(1)
                taus = {v: C.kendall(xs, ys, v) for v in ("a", "b", "c")}
                print(f"  {name:4s} judged_only={judged!s:5s} x={xname:17s} "
                      + " ".join(f"tau_{v}={t:.4f}" for v, t in taus.items())
                      + f"   (published {pub:.4f})")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "g-j1", "01_table3_grid.tsv"))
    args = ap.parse_args()

    data = D.load(args.data_dir)
    pub = D.read_published(PUBLISHED)
    judges = sorted(data["judges"])
    missing = [j for j in judges if j not in pub]
    assert not missing, missing
    systems = sorted(data["runs"])
    runs = data["runs"]
    test = data["test_topics"]
    print(f"systems={len(systems)} judges={len(judges)} test topics={len(test)}\n")

    kappa_check(data, pub)
    figure1_check(data, systems, args.data_dir)

    truths = {"llmjudge25": (data["human"], test)}
    if data["full"] is not None:
        truths["full25"] = (data["full"], test)
        truths["full82"] = (data["full"], sorted(data["full"]))

    relabelled = {m: {j: D.relabel(data["judges"][j], m) for j in judges} for m in ("clip03", "oob0")}
    changed = {m: [j for j in judges if relabelled[m][j] != data["judges"][j]] for m in relabelled}
    print("judges affected by label handling:", changed, "\n")

    print("== 3. Grid ==")
    target = np.array([pub[j]["tau"] for j in judges])
    rows = []
    for metric in METRICS:
        for judged in (False, True):
            J = {j: matrix(runs, systems, data["judges"][j], test, metric, judged).mean(1) for j in judges}
            Jm = {"raw": J}
            for m in relabelled:
                Jm[m] = dict(J)
                for j in changed[m]:
                    Jm[m][j] = matrix(runs, systems, relabelled[m][j], test, metric, judged).mean(1)
            H = {t: matrix(runs, systems, qr, tp, metric, judged).mean(1) for t, (qr, tp) in truths.items()}
            for t, lab, r4, v in itertools.product(H, Jm, (False, True), ("a", "b", "c")):
                h = np.round(H[t], 4) if r4 else H[t]
                taus = np.array([C.kendall(h, np.round(Jm[lab][j], 4) if r4 else Jm[lab][j], v)
                                 for j in judges])
                err = np.round(taus, 4) - target
                rows.append(dict(truth=t, metric=metric[0], judged=judged, labels=lab, round4=r4, tau=v,
                                 within01=int((np.abs(err) <= 0.01 + 1e-9).sum()),
                                 within005=int((np.abs(err) <= 0.005 + 1e-9).sum()),
                                 exact=int((np.abs(err) < 5e-5).sum()),
                                 mae=float(np.abs(err).mean()), maxerr=float(np.abs(err).max()),
                                 bias=float(err.mean()), taus=taus))
        print(f"  done {metric[0]}", flush=True)

    rows.sort(key=lambda r: (-r["within01"], -r["exact"], r["mae"]))
    cols = ["truth", "metric", "judged", "labels", "round4", "tau", "within01", "within005", "exact",
            "mae", "maxerr", "bias"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {len(rows)} configurations to {os.path.relpath(args.out, ROOT)}")
    print(f"\ntop 20 of {len(rows)} (n judges = {len(judges)}; criterion: within +-0.01)")
    print("  " + " ".join(f"{c:>12s}" for c in cols))
    for r in rows[:20]:
        print("  " + " ".join(f"{r[c]:>12.4f}" if isinstance(r[c], float) else f"{str(r[c]):>12s}"
                              for c in cols))

    best = rows[0]
    print(f"\nbest: {best['truth']} {best['metric']} judged={best['judged']} labels={best['labels']} "
          f"round4={best['round4']} tau-{best['tau']}")
    print(f"  {'judge':26s} {'published':>9s} {'reproduced':>10s} {'diff':>8s}")
    for j, t in sorted(zip(judges, best["taus"]), key=lambda x: -pub[x[0]]["tau"]):
        print(f"  {j:26s} {pub[j]['tau']:9.4f} {t:10.4f} {t - pub[j]['tau']:+8.4f}")

    print("\nbest within01 by dimension value:")
    for c in ["truth", "metric", "judged", "labels", "round4", "tau"]:
        vals = sorted({str(r[c]) for r in rows})
        print(f"  {c}: " + ", ".join(f"{v}={max(r['within01'] for r in rows if str(r[c]) == v)}" for v in vals))


if __name__ == "__main__":
    main()
