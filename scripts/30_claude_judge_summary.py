"""Step 30. Summary of the eight new LLM judges (steps 28-29) for the paper, and their labels for release.

Reads the step 29 power table and the label folders of step 28, and writes to OUT:
  qrels/<judge>.txt   each judge's labels for every NIST-judged pair of the 82 TREC DL 2023 topics (ids and labels
                      only, no passage text), merged from the folders in order (a later folder only fills gaps)
  summary.tsv         name, value, meaning: label coverage, tau ranges, agreement of the judge order between the
                      25 LLMJudge topics and the 57 held-out topics, held-out power, and prediction errors
The paper's macros are made from summary.tsv by step 13.

Usage: python scripts/30_claude_judge_summary.py DATA_DIR --power TSV --labels DIR [DIR ...] --out OUT
"""
import argparse
import csv
import itertools
import os

import numpy as np

from judge_significance import data as D
from judge_significance import llm_label as L
from judge_significance.correlation import taub_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--power", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = D.load(args.data_dir)
    full, human, test = data["full"], data["human"], data["test_topics"]
    rows = list(csv.DictReader(open(args.power), delimiter="\t"))
    judges = sorted({r["judge_a"] for r in rows} | {r["judge_b"] for r in rows})
    labels = L.merge_label_dirs(args.labels, D.read_qrels)
    os.makedirs(os.path.join(args.out, "qrels"), exist_ok=True)
    out = []

    def add(name, value, meaning):
        out.append(dict(name=name, value=value, meaning=meaning))

    n_full = sum(len(v) for v in full.values())
    for j in judges:
        q = {t: {d: v for d, v in labels[j].get(t, {}).items() if d in full[t]} for t in full}
        L.write_qrels(os.path.join(args.out, "qrels", f"{j}.txt"), q)
        add(f"labelled_{j}", sum(len(v) for v in q.values()), f"{j}: NIST-judged pairs with a label (of {n_full})")
    add("n_judges", len(judges), "new LLM judges")
    add("n_pairs_judges", len(judges) * (len(judges) - 1) // 2, "judge pairs")
    add("n_pairs_nist", n_full, "NIST-judged pairs, 82 topics")
    add("n_topics_heldout", len([t for t in full if t not in set(test)]), "held-out topics")
    n_test_full = sum(len(full[t]) for t in test)
    omitted = [full[t][d] for t in test for d in full[t] if d not in human[t]]
    add("n_test_llmjudge", sum(len(human[t]) for t in test), "LLMJudge human-judged pairs on the 25 test topics")
    add("n_test_nist", n_test_full, "NIST-judged pairs on the 25 test topics")
    add("n_test_omitted", len(omitted), "NIST-judged pairs on the 25 test topics absent from the LLMJudge qrels")
    add("n_test_omitted_nonrel", sum(v == 0 for v in omitted), "of those, NIST label 0")
    nonrel = sum(v == 0 for t in test for v in full[t].values())
    add("n_test_nist_nonrel", nonrel, "NIST label-0 pairs on the 25 test topics")

    tau = {}
    for r in rows:
        tau.setdefault(r["pool"], {})[r["judge_a"]] = float(r["tau_a"])
        tau[r["pool"]][r["judge_b"]] = float(r["tau_b"])
    for pool in tau:
        v = [tau[pool][j] for j in judges]
        add(f"tau_min_{pool}", f"{min(v):.4f}", f"lowest tau, pool {pool}")
        add(f"tau_max_{pool}", f"{max(v):.4f}", f"highest tau, pool {pool}")
        add(f"top_{pool}", max(judges, key=lambda j: tau[pool][j]), f"highest-tau judge, pool {pool}")
        add(f"bottom_{pool}", min(judges, key=lambda j: tau[pool][j]), f"lowest-tau judge, pool {pool}")
    for a, b in (("test", "heldout"), ("test-full", "heldout")):
        x = np.array([tau[a][j] for j in judges])
        y = np.array([tau[b][j] for j in judges])
        kt = float(taub_rows(x, y[None])[0])
        flips = sum((tau[a][p] - tau[a][q]) * (tau[b][p] - tau[b][q]) < 0 for p, q in itertools.combinations(judges, 2))
        add(f"order_kendall_{a}_{b}", f"{kt:.4f}", f"Kendall tau_b between judge orders, {a} vs {b}")
        add(f"order_flips_{a}_{b}", flips, f"judge pairs ordered oppositely, {a} vs {b}")

    P = {(r["pool"], r["judge_a"], r["judge_b"], int(r["n"])): float(r["power"]) for r in rows}
    G = {(r["pool"], r["judge_a"], r["judge_b"]): float(r["abs_delta"]) for r in rows}
    pairs = sorted({(r["judge_a"], r["judge_b"]) for r in rows})
    sizes = sorted({int(r["n"]) for r in rows})
    add("reps", rows[0]["reps"], "replicates per power value")
    add("max_gap_heldout", f"{max(G[('heldout', a, b)] for a, b in pairs):.4f}", "largest gap in tau, held-out topics")
    add("max_gap_test", f"{max(G[('test', a, b)] for a, b in pairs):.4f}", "largest gap in tau, LLMJudge topics")
    for n in sizes:
        meas = np.array([P[("heldout", a, b, n)] for a, b in pairs])
        add(f"heldout_max_power_{n}", f"{meas.max():.4f}", f"largest power at n={n}, held-out topics")
        add(f"heldout_n_power80_{n}", int((meas >= 0.8).sum()), f"pairs with power >= 0.8 at n={n}, held-out topics")
        for src in ("test", "test-full"):
            pred = np.array([P[(src, a, b, n)] for a, b in pairs])
            add(f"pred_err_{src}_{n}", f"{np.mean(pred - meas):+.4f}", f"mean predicted ({src}) minus measured (held-out) power, n={n}")
            add(f"pred_abserr_{src}_{n}", f"{np.median(np.abs(pred - meas)):.4f}", f"median |predicted ({src}) - measured|, n={n}")
            add(f"pred_corr_{src}_{n}", f"{np.corrcoef(pred, meas)[0, 1]:.4f}", f"correlation predicted ({src}) vs measured, n={n}")
    with open(os.path.join(args.out, "summary.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "value", "meaning"], delimiter="\t")
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['name']}\t{r['value']}")


if __name__ == "__main__":
    main()
