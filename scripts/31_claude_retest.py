"""Step 31. Run-to-run variation of the Claude judges on the 25 LLMJudge topics.

Claude models accept no sampling parameters, so labelling the same pairs twice gives different labels. The pilot
labelled the 4,423 LLMJudge test pairs twice with haiku-llmjudge and haiku-basic (step 28: once synchronously,
whose labels are released in qrels/, and once in a batch). This step compares each judge's two runs: label
agreement, tau of each run, and the Kendall tau_b between the two system rankings. It then replaces a judge's
released labels on the 25 topics by its second run and recomputes how well the eight judges' order on the 25
topics agrees with their order on the 57 held-out topics (step 30 reports it for the released labels).

A pair the second run did not label counts as unjudged, as for every judge; --fill takes it from the first run
instead (a sensitivity check).

Writes to OUT:
  retest/<judge>.txt   the second run's labels on the LLMJudge test pairs (ids and labels only)
  retest.tsv           name, value, meaning; step 13 makes the paper's macros from it

Usage: python scripts/31_claude_retest.py DATA_DIR --labels DIR --retest DIR --out OUT [--fill]
"""
import argparse
import csv
import itertools
import os

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance import llm_label as L
from judge_significance.correlation import taub_rows
from judge_significance.protocol import table3_systems
from judge_significance.simulation import stat_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--labels", required=True, help="released labels of all eight judges (qrels/ from step 30)")
    ap.add_argument("--retest", required=True, help="folder with the second run of some judges")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fill", action="store_true", help="fill pairs the second run lacks from the first run")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    if data["full"] is None:
        raise SystemExit(f"needs {D.FULL_QRELS} in the data folder")
    systems = table3_systems(data["runs"])
    test, human, full = data["test_topics"], data["human"], data["full"]
    held = [t for t in sorted(full) if t not in set(test)]
    first = L.merge_label_dirs([args.labels], D.read_qrels)
    second = L.merge_label_dirs([args.retest], D.read_qrels)
    judges = sorted(first)
    rerun = sorted(set(second) & set(first))
    if not rerun:
        raise SystemExit(f"no judge of {args.labels} has labels in {args.retest}")

    H = {}

    def means(labels, topics, hq):
        key = (id(hq), tuple(topics))
        if key not in H:
            H[key] = E.score_matrix(data["runs"], systems, hq, topics, "ndcg_cut_10").mean(1)
        sub = {t: {d: v for d, v in labels.get(t, {}).items() if d in hq.get(t, {})} for t in topics}
        return H[key], E.score_matrix(data["runs"], systems, sub, topics, "ndcg_cut_10").mean(1)

    def tau(labels, topics, hq):
        h, j = means(labels, topics, hq)
        return float(stat_rows(h[None], j[None])[0])

    def order(a, b):
        x = np.array([a[j] for j in judges])
        y = np.array([b[j] for j in judges])
        flips = sum((a[p] - a[q]) * (b[p] - b[q]) < 0 for p, q in itertools.combinations(judges, 2))
        return float(taub_rows(x, y[None])[0]), flips

    out = []

    def add(name, value, meaning):
        out.append(dict(name=name, value=value, meaning=meaning))

    tau_test = {j: tau(first[j], test, human) for j in judges}
    tau_held = {j: tau(first[j], held, full) for j in judges}
    gaps = np.diff(sorted(tau_test.values()))
    pairs = [(t, d) for t in test for d in human[t]]
    os.makedirs(os.path.join(args.out, "retest"), exist_ok=True)
    swapped = {}
    for j in rerun:
        two = {t: {d: v for d, v in second[j].get(t, {}).items() if d in human[t]} for t in test}
        L.write_qrels(os.path.join(args.out, "retest", f"{j}.txt"), two)
        n_two = sum(len(v) for v in two.values())
        if args.fill:
            two = {t: {**{d: v for d, v in first[j].get(t, {}).items() if d in human[t]}, **two[t]} for t in test}
        both = [(t, d) for t, d in pairs if d in first[j].get(t, {}) and d in two[t]]
        agree = sum(first[j][t][d] == two[t][d] for t, d in both)
        t2 = tau(two, test, human)
        _, m1 = means(first[j], test, human)
        _, m2 = means(two, test, human)
        swapped[j] = t2
        diff = abs(t2 - tau_test[j])
        add(f"n_retest_{j}", n_two, f"{j}: LLMJudge test pairs labelled by the second run (of {len(pairs)})")
        add(f"agree_{j}", f"{agree / len(both):.4f}", f"{j}: share of pairs labelled by both runs with equal labels")
        add(f"tau_first_{j}", f"{tau_test[j]:.4f}", f"{j}: tau on the 25 topics, released run")
        add(f"tau_second_{j}", f"{t2:.4f}", f"{j}: tau on the 25 topics, second run")
        add(f"tau_diff_{j}", f"{diff:.4f}", f"{j}: absolute difference in tau between the two runs")
        add(f"system_kendall_{j}", f"{float(taub_rows(m1, m2[None])[0]):.4f}",
            f"{j}: Kendall tau_b between the system rankings of the two runs")
        add(f"gaps_below_{j}", int((gaps < diff).sum()),
            f"{j}: adjacent gaps among the {len(judges)} judges on the 25 topics smaller than its run difference")
        add(f"rank_first_{j}", 1 + sum(tau_test[k] > tau_test[j] for k in judges if k != j),
            f"{j}: rank among the {len(judges)} judges on the 25 topics with its released run")
        rank = 1 + sum(tau_test[k] > t2 for k in judges if k != j)
        add(f"rank_second_{j}", rank, f"{j}: rank among the {len(judges)} judges on the 25 topics with its second run")
        kt, fl = order({**tau_test, j: t2}, tau_held)
        add(f"order_kendall_swap_{j}", f"{kt:.4f}", f"judge order, 25 vs held-out topics, with {j}'s second run")
        add(f"order_flips_swap_{j}", fl, f"pairs ordered oppositely, with {j}'s second run")
    add("n_adjacent_gaps", len(gaps), f"adjacent gaps among the {len(judges)} judges on the 25 topics")
    add("max_tau_diff", f"{max(abs(swapped[j] - tau_test[j]) for j in rerun):.4f}", "largest run-to-run difference in tau")
    add("min_tau_diff", f"{min(abs(swapped[j] - tau_test[j]) for j in rerun):.4f}", "smallest run-to-run difference in tau")
    variants = [order({**tau_test, **{j: swapped[j] for j in s}}, tau_held)
                for r in range(len(rerun) + 1) for s in itertools.combinations(rerun, r)]
    add("order_kendall_min_runs", f"{min(k for k, _ in variants):.4f}",
        "lowest agreement of the judge orders, 25 vs held-out topics, over every choice of run")
    add("order_kendall_max_runs", f"{max(k for k, _ in variants):.4f}",
        "highest agreement of the judge orders, 25 vs held-out topics, over every choice of run")
    add("order_flips_min_runs", min(f for _, f in variants), "fewest pairs ordered oppositely over every choice of run")
    add("order_flips_max_runs", max(f for _, f in variants), "most pairs ordered oppositely over every choice of run")
    add("fill", int(args.fill), "1 if pairs the second run lacks were taken from the first run")
    with open(os.path.join(args.out, "retest.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "value", "meaning"], delimiter="\t")
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['name']}\t{r['value']}")


if __name__ == "__main__":
    main()
