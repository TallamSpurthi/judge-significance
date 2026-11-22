"""Step 29. Power to separate the Claude judges (step 28 labels), predicted from the 25 LLMJudge topics and
measured on other TREC DL 2023 topics.

For each topic pool, each judge's tau against the human ordering of the 31 Table 3 systems (nDCG@10) and,
for each judge pair, power at n topics: topics are drawn with replacement from the pool, as the paper does
for LLMJudge (scripts 16 and 24), and power is the share of replicates in which the test rejects at
alpha = 0.05 (bootstrap z-test, null: equal tau; or the paired permutation test).

Pools:
  test       the 25 LLMJudge test topics with the LLMJudge qrels (4,423 pairs), as in the paper (the prediction)
  test-full  the same 25 topics with all their NIST qrels (6,427 pairs), judged like the held-out topics
  heldout  the other judged DL 2023 topics (NIST qrels), which no LLMJudge judge labelled
  all      all judged DL 2023 topics (NIST qrels)
Pilot (Gate 2): `--pools test` with the pilot labels. Main study: `--pools test heldout all`; the
held-out check compares each pair's power predicted from `test` with its power measured on `heldout`.

Usage: python scripts/29_claude_judge_power.py DATA_DIR --labels DIR [DIR ...] [--pools test heldout all]
           [--test bootstrap|permutation] [--sizes 25 50 100 200] [--reps 200] [--boot 300] [--perms 500]
           [--seed 0] [--workers 2] [--tsv FILE]
DIR: folders of <judge>.txt qrels from step 28; a later folder only fills labels the earlier ones lack.
"""
import argparse
import csv
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance import llm_label as L
from judge_significance.permutation import bootstrap_z_test, perm_test
from judge_significance.protocol import table3_systems
from judge_significance.simulation import stat_rows

_G = {}


def pools(data, names):
    """{pool: (topics, human qrels)}; heldout and all need the NIST qrels of all judged topics."""
    test = data["test_topics"]
    out = {}
    for name in names:
        if name == "test":
            out[name] = (test, data["human"])
            continue
        if data["full"] is None:
            raise SystemExit(f"pool {name} needs {D.FULL_QRELS} in the data folder")
        if name == "test-full":
            out[name] = (test, data["full"])
            continue
        allt = sorted(data["full"])
        out[name] = ([t for t in allt if t not in set(test)] if name == "heldout" else allt, data["full"])
    return out


def _init(G):
    _G.update(G)


def _power(task):
    key, pool, a, b, sizes, reps, test, n_boot, perms, alpha, seed = task
    H, J = _G[pool]["H"], _G[pool]["J"]
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        rej = 0
        for _ in range(reps):
            S = rng.integers(0, H.shape[1], size=n)
            if test == "bootstrap":
                p = bootstrap_z_test(H[:, S], J[a][:, S], J[b][:, S], n_boot, rng)[1]
            else:
                p = perm_test(H[:, S].mean(1), J[a][:, S], J[b][:, S], perms, rng)[1]
            rej += p < alpha
        out.append(rej / reps)
    return key, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--pools", nargs="+", choices=("test", "test-full", "heldout", "all"), default=["test"])
    ap.add_argument("--test", choices=("bootstrap", "permutation"), default="bootstrap")
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--judges", nargs="+", help="judges to include (default: every judge found in --labels)")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    data = D.load(args.data_dir)
    systems = table3_systems(data["runs"])
    judges = L.merge_label_dirs(args.labels, D.read_qrels)
    if args.judges:
        missing = sorted(set(args.judges) - set(judges))
        if missing:
            raise SystemExit(f"no labels for {missing} in {args.labels}")
        judges = {j: judges[j] for j in args.judges}
    names = sorted(judges)
    if len(names) < 2:
        raise SystemExit(f"need at least two judges in {args.labels}; found {names}")
    pairs = list(itertools.combinations(names, 2))
    G, info = {}, {}
    for pool, (topics, human) in pools(data, args.pools).items():
        H = E.score_matrix(data["runs"], systems, human, topics, "ndcg_cut_10")
        # a judge is scored only on the pairs the pool's human qrels judge, as LLMJudge scored its judges: the
        # main study also labelled 2,004 NIST pairs on the test topics that the LLMJudge qrels leave out
        sub = {j: {t: {d: v for d, v in judges[j].get(t, {}).items() if d in human.get(t, {})} for t in topics}
               for j in names}
        J = {j: E.score_matrix(data["runs"], systems, sub[j], topics, "ndcg_cut_10") for j in names}
        cover = {j: sum(d in judges[j].get(t, {}) for t in topics for d in human.get(t, {})) for j in names}
        n_h = sum(len(human.get(t, {})) for t in topics)
        tau = {j: float(stat_rows(H.mean(1)[None], J[j].mean(1)[None])[0]) for j in names}
        G[pool] = dict(H=H, J=J)
        info[pool] = dict(T=len(topics), tau=tau, cover=cover, n_h=n_h)

    rng = np.random.default_rng(args.seed)
    tasks = [((pool, a, b), pool, a, b, args.sizes, args.reps, args.test, args.boot, args.perms, args.alpha,
              int(rng.integers(2**63))) for pool in G for a, b in pairs]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(G,)) as ex:
        res = dict(ex.map(_power, tasks))

    print(f"{len(systems)} systems; {args.test} test, alpha {args.alpha}; reps {args.reps}"
          f"{f', B {args.boot}' if args.test == 'bootstrap' else f', {args.perms} permutations'}; seed {args.seed}")
    rows = []
    for pool in G:
        inf = info[pool]
        print(f"\n== pool {pool}: {inf['T']} topics, {inf['n_h']:,} human-judged pairs ==")
        for j in names:
            print(f"  {j}: tau {inf['tau'][j]:.4f}; labels for {inf['cover'][j]:,}/{inf['n_h']:,} human-judged pairs")
        print("  pair: gap in tau; power at n = " + ", ".join(map(str, args.sizes)))
        for a, b in pairs:
            gap = abs(inf["tau"][a] - inf["tau"][b])
            pw = res[(pool, a, b)]
            print(f"    {a} vs {b}: {gap:.4f}; " + ", ".join(f"{p:.3f}" for p in pw))
            for n, p in zip(args.sizes, pw):
                rows.append(dict(pool=pool, topics=inf["T"], judge_a=a, judge_b=b, tau_a=inf["tau"][a], tau_b=inf["tau"][b],
                                 abs_delta=gap, n=n, power=p, test=args.test, reps=args.reps))
    for src in ("test", "test-full"):
        if src not in G or "heldout" not in G:
            continue
        print(f"\n== held-out check: power predicted from pool {src} vs measured on the held-out topics ==")
        for k, n in enumerate(args.sizes):
            pred = np.array([res[(src, a, b)][k] for a, b in pairs])
            meas = np.array([res[("heldout", a, b)][k] for a, b in pairs])
            r = np.corrcoef(pred, meas)[0, 1] if pred.std() > 0 and meas.std() > 0 else float("nan")
            print(f"  n = {n}: mean error {np.mean(pred - meas):+.3f}, median |error| {np.median(np.abs(pred - meas)):.3f}, "
                  f"correlation {r:.2f}")
    if args.tsv:
        os.makedirs(os.path.dirname(os.path.abspath(args.tsv)), exist_ok=True)
        with open(args.tsv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.tsv}")


if __name__ == "__main__":
    main()
