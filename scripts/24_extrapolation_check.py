"""Step 24. Does power extrapolated from 25 topics predict power on more real topics? And does the
number of systems change the resolution?

Uses TREC DL 2019 (43 topics) and 2020 (54 topics) with their 10 LLM judge configurations (step 19).
(a) Extrapolation: for each of K random 25-topic subsets, power at n topics is predicted by drawing n
    topics with replacement from the subset (as the paper does for LLMJudge); the reference is power
    from drawing n topics with replacement from all of the year's topics. Compared per judge pair, and
    as the resolution (smallest gap that 80% of the pairs at or above it detect with 80% power).
(b) Systems vs topics (DL 2020): the resolution at n topics with all 59 systems against K random subsets
    of 31 systems (the LLMJudge count).
Test: bootstrap z-test (null: equal tau), alpha = 0.05.

Usage: python scripts/24_extrapolation_check.py DLBENCH_DIR [--k 10] [--reps 100] [--boot 300]
           [--sizes-target full 100] [--seed 0] [--workers 6] [--tsv PREFIX]
"""
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from judge_significance import dlbench as DB
from judge_significance import evaluation as E
from judge_significance.permutation import bootstrap_z_test
from judge_significance.simulation import stat_rows

_G = {}


def _init(data):
    _G.update(data)


def _power(task):
    """task: (key, year, topic_idx or None, system_idx or None, a, b, sizes, reps, n_boot, seed)."""
    key, year, tix, six, a, b, sizes, reps, n_boot, seed = task
    H, J = _G[year]["H"], _G[year]["J"]
    if six is not None:
        H, Ja, Jb = H[six], J[a][six], J[b][six]
    else:
        Ja, Jb = J[a], J[b]
    pool = np.arange(H.shape[1]) if tix is None else np.asarray(tix)
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        rej = 0
        for _ in range(reps):
            S = pool[rng.integers(0, len(pool), size=n)]
            rej += bootstrap_z_test(H[:, S], Ja[:, S], Jb[:, S], n_boot, rng)[1] < 0.05
        out.append(rej / reps)
    return key, out


def resolution(gaps, power):
    for g in np.unique(gaps):
        if (power[gaps >= g - 1e-12] >= 0.8).mean() >= 0.8:
            return float(g)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tsv", help="prefix for TSV outputs")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data, info, tasks = {}, {}, []
    for year in DB.YEARS:
        d = DB.load_year(args.data_dir, year)
        systems, topics = sorted(d["runs"]), d["topics"]
        H = E.score_matrix(d["runs"], systems, d["human"], topics, "ndcg_cut_10")
        judges, _ = DB.build_judges(d["samples"], d["human"])
        J = {j: E.score_matrix(d["runs"], systems, qr, topics, "ndcg_cut_10") for j, qr in judges.items()}
        data[year] = dict(H=H, J=J)
        T = len(topics)
        sizes = [T, 100]
        names = sorted(J)
        pairs = list(itertools.combinations(names, 2))
        tau = {j: float(stat_rows(H.mean(1)[None], J[j].mean(1)[None])[0]) for j in names}
        info[year] = dict(T=T, S=len(systems), sizes=sizes, pairs=pairs, tau=tau)
        subsets = [np.sort(rng.choice(T, size=25, replace=False)) for _ in range(args.k)]
        for a, b in pairs:
            tasks.append((("ref", year, -1, a, b), year, None, None, a, b, sizes, args.reps, args.boot, int(rng.integers(2**63))))
            for s, tix in enumerate(subsets):
                tasks.append((("sub", year, s, a, b), year, tix.tolist(), None, a, b, sizes, args.reps, args.boot,
                              int(rng.integers(2**63))))
        if year == "dl20":
            ssubs = [np.sort(rng.choice(len(systems), size=31, replace=False)) for _ in range(args.k // 2)]
            info[year]["ssubs"] = ssubs
            for s, six in enumerate(ssubs):
                for a, b in pairs:
                    tasks.append((("sys", year, s, a, b), year, None, six.tolist(), a, b, sizes, args.reps, args.boot,
                                  int(rng.integers(2**63))))

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(data,)) as ex:
        res = dict(ex.map(_power, tasks, chunksize=8))

    print(f"bootstrap z-test, alpha=0.05; K={args.k} random 25-topic subsets; reps={args.reps} B={args.boot} seed={args.seed}")
    rows = []
    for year in DB.YEARS:
        I = info[year]
        gaps = np.array([abs(I["tau"][a] - I["tau"][b]) for a, b in I["pairs"]])
        print(f"\n== {year}: {I['T']} topics, {I['S']} systems, {len(I['pairs'])} judge pairs ==")
        for k, n in enumerate(I["sizes"]):
            ref = np.array([res[("ref", year, -1, a, b)][k] for a, b in I["pairs"]])
            sub = np.array([[res[("sub", year, s, a, b)][k] for a, b in I["pairs"]] for s in range(args.k)])
            pred = sub.mean(0)
            err = pred - ref
            r = float(np.corrcoef(pred, ref)[0, 1]) if ref.std() > 0 and pred.std() > 0 else float("nan")
            res_ref = resolution(gaps, ref)
            res_sub = [resolution(gaps, sub[s]) for s in range(args.k)]
            res_sub_num = [x for x in res_sub if x is not None]
            label = "all topics" if n == I["T"] else f"{n} topics"
            print(f"  n = {n} ({label}): power predicted from 25-topic subsets vs measured on all {I['T']} topics: "
                  f"mean error {err.mean():+.3f}, median |error| {np.median(np.abs(err)):.3f}, correlation {r:.2f}")
            print(f"    resolution: measured {res_ref if res_ref is not None else 'none'}; predicted from subsets "
                  f"{('median %.3f, range %.3f-%.3f' % (np.median(res_sub_num), min(res_sub_num), max(res_sub_num))) if res_sub_num else 'none'}"
                  f" ({len(res_sub) - len(res_sub_num)} of {args.k} subsets: none)")
            rows.append(dict(part="extrapolation", year=year, n=n, mean_error=float(err.mean()),
                             median_abs_error=float(np.median(np.abs(err))), corr=r,
                             res_measured=res_ref, res_pred_median=float(np.median(res_sub_num)) if res_sub_num else None,
                             res_pred_min=min(res_sub_num) if res_sub_num else None,
                             res_pred_max=max(res_sub_num) if res_sub_num else None,
                             res_pred_none=len(res_sub) - len(res_sub_num)))
        if "ssubs" in I:
            for k, n in enumerate(I["sizes"]):
                ref = np.array([res[("ref", year, -1, a, b)][k] for a, b in I["pairs"]])
                ss = [np.array([res[("sys", year, s, a, b)][k] for a, b in I["pairs"]]) for s in range(len(I["ssubs"]))]
                # gaps change with the system set: recompute tau on each system subset
                rs = []
                for s, six in enumerate(I["ssubs"]):
                    Hs = data[year]["H"][six]
                    ts = {j: float(stat_rows(Hs.mean(1)[None], data[year]["J"][j][six].mean(1)[None])[0]) for j in I["tau"]}
                    g = np.array([abs(ts[a] - ts[b]) for a, b in I["pairs"]])
                    rs.append(resolution(g, ss[s]))
                rnum = [x for x in rs if x is not None]
                print(f"  systems: n = {n}: resolution with all {I['S']} systems {resolution(gaps, ref)}; with 31 systems "
                      f"{('median %.3f, range %.3f-%.3f' % (np.median(rnum), min(rnum), max(rnum))) if rnum else 'none'}"
                      f" ({len(rs) - len(rnum)} of {len(rs)} subsets: none); median power over pairs "
                      f"{np.median(ref):.2f} (all) vs {np.median([np.median(x) for x in ss]):.2f} (31 systems)")
                rows.append(dict(part="systems", year=year, n=n, res_measured=resolution(gaps, ref),
                                 res_pred_median=float(np.median(rnum)) if rnum else None,
                                 res_pred_min=min(rnum) if rnum else None, res_pred_max=max(rnum) if rnum else None,
                                 res_pred_none=len(rs) - len(rnum),
                                 median_power_all=float(np.median(ref)),
                                 median_power_31=float(np.median([np.median(x) for x in ss]))))
    print(f"\n  Monte Carlo SE of a power estimate near 0.5 at {args.reps} reps: {np.sqrt(0.25 / args.reps):.3f}")
    if args.tsv:
        cols = sorted({c for r in rows for c in r})
        with open(f"{args.tsv}_extrapolation.tsv", "w") as f:
            f.write(f"# bootstrap z-test; K={args.k} reps={args.reps} boot={args.boot} seed={args.seed}\n" + "\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join("" if r.get(c) is None else (f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]))
                                  for c in cols) + "\n")


if __name__ == "__main__":
    main()
