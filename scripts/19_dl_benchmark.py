"""Step 19. Second benchmark: LLM relevance judges on TREC DL 2019 and 2020 passage ranking.

Judges come from Arabzadeh and Clarke (GPT-4o and Llama 3.2: binary, UMBRELA, EXAM) and Farzi and Dietz
(UMBRELA with DeepSeek-V3, Llama-3.3-70B, Llama-3-8B, Flan-T5-large). Protocol as for LLMJudge: per-topic
nDCG@10 with pytrec_eval (linear gain, unjudged = non-relevant), Kendall tau-b between mean-nDCG system
rankings under each judge's qrels and under the NIST qrels. Label rules: judge_significance.dlbench.

For each year (separately):
  1. validation: (a) per-topic human nDCG@10 against runs/<year>.ndcg_cut10 (trec_eval output, 4 decimals),
     and pytrec_eval against an independent pure-Python nDCG@10; (b) Farzi and Dietz Table 2 Cohen's kappa
     and Kendall tau, with all runs and with every exclusion that brings the run count to theirs
  2. leaderboard: coverage, label handling and tau_b for every configuration; label-rule sensitivity
  3. separability: adjacent pairs in leaderboard order and top judge vs each other judge, paired
     permutation test (--perms; null: per-topic exchangeability) and bootstrap z-test (--boot; null:
     equal tau); Holm and Benjamini-Hochberg adjusted p. The jackknife t-test is logged for the record
     only: its variance estimate is not consistent for tau, so it has almost no power.
  4. power (unless --no-power): every judge pair, topics drawn with replacement from the year's topics,
     rejection rate of the permutation test (--power-perms) and the bootstrap z-test (--power-boot) at
     alpha, summarised by the full-year gap in tau. Near-duplicate pairs (labels differ on fewer than
     --dup-threshold of the human-judged pairs) are left out of the summaries but kept in the TSV.

Usage: python scripts/19_dl_benchmark.py DATA_DIR [--years dl19 dl20] [--perms 10000] [--boot 10000]
           [--reps 200] [--power-perms 500] [--power-boot 300] [--sizes 25 50 100 200 400]
           [--alpha 0.05] [--dup-threshold 0.01] [--seed 0] [--workers 4] [--no-power] [--tsv PREFIX]

DATA_DIR is the folder that holds the two sparse checkouts (arabzadeh/ and farzi/); see dlbench.py.
--tsv PREFIX writes PREFIX_<year>_{leaderboard,separability,power}.tsv.
"""
import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from judge_significance import dlbench as DB
from judge_significance import evaluation as E
from judge_significance.correlation import cohen_kappa, kendall
from judge_significance.permutation import benjamini_hochberg, bootstrap_z_test, holm, jackknife_t_test, perm_test

BINS = (0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 1.0)
TESTS = ("perm", "boot")          # reported tests
SEP_TESTS = ("perm", "boot", "jack")  # separability log also keeps the jackknife
NAMES = {"perm": "permutation test", "boot": "bootstrap z-test", "jack": "jackknife t-test (record only)"}
_G = {}


def tau_of(h, M):
    return float(kendall(h, M.mean(1), "b"))


def kappas(human, qrels):
    """Cohen's kappa on pairs both label: four-scale and 01-vs-23."""
    pairs = [(q, d) for q in human for d in human[q] if d in qrels.get(q, {})]
    a = [human[q][d] for q, d in pairs]
    b = [qrels[q][d] for q, d in pairs]
    return cohen_kappa(a, b), cohen_kappa([x >= 2 for x in a], [x >= 2 for x in b])


def validate(d, systems, H):
    year, runs, topics = d["year"], d["runs"], d["topics"]
    pub = d["published_ndcg"]
    print(f"\n[1a] per-topic human nDCG@10 vs runs/{year}.ndcg_cut10 (trec_eval -q, printed to 4 decimals)")
    missing_runs = sorted(set(systems) ^ set(pub))
    diffs, missing = [], 0
    for i, s in enumerate(systems):
        for k, q in enumerate(topics):
            if q in pub.get(s, {}):
                diffs.append(abs(H[i, k] - pub[s][q]))
            else:
                missing += 1
    extra = sum(1 for s in pub for q in pub[s] if q not in topics)
    diffs = np.array(diffs)
    print(f"  runs: {len(systems)} in runs/{year}/, {len(pub)} in the CSV, symmetric difference {missing_runs or 'none'}")
    print(f"  run-topic cells compared {len(diffs)}; missing from CSV {missing}; CSV cells on other topics {extra}")
    print(f"  max |pytrec_eval - CSV| = {diffs.max():.3g}; cells above 5e-5 (rounding bound): {(diffs > 5e-5).sum()}")
    I = E.ndcg_cut_linear_matrix(runs, systems, d["human"], topics)
    print(f"  max |pytrec_eval - independent pure-Python nDCG@10| = {np.abs(I - H).max():.3g}")


def validate_fd(d, systems, h, judges, J):
    year = d["year"]
    print(f"\n[1b] Farzi and Dietz Table 2 ({DB.FD_SYSTEMS[year]} systems in their Table 1; {len(systems)} runs here)")
    fd_t, fd_k = DB.FD_TABLE2_TAU[year], DB.FD_TABLE2_KAPPA[year]
    path70 = d["samples"]["llama-3.3-70b/umbrela"][2]
    rep70 = DB.to_qrels(DB.read_farzi_log(path70, reparse=True), "last")
    print("  config                        kappa4  pub    kappa01 pub    | tau(all runs) pub    diff")
    for j in fd_t:
        k4, kb = kappas(d["human"], judges[j])
        t = tau_of(h, J[j])
        print(f"  {j:28s} {k4:6.3f} {fd_k[j][0]:6.3f} {kb:7.3f} {fd_k[j][1]:6.3f} | {t:.4f}       {fd_t[j]:.3f} {t - fd_t[j]:+.4f}")
    M70 = E.score_matrix(d["runs"], systems, rep70, d["topics"], "ndcg_cut_10")
    k4, kb = kappas(d["human"], rep70)
    t = tau_of(h, M70)
    j = "llama-3.3-70b/umbrela"
    print(f"  {'70B re-parsed (sensitivity)':28s} {k4:6.3f} {fd_k[j][0]:6.3f} {kb:7.3f} {fd_k[j][1]:6.3f} | {t:.4f}       "
          f"{fd_t[j]:.3f} {t - fd_t[j]:+.4f}")
    k = len(systems) - DB.FD_SYSTEMS[year]
    if k > 0:
        names = list(fd_t)
        rows = []
        for drop in itertools.combinations(range(len(systems)), k):
            keep = [i for i in range(len(systems)) if i not in drop]
            ts = [float(kendall(h[keep], J[n].mean(1)[keep])) for n in names]
            rows.append((max(abs(a - fd_t[n]) for a, n in zip(ts, names)), [systems[i] for i in drop], ts))
        rows.sort(key=lambda r: (r[0], r[1]))
        exact = [r[1] for r in rows if all(abs(round(a, 3) - fd_t[n]) < 1e-9 for a, n in zip(r[2], names))]
        print(f"  exclusions of {k} run(s) to reach {DB.FD_SYSTEMS[year]}: {len(rows)}; max |diff| over the four <= 0.01: "
              f"{sum(r[0] <= 0.01 for r in rows)}, <= 0.005: {sum(r[0] <= 0.005 for r in rows)}; "
              f"all four equal at 3 dp: {exact or 'none'}")
        best = rows[0][0]
        tied = [r for r in rows if abs(r[0] - best) < 1e-12]
        print(f"  best max |diff| {best:.4f}, reached by {len(tied)} exclusion(s): " + "; ".join(
            ",".join(r[1]) + " (" + " ".join(f"{a:.4f}" for a in r[2]) + ")" for r in tied))


def leaderboard(d, systems, h, judges, J, report):
    print("\n[2] configurations (coverage = human-judged pairs the judge labels; samples = labels per pair)")
    for r in report:
        tau = f"{tau_of(h, J[r['name']]):.4f}" if r["included"] else "-"
        print(f"  {r['name']:28s} {r['covered']:6d}/{r['total']} ({r['covered'] / r['total']:.4%}) "
              f"samples {r['mult']} disagreeing {r['conflict']:5d} | "
              + ("included" if r["included"] else "EXCLUDED: " + r["reason"])
              + (f" [{r['note']}]" if r["note"] else "") + f" tau={tau}")
    tau = {j: tau_of(h, M) for j, M in J.items()}
    order = sorted(J, key=lambda j: (-tau[j], j))
    print(f"\n  leaderboard ({len(order)} configurations, {len(systems)} runs, {len(d['topics'])} topics):")
    for i, j in enumerate(order, 1):
        print(f"   {i:2d}. {j:28s} tau_b = {tau[j]:.4f}")
    return tau, order


def sensitivity(d, systems, h, tau, order):
    print("\n[2s] label-rule sensitivity (tau_b; rank among the primary configurations if substituted)")
    runs, topics, human, samples = d["runs"], d["topics"], d["human"], d["samples"]

    def show(label, qr, base):
        t = tau_of(h, E.score_matrix(runs, systems, qr, topics, "ndcg_cut_10"))
        others = [tau[j] for j in order if j != base]
        rank = 1 + sum(x > t for x in others)
        primary = f"primary {tau[base]:.4f}; " if base in tau else "not in primary set; "
        print(f"  {label:52s} tau_b = {t:.4f} ({primary}rank {rank} of {len(others) + 1})")

    s, _, _ = samples["llama3.2/umbrela"]
    for agg in ("mean", "mode", "mean4", "first", "last"):
        show(f"llama3.2/umbrela, aggregation {agg}", DB.restrict(DB.to_qrels(s, agg), human), "llama3.2/umbrela")
    s, _, path = samples["llama-3.3-70b/umbrela"]
    show("llama-3.3-70b/umbrela, duplicates: first line wins", DB.restrict(DB.to_qrels(s, "first"), human),
         "llama-3.3-70b/umbrela")
    show("llama-3.3-70b/umbrela, labels re-parsed from output",
         DB.restrict(DB.to_qrels(DB.read_farzi_log(path, reparse=True), "last"), human), "llama-3.3-70b/umbrela")
    s, _, _ = samples["gpt-4o/exam-graded-max"]
    show("gpt-4o/exam-graded-max mapped 0-5 -> 0-3", DB.restrict(DB.to_qrels(s, frac_max=5), human),
         "gpt-4o/exam-graded-max")
    for name in ("gpt-4o/exam-graded-mean", "gpt-4o/exam-binary"):
        s, scale, _ = samples[name]
        show(f"{name} binned to 0-3 ({scale})",
             DB.restrict(DB.to_qrels(s, frac_max=float(scale.split(':')[1])), human), name)


def separability(h, H, J, tau, order, perms, n_boot, rng):
    print(f"\n[3] separability: paired permutation test (P = {perms}), bootstrap z-test (B = {n_boot}); "
          "jackknife t-test (df = topics - 1) for the record only")

    def compare(a, b):
        dp, pp = perm_test(h, J[a], J[b], perms, rng)
        db, pb = bootstrap_z_test(H, J[a], J[b], n_boot, rng)
        dj, pj = jackknife_t_test(H, J[a], J[b])
        assert abs(dp - db) < 1e-12 and abs(dp - dj) < 1e-12
        return dp, pp, pb, pj

    rows = []
    for kind, pairs in (("adjacent", list(zip(order, order[1:]))), ("top_vs_other", [(order[0], b) for b in order[1:]])):
        res = [(a, b) + compare(a, b) for a, b in pairs]
        P = {t: np.array([r[3 + i] for r in res]) for i, t in enumerate(SEP_TESTS)}
        adj = {t: (holm(P[t]), benjamini_hochberg(P[t])) for t in SEP_TESTS}
        print(f"\n  {kind}:" + " " * 49 + "delta   | perm p  holm    BH      | boot p  holm    BH      | jack p  holm    BH")
        for i, (a, b, dlt) in enumerate(r[:3] for r in res):
            print(f"   {a:26s} vs {b:26s} {dlt:+.4f} | " + " | ".join(
                f"{P[t][i]:.4f} {adj[t][0][i]:.4f} {adj[t][1][i]:.4f}" for t in SEP_TESTS))
            rows.append((kind, a, b, tau[a], tau[b], dlt) + tuple(
                x for t in SEP_TESTS for x in (P[t][i], adj[t][0][i], adj[t][1][i])))
        for t in SEP_TESTS:
            print(f"  {kind} separable at 0.05, {NAMES[t]}: uncorrected {(P[t] < 0.05).sum()}/{len(res)}, "
                  f"Holm {(adj[t][0] < 0.05).sum()}/{len(res)}, BH {(adj[t][1] < 0.05).sum()}/{len(res)}")
    return rows


def _init(H, J):
    _G["H"], _G["J"] = H, J


def _power(task):
    a, b, sizes, reps, perms, n_boot, alpha, seed = task
    H, J = _G["H"], _G["J"]
    rng = np.random.default_rng(seed)
    out = {t: [] for t in TESTS}
    for n in sizes:
        rej = {t: 0 for t in TESTS}
        for _ in range(reps):
            S = rng.integers(0, H.shape[1], size=n)
            Hs, As, Bs = H[:, S], J[a][:, S], J[b][:, S]
            _, pp = perm_test(Hs.mean(1), As, Bs, perms, rng)
            _, pb = bootstrap_z_test(Hs, As, Bs, n_boot, rng)
            rej["perm"] += pp < alpha
            rej["boot"] += pb < alpha
        for t in TESTS:
            out[t].append(rej[t] / reps)
    return out


def label_disagreement(human, qa, qb):
    """Share of human-judged pairs on which two judges' labels differ (an unlabelled pair counts as None)."""
    pairs = [(q, d) for q in human for d in human[q]]
    return sum(qa.get(q, {}).get(d) != qb.get(q, {}).get(d) for q, d in pairs) / len(pairs)


def power(H, J, tau, order, judges, human, args, year_idx):
    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    dis = np.array([label_disagreement(human, judges[a], judges[b]) for a, b in pairs])
    dup = dis < args.dup_threshold
    seeds = np.random.SeedSequence([args.seed, 19, year_idx]).spawn(len(pairs))
    tasks = [(a, b, args.sizes, args.reps, args.power_perms, args.power_boot, args.alpha, int(s.generate_state(1)[0]))
             for (a, b), s in zip(pairs, seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(H, J)) as ex:
        res = list(ex.map(_power, tasks, chunksize=1))
    gaps = np.abs(np.array([tau[a] - tau[b] for a, b in pairs]))
    print(f"\n[4] power: {len(pairs)} judge pairs, topics drawn with replacement from the {H.shape[1]} topics; "
          f"reps={args.reps}, permutation test P={args.power_perms}, bootstrap z-test B={args.power_boot}, "
          f"alpha={args.alpha} ({time.time() - t0:.0f} s, {args.workers} workers)")
    o = np.argsort(dis)
    print(f"  label disagreement between judges (share of human-judged pairs): min {dis.min():.4f}, "
          f"median {np.median(dis):.4f}; smallest: " + "; ".join(
              f"{pairs[i][0]} vs {pairs[i][1]} {dis[i]:.4f}" for i in o[:3]))
    print(f"  near-duplicate pairs (disagreement < {args.dup_threshold}) excluded from the summaries: {dup.sum()}"
          + ("" if not dup.any() else " (" + "; ".join(f"{pairs[i][0]} vs {pairs[i][1]}" for i in np.flatnonzero(dup)) + ")"))
    keep = ~dup
    for t in TESTS:
        P = np.array([r[t] for r in res])
        print(f"\n  {NAMES[t]}")
        print("  full-year gap      pairs | median power at n = " + " ".join(f"{n:>5d}" for n in args.sizes)
              + " | share of pairs with power >= 0.8")
        for lo, hi in zip(BINS, BINS[1:]):
            m = (gaps >= lo) & (gaps < hi) & keep
            if not m.any():
                print(f"  [{lo:.3f}, {hi:.3f})  {0:6d} |")
                continue
            print(f"  [{lo:.3f}, {hi:.3f})  {m.sum():6d} | " + " ".join(f"{x:5.2f}" for x in np.median(P[m], 0))
                  + "   | " + " ".join(f"{x:5.2f}" for x in (P[m] >= 0.8).mean(0)))
        m = (gaps < 0.005) & keep
        if m.any():
            print(f"  rejection rate, pairs with gap < 0.005 ({m.sum()} pairs; mean over pairs): "
                  + " ".join(f"n={n}: {x:.3f}" for n, x in zip(args.sizes, P[m].mean(0))))
            for i in np.flatnonzero(m):
                print(f"    {pairs[i][0]} vs {pairs[i][1]} (gap {gaps[i]:.4f}): " + " ".join(f"{x:.3f}" for x in P[i]))
        else:
            print("  no pair has gap < 0.005")
    print(f"  Monte Carlo SE of a rate near 0.5 / near 0.05 at {args.reps} reps: "
          f"{np.sqrt(0.25 / args.reps):.3f} / {np.sqrt(0.0475 / args.reps):.3f}")
    return pairs, gaps, dis, res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--years", nargs="+", default=list(DB.YEARS), choices=DB.YEARS)
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--power-perms", type=int, default=500)
    ap.add_argument("--power-boot", type=int, default=300)
    ap.add_argument("--dup-threshold", type=float, default=0.01)
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--tsv", help="prefix for TSV outputs")
    args = ap.parse_args()
    args.workers = min(args.workers, 4)

    print(f"DL 2019/2020 judge benchmark; data {os.path.basename(os.path.normpath(args.data_dir))}; perms={args.perms} boot={args.boot} reps={args.reps} "
          f"power_perms={args.power_perms} power_boot={args.power_boot} sizes={args.sizes} alpha={args.alpha} "
          f"dup_threshold={args.dup_threshold} seed={args.seed} workers={args.workers}")
    print("label rules:\n  " + DB.LABEL_RULES.replace("\n", "\n  "))
    for year in args.years:
        yi = DB.YEARS.index(year)
        t0 = time.time()
        d = DB.load_year(args.data_dir, year)
        systems, topics = sorted(d["runs"]), d["topics"]
        print(f"\n================ {year}: {len(systems)} runs, {len(topics)} topics, "
              f"{sum(len(v) for v in d['human'].values())} judged pairs ================")
        H = E.score_matrix(d["runs"], systems, d["human"], topics, "ndcg_cut_10")
        h = H.mean(1)
        judges, report = DB.build_judges(d["samples"], d["human"])
        J = {j: E.score_matrix(d["runs"], systems, qr, topics, "ndcg_cut_10") for j, qr in judges.items()}
        validate(d, systems, H)
        validate_fd(d, systems, h, judges, J)
        tau, order = leaderboard(d, systems, h, judges, J, report)
        sensitivity(d, systems, h, tau, order)
        rng = np.random.default_rng([args.seed, 19, yi])
        sep = separability(h, H, J, tau, order, args.perms, args.boot, rng)
        print(f"\n  ({time.time() - t0:.0f} s so far for {year})")
        pw = None if args.no_power else power(H, J, tau, order, judges, d["human"], args, yi)

        if args.tsv:
            with open(f"{args.tsv}_{year}_leaderboard.tsv", "w") as f:
                f.write(f"# {year}: runs={len(systems)} topics={len(topics)}; rank 0 = excluded\n")
                f.write("rank\tconfig\tincluded\treason\tcovered\ttotal\tsamples_per_pair\tdisagreeing_pairs\ttau_b\n")
                for r in report:
                    rank = order.index(r["name"]) + 1 if r["included"] else 0
                    t = f"{tau[r['name']]:.6f}" if r["included"] else ""
                    mult = ",".join(f"{k}:{v}" for k, v in r["mult"].items())
                    f.write(f"{rank}\t{r['name']}\t{int(r['included'])}\t{r['reason'] or r['note']}\t{r['covered']}\t"
                            f"{r['total']}\t{mult}\t{r['conflict']}\t{t}\n")
            with open(f"{args.tsv}_{year}_separability.tsv", "w") as f:
                f.write(f"# {year}: runs={len(systems)} topics={len(topics)} perms={args.perms} boot={args.boot} "
                        f"seed={args.seed}; jackknife columns for the record only\n")
                f.write("comparison\tjudge_a\tjudge_b\ttau_a\ttau_b\tdelta\t" + "\t".join(
                    f"p_{t}{s}" for t in SEP_TESTS for s in ("", "_holm", "_bh")) + "\n")
                for r in sep:
                    f.write("\t".join(r[:3]) + "\t" + "\t".join(f"{x:.6f}" for x in r[3:]) + "\n")
            if pw is not None:
                pairs, gaps, dis, res = pw
                with open(f"{args.tsv}_{year}_power.tsv", "w") as f:
                    f.write(f"# {year}: topics drawn with replacement from {len(topics)}; reps={args.reps} "
                            f"power_perms={args.power_perms} power_boot={args.power_boot} alpha={args.alpha} "
                            f"seed={args.seed}; near_duplicate = label disagreement < {args.dup_threshold}\n")
                    f.write("judge_a\tjudge_b\ttau_a\ttau_b\tabs_delta\tlabel_disagreement\tnear_duplicate\t" + "\t".join(
                        f"{t}_{n}" for t in TESTS for n in args.sizes) + "\n")
                    for (a, b), g, x, r in zip(pairs, gaps, dis, res):
                        f.write(f"{a}\t{b}\t{tau[a]:.6f}\t{tau[b]:.6f}\t{g:.6f}\t{x:.6f}\t{int(x < args.dup_threshold)}\t" + "\t".join(
                            f"{x:.4f}" for t in TESTS for x in r[t]) + "\n")


if __name__ == "__main__":
    main()
