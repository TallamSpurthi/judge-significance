"""Step 13. LaTeX macros for every number the paper cites, computed from the result files.

Writes paper/manuscript/numbers.tex (\newcommand per number) and paper/manuscript/numbers.md (macro,
value, meaning, source file). In the paper, write \nAdjSepHolm{} instead of typing "0", so the
text cannot drift from the results. Regenerate after any re-run; do not edit the outputs by hand.

Usage: python scripts/13_make_numbers.py [--results results] [--outdir paper/manuscript]
"""
import argparse
import csv
import os
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.environ.get("PAPER_ROOT") or os.path.join(ROOT, "paper")  # where generated paper files go (README, Usage)


def read_tsv(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def read_published(path):
    pub = {}
    for r in read_tsv(path):
        pub[r["submission"]] = {k: float(v) for k, v in r.items() if k != "submission"}
    return pub


def _half_up(x, places):
    """Round half up on the decimal value (0.855 -> 0.86), not on the binary float."""
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(round(float(x), 9))).quantize(q, rounding=ROUND_HALF_UP))


def f2(x):
    return _half_up(x, 2)


def f3(x):
    return _half_up(x, 3)


def f4(x):
    return _half_up(x, 4)


def pct(x):
    return _half_up(100 * x, 0)


def need_80(pts):
    """Topics for 80% power by linear interpolation; None if never reached."""
    pts = sorted(pts)
    if pts[0][1] >= 0.8:
        return pts[0][0]
    for (n0, p0), (n1, p1) in zip(pts, pts[1:]):
        if p0 < 0.8 <= p1:
            return n0 + (0.8 - p0) * (n1 - n0) / (p1 - p0)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--outdir", default=os.path.join(PAPER, "manuscript"))
    args = ap.parse_args()
    R1, R2 = os.path.join(args.results, "g-j1"), os.path.join(args.results, "g-j2")
    pub = read_published(os.path.join(ROOT, "resources", "llmjudge_table3.tsv"))
    M = []  # (macro, value, meaning, source)

    def add(name, value, meaning, source):
        assert name.isalpha(), name
        M.append((name, value, meaning, source))

    # ---- permutation tests (06) ----
    src = "results/g-j1/06_permutation_seed0.tsv"
    rows = read_tsv(os.path.join(R1, "06_permutation_seed0.tsv"))
    adj = [r for r in rows if r["comparison"] == "adjacent"]
    top = [r for r in rows if r["comparison"] == "top_vs_other"]
    taus = {}
    for r in adj:
        taus[r["judge_a"]] = float(r["tau_a"])
        taus[r["judge_b"]] = float(r["tau_b"])
    exact = sum(abs(round(t, 4) - pub[j]["tau"]) < 5e-5 for j, t in taus.items())
    add("nJudges", str(len(taus)), "public LLMJudge judges", src)
    add("nTauExact", str(exact), "Table 3 tau values reproduced exactly at 4 dp", src)
    add("nAdjPairs", str(len(adj)), "adjacent judge pairs", src)
    zero = [r for r in adj if abs(float(r["delta"])) < 1e-12]
    add("nAdjZeroGaps", str(len(zero)), "adjacent pairs with identical tau", src)
    add("nAdjNonTied", str(len(adj) - len(zero)), "adjacent pairs with different tau", src)
    add("nAdjSepUnc", str(sum(float(r["p"]) < 0.05 for r in adj)), "adjacent pairs with p < 0.05, uncorrected", src)
    add("nAdjSepHolm", str(sum(float(r["p_holm"]) < 0.05 for r in adj)), "adjacent pairs with Holm p < 0.05", src)
    add("minAdjP", f3(min(float(r["p"]) for r in adj)), "smallest adjacent-pair permutation p", src)
    add("maxAdjGap", f4(max(float(r["delta"]) for r in adj)), "largest adjacent tau gap", src)
    add("nTopOthers", str(len(top)), "comparisons of the top judge with the others", src)
    sig = [r for r in top if float(r["p"]) < 0.05]
    add("nTopSepUnc", str(len(sig)), "top judge vs others with p < 0.05, uncorrected", src)
    add("nTopSepHolm", str(sum(float(r["p_holm"]) < 0.05 for r in top)), "top judge vs others with Holm p < 0.05", src)
    add("minTopSepGap", f3(min(float(r["delta"]) for r in sig)) if sig else "--",
        "smallest tau gap among top-vs-other pairs with p < 0.05", src)
    add("minTopHolmP", f3(min(float(r["p_holm"]) for r in top)), "smallest Holm-adjusted p, top vs others", src)
    ordered = sorted(taus, key=lambda j: -taus[j])
    for name, j in zip(("tauFirst", "tauSecond", "tauThird"), ordered[:3]):
        add(name, f4(taus[j]), f"tau of rank {name[3:].lower()} judge ({j})", src)
    add("topJudge", ordered[0], "top judge", src)

    # ---- bootstrap (08) ----
    src = "results/g-j2/08_bootstrap_seed0.tsv"
    b = [r for r in read_tsv(os.path.join(R2, "08_bootstrap_seed0.tsv")) if r["kind"] == "judge"]
    above = [r for r in b if float(r["estimate"]) > float(r["perc_hi"])]
    inside = [r for r in b if float(r["bca_lo"]) <= float(r["estimate"]) <= float(r["bca_hi"])]
    add("nAbovePerc", str(len(above)), "judges whose tau exceeds their own 95% percentile upper bound", src)
    add("nInsideBCa", str(len(inside)), "judges whose tau lies inside their own BCa interval", src)
    add("medBootBias", f3(np.median([float(r["bias"]) for r in b])), "median bootstrap bias (mean replicate minus tau)", src)
    t0 = next(r for r in b if r["judge_a"] == ordered[0])
    add("topPercLo", f3(float(t0["perc_lo"])), "top judge 95% percentile interval, lower", src)
    add("topPercHi", f3(float(t0["perc_hi"])), "top judge 95% percentile interval, upper", src)
    add("topBCaLo", f3(float(t0["bca_lo"])), "top judge 95% BCa interval, lower", src)
    add("topBCaHi", f3(float(t0["bca_hi"])), "top judge 95% BCa interval, upper", src)
    add("topBCaZ", f2(float(t0["z0"])), "top judge BCa bias-correction constant z0", src)
    add("maxBCaZ", f2(max(float(r["z0"]) for r in b)), "largest BCa z0 over the 33 judges", src)

    # ---- coverage (09) ----
    # n = 25 pools both synthetic judge sets (seeds 0 and 1); n = 15 has seed 0 only.
    src = "results/g-j2/09_coverage.tsv (+ 09_coverage_seed1.tsv at n=25)"
    c = read_tsv(os.path.join(R2, "09_coverage.tsv"))
    s1 = os.path.join(R2, "09_coverage_seed1.tsv")
    if os.path.exists(s1):
        c += [dict(r, target="B:" + r["target"]) for r in read_tsv(s1) if int(r["n"]) == 25]
    words = {25: "TwentyFive", 15: "Fifteen"}
    add("nCovJudgesTwentyFive", str(sum(1 for r in c if int(r["n"]) == 25 and r["kind"] == "tau")),
        "synthetic judges in the 25-topic coverage results (both seeds)", src)
    for n in (25, 15):
        for kind, kw in (("tau", "Tau"), ("delta", "Delta")):
            sub = [r for r in c if int(r["n"]) == n and r["kind"] == kind]
            if not sub:
                continue
            for m, mw in (("percentile", "Perc"), ("bca", "BCa"), ("studentized", "Stud")):
                v = [float(r[f"{m}_cov"]) for r in sub]
                add(f"cov{kw}{mw}{words[n]}Min", f2(min(v)), f"min coverage, {m}, {kind}, n={n}", src)
                add(f"cov{kw}{mw}{words[n]}Max", f2(max(v)), f"max coverage, {m}, {kind}, n={n}", src)
    sub = [r for r in c if int(r["n"]) == 25 and r["kind"] == "tau"]
    bias = [float(r["truth"]) - float(r["mean_est"]) for r in sub]
    add("estBiasMin", f3(min(bias)), "min of (82-topic tau minus mean 25-topic tau)", src)
    add("estBiasMax", f3(max(bias)), "max of (82-topic tau minus mean 25-topic tau)", src)
    # largest systematic reversal: a judge pair whose mean 25-topic delta has the opposite sign to
    # the 82-topic delta
    rev = [r for r in c if int(r["n"]) == 25 and r["kind"] == "delta"
           and float(r["truth"]) * float(r["mean_est"]) < 0]
    add("nReversals", str(len(rev)), "judge pairs whose mean 25-topic delta tau has the opposite sign to the 82-topic delta", src)
    if rev:
        r = max(rev, key=lambda r: abs(float(r["mean_est"])))
        prefix = "B:" if r["target"].startswith("B:") else ""
        a_, b_ = r["target"].removeprefix("B:").split(" - ")
        taurow = {x["target"]: x for x in c if int(x["n"]) == 25 and x["kind"] == "tau"}
        ra, rb = taurow[prefix + a_], taurow[prefix + b_]
        add("revTauTrueA", f3(float(ra["truth"])), f"reversal example (judge set {'B' if prefix else 'A'}): 82-topic tau of {a_}", src)
        add("revTauTrueB", f3(float(rb["truth"])), f"reversal example (judge set {'B' if prefix else 'A'}): 82-topic tau of {b_}", src)
        add("revTauEstA", f3(float(ra["mean_est"])), f"reversal example (judge set {'B' if prefix else 'A'}): mean 25-topic tau of {a_}", src)
        add("revTauEstB", f3(float(rb["mean_est"])), f"reversal example (judge set {'B' if prefix else 'A'}): mean 25-topic tau of {b_}", src)

    path = os.path.join(R2, "09_coverage_shared.tsv")
    if os.path.exists(path):
        src = "results/g-j2/09_coverage_shared.tsv"
        cs = [r for r in read_tsv(path) if int(r["n"]) == 25]
        for kind, kw in (("tau", "Tau"), ("delta", "Delta")):
            sub = [r for r in cs if r["kind"] == kind]
            for m, mw in (("percentile", "Perc"), ("bca", "BCa"), ("studentized", "Stud")):
                v = [float(r[f"{m}_cov"]) for r in sub]
                add(f"covShared{kw}{mw}Min", f2(min(v)), f"shared errors: min coverage, {m}, {kind}, n=25", src)
                add(f"covShared{kw}{mw}Max", f2(max(v)), f"shared errors: max coverage, {m}, {kind}, n=25", src)
        bias = [float(r["truth"]) - float(r["mean_est"]) for r in cs if r["kind"] == "tau"]
        add("estBiasSharedMin", f3(min(bias)), "shared errors: min of (82-topic tau minus mean 25-topic tau)", src)
        add("estBiasSharedMax", f3(max(bias)), "shared errors: max of (82-topic tau minus mean 25-topic tau)", src)

    # ---- power (10) ----
    for fname, tag in (("10_power.tsv", ""), ("10_power_shared.tsv", "Shared"),
                       ("10_power_jackknife.tsv", "Jack"), ("10_power_shared_jackknife.tsv", "SharedJack"),
                       ("10_power_bootstrap.tsv", "Boot"), ("10_power_shared_bootstrap.tsv", "SharedBoot")):
        path = os.path.join(R2, fname)
        if not os.path.exists(path):
            continue
        src = f"results/g-j2/{fname}"
        p = read_tsv(path)
        for t, tw in ((0.0, "Zero"), (0.01, "One"), (0.025, "TwoHalf"), (0.05, "Five"), (0.10, "Ten")):
            sub = [r for r in p if abs(float(r["target"]) - t) < 1e-9]
            if not sub:
                continue
            at25 = [float(r["power"]) for r in sub if int(r["n"]) == 25]
            add(f"pow{tag}{tw}AtTwentyFiveMin", f2(min(at25)), f"min rejection rate at 25 topics, target {t}", src)
            add(f"pow{tag}{tw}AtTwentyFiveMax", f2(max(at25)), f"max rejection rate at 25 topics, target {t}", src)
            add(f"pow{tag}{tw}Max", f2(max(float(r["power"]) for r in sub)),
                f"max rejection rate at any size, target {t}", src)
            pairs = sorted({(r["judge_a"], r["judge_b"]) for r in sub})
            needs = [need_80([(int(r["n"]), float(r["power"])) for r in sub
                              if (r["judge_a"], r["judge_b"]) == pr]) for pr in pairs]
            reached = [x for x in needs if x is not None]
            nmax = max(int(r["n"]) for r in sub)
            add(f"pow{tag}{tw}NeedMin", f"{min(reached):.0f}" if reached else f"$>{nmax}$",
                f"fewest topics for 80% power among pairs, target {t}", src)
            add(f"pow{tag}{tw}NeedMax", (f"$>{nmax}$" if len(reached) < len(needs) else f"{max(reached):.0f}"),
                f"most topics for 80% power among pairs (>{nmax} if some pair never reaches it), target {t}", src)
            add(f"pow{tag}{tw}NotReached", str(len(needs) - len(reached)),
                f"pairs that never reach 80% power up to {nmax} topics, target {t}", src)
            firsts = []
            for pr in pairs:
                pts = sorted((int(r["n"]), float(r["power"])) for r in sub if (r["judge_a"], r["judge_b"]) == pr)
                firsts.append(next((n for n, pw in pts if pw >= 0.8), None))
            hit = [x for x in firsts if x is not None]
            add(f"pow{tag}{tw}FirstMin", str(min(hit)) if hit else f"$>{nmax}$",
                f"smallest grid size at which some pair reaches 80% power (no interpolation), target {t}", src)
            add(f"pow{tag}{tw}FirstMax", (str(max(hit)) if len(hit) == len(firsts) else f"$>{nmax}$"),
                f"grid size by which every pair reaches 80% power (>{nmax} if some never does), target {t}", src)
            add(f"pow{tag}{tw}Reached", str(len(hit)), f"pairs reaching 80% power by {nmax} topics, target {t}", src)

    # ---- power mechanism (14) ----
    path = os.path.join(R2, "14_power_mechanism.tsv")
    if os.path.exists(path):
        src = "results/g-j2/14_power_mechanism.tsv"
        with open(path) as fh:
            sd = float(fh.readline().split("median_sd=")[1])
        add("mechSETwentyFive", f3(sd / 5), "median SE of a system-pair mean nDCG@10 difference at 25 topics", src)
        add("mechSEFourHundred", f3(sd / 20), "median SE of a system-pair mean nDCG@10 difference at 400 topics", src)
        m = read_tsv(path)
        flat = next(r for r in m if r["judge_a"] == "random-0.16" and r["judge_b"] == "random-0.12")
        add("mechFlatDelta", f3(float(flat["delta82"])), "flat-power example: 82-topic delta tau", src)
        add("mechFlatContested", flat["n_contested"], "flat-power example: contested system pairs", src)
        add("mechFlatMaxGap", f3(float(flat["max_gap"])), "flat-power example: largest human gap among contested pairs", src)
        add("mechFlatPowerMax", f2(max(float(flat["power25"]), float(flat["power400"]))),
            "flat-power example: rejection rate at 25 or 400 topics, whichever is larger", src)
        noisy = next(r for r in m if r["judge_a"] == "random-0.04" and r["judge_b"] == "random-0.16")
        add("mechNoisyMaxGap", f3(float(noisy["max_gap"])), "noise-difference example: largest human gap among contested pairs", src)
        add("mechNoisyPowerTwentyFive", f2(float(noisy["power25"])), "noise-difference example: rejection rate at 25 topics", src)
        add("mechNoisyPowerFourHundred", f2(float(noisy["power400"])), "noise-difference example: rejection rate at 400 topics", src)

    # ---- disagreement (11) ----
    src = "results/g-j2/11_disagreement.tsv"
    d = read_tsv(os.path.join(R2, "11_disagreement.tsv"))
    dadj = [r for r in d if r["kind"] == "adjacent"]
    ncon = sum(int(r["n_contested"]) for r in dadj)
    nres = sum(int(r["n_sig"]) for r in dadj)
    add("nContested", str(ncon), "contested (judge pair, system pair) instances over all adjacent judge pairs", src)
    add("nContestedUnres", str(ncon - nres), "of those, instances not significantly ordered by human labels (|t| < t crit, 24 df)", src)
    add("pctContestedUnres", pct((ncon - nres) / ncon), "percent of contested instances not significantly ordered by human labels", src)
    head = open(os.path.join(R2, "11_disagreement.tsv")).readline()
    meta = dict(kv.split("=") for kv in head[1:].split())
    add("tCrit", f"{float(meta['t_crit']):.2f}", "two-sided 5% critical value of t with 24 df", src)
    if "matched_unres" in meta:
        add("pctMatchedUnres", pct(float(meta["matched_unres"])),
            "percent not significantly ordered among non-contested system pairs matched on the human gap", src)
    add("nContestedDistinct", meta["distinct_adjacent"], "distinct system pairs contested by at least one adjacent judge pair", src)
    add("pctContestedDistinctUnres", pct(int(meta["distinct_adjacent_unres"]) / int(meta["distinct_adjacent"])),
        "percent of those distinct system pairs not significantly ordered by human labels", src)
    zs = [float(r["median_abs_t"]) for r in dadj[:2] if r["median_abs_t"] not in ("", "nan")]
    add("medZTopThree", f2(max(zs)), "median |t| of contested pairs among the top three judges (max over the 2 pairs)", src)
    log = open(os.path.join(R2, "11_disagreement.log")).readline()
    frac = log.split("(")[1].split("%")[0]
    add("pctAllUnres", frac, "percent of all 465 system pairs not significantly ordered by human labels", "results/g-j2/11_disagreement.log")

    # ---- equal-tau tests and single precision (18) ----
    src = "results/g-j2/18_equal_tau_tests.tsv"
    path = os.path.join(R2, "18_equal_tau_tests.tsv")
    if os.path.exists(path):
        head = open(path).readline()
        meta = dict(kv.split("=") for kv in head[1:].split())
        e = read_tsv(path)
        ea = [r for r in e if r["kind"] == "adjacent"]
        et = [r for r in e if r["kind"] == "top"]
        fl = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else float("nan")
        excl = lambda r: np.isfinite(fl(r, "stud_lo")) and (fl(r, "stud_lo") > 0 or fl(r, "stud_hi") < 0)
        add("nAdjSepJack", str(sum(fl(r, "p_jack") < 0.05 for r in ea)), "adjacent pairs with jackknife t-test p < 0.05", src)
        add("minAdjPJackNonzero", f2(min(fl(r, "p_jack") for r in ea if abs(fl(r, "delta")) > 1e-9)),
            "smallest jackknife p among adjacent pairs with nonzero gap", src)
        add("nAdjStudExcl", str(sum(excl(r) for r in ea)), "adjacent pairs whose studentized 95% interval for the gap excludes 0", src)
        add("nAdjSepBoot", str(sum(fl(r, "p_boot") < 0.05 for r in ea)), "adjacent pairs with bootstrap z-test p < 0.05", src)
        add("minAdjPBootNonzero", f2(min(fl(r, "p_boot") for r in ea if abs(fl(r, "delta")) > 1e-9)),
            "smallest bootstrap z-test p among adjacent pairs with nonzero gap", src)
        add("nTopSepBootUnc", str(sum(fl(r, "p_boot") < 0.05 for r in et)), "top-vs-other pairs with bootstrap p < 0.05 (uncorrected)", src)
        add("minTopPBoot", f3(min(fl(r, "p_boot") for r in et)), "smallest bootstrap p, top vs others (uncorrected)", src)
        add("minTopHolmPBoot", f2(min(fl(r, "p_boot_holm") for r in et)), "smallest Holm-adjusted bootstrap p, top vs others", src)
        add("minTopBHPBoot", f2(min(fl(r, "p_boot_bh") for r in et)), "smallest BH-adjusted bootstrap p, top vs others", src)
        add("nTopSepJackUnc", str(sum(fl(r, "p_jack") < 0.05 for r in et)), "top-vs-other pairs with jackknife p < 0.05 (uncorrected)", src)
        add("minTopPJack", f2(min(fl(r, "p_jack") for r in et)), "smallest jackknife p, top judge vs others (uncorrected)", src)
        add("minTopBHPJack", f2(min(fl(r, "p_jack_bh") for r in et)), "smallest BH-adjusted jackknife p, top vs others", src)
        add("nTopSepBHPerm", str(sum(fl(r, "p_perm_bh") < 0.05 for r in et)), "top-vs-other pairs separable by the permutation test after BH", src)
        add("minTopBHPerm", f3(min(fl(r, "p_perm_bh") for r in et)), "smallest BH-adjusted permutation p, top vs others", src)
        add("nTopStudExcl", str(sum(excl(r) for r in et)), "top-vs-other pairs whose studentized 95% interval excludes 0 (uncorrected)", src)
        add("dblTauChanged", meta["double_precision_tau_changed"], "published taus that change at 4 decimals with double-precision ordering", src)
        add("dblRankMoved", meta["rank_moved"], "judges whose leaderboard rank changes with double-precision ordering", src)
        add("dblMaxMove", meta["max_rank_move"], "largest rank change (places) with double-precision ordering", src)

    # ---- error placement (17) ----
    path = os.path.join(R2, "17_error_placement.tsv")
    if os.path.exists(path):
        src = "results/g-j2/17_error_placement.tsv"
        e = [r for r in read_tsv(path) if r["variant"] == "label+stratum"]
        k = np.array([float(r["kendall_pub"]) for r in e])
        rk = np.array([int(r["top_judge_rank"]) for r in e])
        add("placeDraws", str(len(e)), "re-draws of error placement", src)
        add("placeKendallMed", f2(np.median(k)), "median Kendall tau between published and re-drawn leaderboard", src)
        add("placeKendallMin", f2(k.min()), "min Kendall tau between published and re-drawn leaderboard", src)
        add("placeKendallMax", f2(k.max()), "max Kendall tau between published and re-drawn leaderboard", src)
        add("placeTopRankMin", str(rk.min()), "best rank of the published top judge after re-drawing", src)
        add("placeTopRankMax", str(rk.max()), "worst rank of the published top judge after re-drawing", src)
        bp = os.path.join(R2, "17_error_placement_between.tsv")
        if os.path.exists(bp):
            bt = [r for r in read_tsv(bp) if r["variant"] == "label+stratum"][0]
            add("placeBetweenMed", f2(float(bt["median_kendall_between_draws"])), "median Kendall tau between two re-drawn leaderboards", bp[len(ROOT) + 1:])

    # ---- real-judge coverage (15) ----
    path = os.path.join(R2, "15_real_judge_coverage.tsv")
    if os.path.exists(path):
        src = "results/g-j2/15_real_judge_coverage.tsv"
        c = read_tsv(path)
        words = {10: "Ten", 15: "Fifteen", 20: "Twenty"}
        for n in sorted({int(r["n"]) for r in c}):
            w = words.get(n, str(n))
            for kind, kw in (("tau", "Tau"), ("delta", "Delta")):
                sub = [r for r in c if int(r["n"]) == n and r["kind"] == kind and float(r["percentile_width"]) > 1e-9]
                for m, mw in (("percentile", "Perc"), ("bca", "BCa"), ("studentized", "Stud")):
                    v = np.array([float(r[f"{m}_cov"]) for r in sub if r[f"{m}_cov"] not in ("", "nan")])
                    if not len(v):
                        continue
                    add(f"real{kw}{mw}{w}Min", f2(v.min()), f"real judges, {n} of 25 topics: min coverage, {m}, {kind}", src)
                    add(f"real{kw}{mw}{w}Med", f2(np.median(v)), f"real judges, {n} of 25 topics: median coverage, {m}, {kind}", src)
                    add(f"real{kw}{mw}{w}Max", f2(v.max()), f"real judges, {n} of 25 topics: max coverage, {m}, {kind}", src)
                    add(f"real{kw}{mw}{w}Ninety", str(int((v >= 0.90).sum())),
                        f"real judges, {n} of 25 topics: targets with coverage >= 0.90, {m}, {kind}", src)
                if kind == "tau":
                    tr = np.array([float(r["truth"]) for r in sub])
                    me = np.array([float(r["mean_est"]) for r in sub])
                    ordered = [(i, j) for i in range(len(sub)) for j in range(len(sub)) if tr[i] > tr[j] + 1e-9]
                    rev = sum(me[i] < me[j] for i, j in ordered)
                    add(f"realRev{w}", str(rev), f"real judges, {n} topics: judge pairs whose mean sampled tau reverses the published order", src)
                    add(f"realRevOf{w}", str(len(ordered)), "real judges: judge pairs with different published tau", src)
                    add(f"realRevPct{w}", pct(rev / len(ordered)), f"real judges, {n} topics: percent of pairs reversed in expectation", src)
                    b = np.array([float(r["mean_est"]) - float(r["truth"]) for r in sub])
                    add(f"realBias{w}Min", f3(b.min()), f"real judges, {n} topics: min (mean estimate - target)", src)
                    add(f"realBias{w}Max", f3(b.max()), f"real judges, {n} topics: max (mean estimate - target)", src)
                    add(f"realBias{w}Med", f3(np.median(b)), f"real judges, {n} topics: median (mean estimate - target)", src)

    # ---- real-judge power (16) ----
    ov = os.path.join(R2, "18_label_overlap.tsv")
    dup = ({frozenset((r["judge_a"], r["judge_b"])) for r in read_tsv(ov) if float(r["label_diff_share"]) < 0.01}
           if os.path.exists(ov) else set())
    for fname, tw in (("16_real_judge_power.tsv", "Perm"), ("16_real_judge_power_bootstrap.tsv", "Boot")):
        path = os.path.join(R2, fname)
        if not os.path.exists(path):
            continue
        src = f"results/g-j2/{fname}"
        rows = [r for r in read_tsv(path) if frozenset((r["judge_a"], r["judge_b"])) not in dup]
        sizes = sorted(int(k.split("_")[1]) for k in rows[0] if k.startswith("power_"))
        gaps = np.array([float(r["abs_delta"]) for r in rows])
        P = np.array([[float(r[f"power_{n}"]) for n in sizes] for r in rows])
        add(f"realPairs{tw}", str(len(rows)), f"judge pairs in the power analysis (near-duplicates excluded: {len(dup)})", src)
        words = {25: "TwentyFive", 50: "Fifty", 100: "Hundred", 200: "TwoHundred", 400: "FourHundred"}
        near = gaps < 0.005
        add(f"realNearPairs{tw}", str(int(near.sum())), "pairs with published gap < 0.005", src)
        for k, n in enumerate(sizes):
            w = words.get(n, str(n))
            add(f"realSize{tw}{w}", f2(np.median(P[near, k])), f"median rejection rate, gap < 0.005, {n} topics", src)
            add(f"realSizeHigh{tw}{w}", f2(np.quantile(P[near, k], 0.9)), f"90th percentile rejection rate, gap < 0.005, {n} topics", src)
            g_need = None
            for g in np.unique(gaps):  # smallest threshold whose whole tail (gap >= g) qualifies
                tail = gaps >= g - 1e-12
                if (P[tail, k] >= 0.8).mean() >= 0.8:
                    g_need = g
                    break
            add(f"mde{tw}{w}", f2(g_need) if g_need is not None else "--",
                f"smallest gap g such that >= 80% of pairs with gap >= g reach 80% power at {n} topics", src)
            add(f"mdeN{tw}{w}", str(int((gaps >= g_need - 1e-12).sum())) if g_need is not None else "0",
                f"judge pairs at least the resolution apart at {n} topics", src)
            if g_need is not None:
                add(f"mde{tw}{w}Exact", f3(g_need), f"resolution at {n} topics to three decimals (the observed gap it equals)", src)
            if tw == "Boot" and n in (200, 400):
                # Monte Carlo stability: redraw each pair's rejection count from Binomial(reps, power)
                reps = int(re.search(r"reps=(\d+)", open(path).readline()).group(1))
                ug = np.unique(gaps)

                def res_of(Pk):
                    for g in ug:
                        tail = gaps >= g - 1e-12
                        if (Pk[tail] >= 0.8).mean() >= 0.8:
                            return g
                    return np.nan
                mc_rng = np.random.default_rng(13)
                sims = np.array([res_of(mc_rng.binomial(reps, P[:, k]) / reps) for _ in range(2000)])
                ok = sims[~np.isnan(sims)]
                add(f"mdeMC{tw}{w}Lo", f3(np.quantile(ok, 0.05)), f"resolution at {n} topics, 5% Monte Carlo quantile", src)
                add(f"mdeMC{tw}{w}Hi", f3(np.quantile(ok, 0.95)), f"resolution at {n} topics, 95% Monte Carlo quantile", src)
                add(f"mdeMCNone{tw}{w}", pct(np.isnan(sims).mean()), f"% of Monte Carlo redraws resolving no gap at {n} topics", src)
            for lo, hi, bw in ((0.025, 0.05, "Mid"), (0.05, 0.10, "Five"), (0.10, 1.0, "Ten")):
                m = (gaps >= lo) & (gaps < hi)
                if m.any():
                    add(f"realPow{tw}{bw}{w}", f2(np.median(P[m, k])), f"median rejection rate, gap in [{lo}, {hi}), {n} topics", src)
        exact = gaps < 1e-9
        add(f"realExactPairs{tw}", str(int(exact.sum())), "pairs of distinct judges with exactly equal published tau (near-duplicates excluded)", src)
        allrows = read_tsv(path)
        P_all = np.array([[float(r[f"power_{n}"]) for n in sizes] for r in allrows])
        exact_all = np.array([float(r["abs_delta"]) < 1e-9 for r in allrows])
        add(f"realExactPairsAll{tw}", str(int(exact_all.sum())), "pairs with exactly equal published tau, near-duplicates included", src)
        adjr = (gaps > 1e-9) & (gaps < 0.025)  # the range of the leaderboard's nonzero adjacent gaps
        for k, n in enumerate(sizes):
            w = words.get(n, str(n))
            add(f"realExactSize{tw}{w}", f2(np.median(P[exact, k])), f"median rejection rate, exactly equal tau, {n} topics", src)
            add(f"realExactSizeMax{tw}{w}", f2(P[exact, k].max()), f"max rejection rate, exactly equal tau, {n} topics", src)
            add(f"realExactSizeAll{tw}{w}", f2(np.median(P_all[exact_all, k])),
                f"median rejection rate, exactly equal tau incl. near-duplicates, {n} topics", src)
            add(f"realPowAdjRange{tw}{w}", f2(np.median(P[adjr, k])), f"median rejection rate, 0 < gap < 0.025, {n} topics", src)
        pub = read_published(os.path.join(ROOT, "resources", "llmjudge_table3.tsv"))
        present = {r["judge_a"] for r in rows} | {r["judge_b"] for r in rows}
        order_j = sorted(present, key=lambda j: (-pub[j]["tau"], j))  # the 33 public judges only
        adj = {frozenset(p) for p in zip(order_j, order_j[1:])}
        A = [i for i, r in enumerate(rows) if frozenset((r["judge_a"], r["judge_b"])) in adj and float(r["abs_delta"]) > 1e-9]
        add(f"realAdjMax{tw}", f2(P[A, -1].max()), f"highest rejection rate at {sizes[-1]} topics among adjacent pairs with nonzero gap", src)
        add(f"realAdjReach{tw}", str(int((P[A, -1] >= 0.8).sum())), f"adjacent pairs with nonzero gap reaching 80% at {sizes[-1]} topics", src)
        add(f"realAdjN{tw}", str(len(A)), "adjacent pairs with nonzero gap (near-duplicates excluded)", src)
        low3 = set(order_j[-3:])  # the three lowest-ranked public judges
        big = [r for r in rows if float(r["abs_delta"]) >= 0.10 - 1e-12]
        add(f"realBigPairs{tw}", str(len(big)), "judge pairs with published gap >= 0.10 (near-duplicates excluded)", src)
        add(f"realBigPairsLow{tw}", str(sum(bool({r["judge_a"], r["judge_b"]} & low3) for r in big)),
            "judge pairs with gap >= 0.10 that include one of the three lowest-ranked judges", src)
        if tw == "Boot":
            # post-hoc sensitivity check: the resolution without the three lowest-ranked judges
            assert pub[order_j[-4]]["tau"] > pub[order_j[-3]]["tau"], "tie at the bottom-three boundary"
            keep = np.array([not ({r["judge_a"], r["judge_b"]} & low3) for r in rows])
            for k, n in enumerate(sizes):
                if n not in (200, 400):
                    continue
                g_low = None
                for g in np.unique(gaps[keep]):
                    tail = keep & (gaps >= g - 1e-12)
                    if (P[tail, k] >= 0.8).mean() >= 0.8:
                        g_low = g
                        break
                w = words.get(n, str(n))
                add(f"mdeNoLow{tw}{w}", f3(g_low) if g_low is not None else "--",
                    f"resolution at {n} topics without the three lowest-ranked judges (post-hoc sensitivity check)", src)
                add(f"mdeNNoLow{tw}{w}", str(int((keep & (gaps >= g_low - 1e-12)).sum())) if g_low is not None else "0",
                    f"judge pairs at least that resolution apart, without the three lowest-ranked judges, {n} topics", src)

    # ---- size-adjusted resolution of the bootstrap z-test (32) ----
    path = os.path.join(R2, "32_size_adjusted_resolution.tsv")
    if os.path.exists(path):
        src = "results/g-j2/32_size_adjusted_resolution.tsv"
        words = {25: "TwentyFive", 50: "Fifty", 100: "Hundred", 200: "TwoHundred", 400: "FourHundred"}
        for r in read_tsv(path):
            n = int(r["n"]); w = words.get(n, str(n))
            val = lambda x: "--" if x == "none" else f3(float(x))
            add(f"mdeCalBoot{w}", val(r["pooled_res"]), f"resolution at {n} topics, bootstrap test calibrated on the tied pairs (pooled)", src)
            add(f"mdeCalNBoot{w}", r["pooled_npairs"], f"pairs at least the calibrated resolution apart, {n} topics", src)
            add(f"mdeCalStrictBoot{w}", val(r["worst_pair_res"]), f"resolution at {n} topics, strictest (worst-pair) calibration", src)
            add(f"mdeCalMCNoneBoot{w}", pct(float(r["mc_none"])), f"% of Monte Carlo draws resolving no gap, pooled calibration, {n} topics", src)
            add(f"mdeCalCritBoot{w}", f3(float(r["pooled_crit"])), f"pooled critical p-value at {n} topics", src)

    # ---- stratified bootstrap (21) ----
    path = os.path.join(R2, "21_stratified_check.tsv")
    if os.path.exists(path):
        src = "results/g-j2/21_stratified_check.tsv"
        meta = dict(kv.split("=") for kv in open(path).readline()[1:].split() if "=" in kv)
        st = read_tsv(path)
        add("stratAdjSep", str(sum(float(r["p"]) < 0.05 for r in st if r["kind"] == "adjacent")),
            "adjacent pairs separable, bootstrap z-test stratified by query type", src)
        add("stratTopSepHolm", str(sum(float(r["p_holm"]) < 0.05 for r in st if r["kind"] == "top")),
            "top-vs-other pairs separable after Holm, stratified bootstrap z-test", src)
        add("stratAbovePerc", meta["percentile_above"], "judges with tau above their stratified percentile interval", src)

    # ---- size and power of three tests incl. studentized permutation (23) ----
    path = os.path.join(R2, "23_test_size_check.tsv")
    if os.path.exists(path):
        src = "results/g-j2/23_test_size_check.tsv"
        t23 = read_tsv(path)
        for g, gw in (("exact_tie", "Tie"), ("gap_ge_0.10", "Big")):
            for n, nw in ((25, "TwentyFive"), (100, "Hundred")):
                sub = [r for r in t23 if r["group"] == g and int(r["n"]) == n]
                for col, cw in (("permutation", "perm"), ("studentized_permutation", "studPerm"), ("bootstrap_z", "bootZ")):
                    add(f"{cw}{gw}{nw}", f2(np.median([float(r[col]) for r in sub])),
                        f"median rejection rate, {col}, {g.replace('_', ' ')}, {n} topics ({len(sub)} pairs)", src)

    path = os.path.join(R2, "23_test_size_check_adjusted.tsv")
    if os.path.exists(path):
        src = "results/g-j2/23_test_size_check_adjusted.tsv"
        for r in read_tsv(path):
            tw = {"permutation": "perm", "studentized_permutation": "studPerm", "bootstrap_z": "bootZ"}[r["test"]]
            nw = {"25": "TwentyFive", "100": "Hundred"}[r["n"]]
            add(f"{tw}AdjPow{nw}", f2(float(r["median_power"])),
                f"size-adjusted median power, {r['test']}, pairs >= 0.10 apart, {r['n']} topics", src)

    # ---- Wald interval coverage (22) ----
    path = os.path.join(R2, "22_wald_coverage.tsv")
    if os.path.exists(path):
        src = "results/g-j2/22_wald_coverage.tsv"
        w22 = read_tsv(path)
        for n, nw in ((10, "Ten"), (15, "Fifteen"), (20, "Twenty")):
            v = np.array([float(r["coverage"]) for r in w22 if int(r["n"]) == n and r["coverage"] not in ("", "nan")])
            if len(v):
                add(f"waldCov{nw}Min", f2(v.min()), f"Wald interval coverage of adjacent gaps, {n} of 25 topics: min", src)
                add(f"waldCov{nw}Med", f2(np.median(v)), f"Wald interval coverage of adjacent gaps, {n} of 25 topics: median", src)
                add(f"waldCov{nw}Max", f2(v.max()), f"Wald interval coverage of adjacent gaps, {n} of 25 topics: max", src)

    # ---- leaderboard forest plot (25) ----
    path = os.path.join(R2, "25_leaderboard_intervals.tsv")
    if os.path.exists(path):
        src = "results/g-j2/25_leaderboard_intervals.tsv"
        meta = dict(kv.split("=") for kv in open(path).readline()[1:].split() if "=" in kv)
        lb = read_tsv(path)
        add("maxTopGap", f3(max(float(r["gap"]) for r in lb)), "largest gap between the top judge and another judge", src)
        add("nTopExclUnc", meta["excl95"], "uncorrected 95% bootstrap intervals for the gap to the top judge excluding 0", src)
        add("nTopExclSim", meta["excl_sim"], "simultaneous (Bonferroni) intervals for the gap to the top judge excluding 0", src)

    # ---- label-level agreement (26) ----
    path = os.path.join(R2, "26_kappa_separability.tsv")
    if os.path.exists(path):
        src = "results/g-j2/26_kappa_separability.tsv"
        k26 = read_tsv(path)
        for kind, kw in (("kappa_adjacent", "KappaAdj"), ("kappa_top", "KappaTop"), ("tau_adjacent", "KappaTauAdj"),
                         ("kappa_top_tau", "KappaTopTau")):
            sub = [r for r in k26 if r["kind"] == kind]
            add(f"n{kw}Unc", str(sum(float(r["p"]) < 0.05 for r in sub)), f"{kind}: separable by kappa at p < 0.05", src)
            add(f"n{kw}Holm", str(sum(float(r["p_holm"]) < 0.05 for r in sub)), f"{kind}: separable by kappa after Holm", src)

    # ---- extrapolation check and systems vs topics (24) ----
    path = os.path.join(args.results, "g-j3", "24_extrapolation.tsv")
    if os.path.exists(path):
        src = "results/g-j3/24_extrapolation.tsv"
        x24 = read_tsv(path)
        for r in x24:
            yw = "Nineteen" if r["year"] == "dl19" else "Twenty"
            nw = "Hundred" if r["n"] == "100" else "Native"
            tag = ("Ext" if r["part"] == "extrapolation" else "Sys") + yw + nw
            for col, cw in (("res_measured", "Meas"), ("res_pred_median", "Pred"), ("res_pred_min", "PredMin"),
                            ("res_pred_max", "PredMax"), ("mean_error", "Err"), ("median_abs_error", "AbsErr"), ("corr", "Corr"),
                            ("median_power_all", "PowAll"), ("median_power_31", "PowThirtyOne")):
                if col in r and r[col] not in ("", None):
                    add(f"{tag}{cw}", f2(float(r[col])), f"{r['part']}, {r['year']}, n={r['n']}: {col}", src)
                elif col.startswith("res_"):
                    add(f"{tag}{cw}", "none", f"{r['part']}, {r['year']}, n={r['n']}: {col} (no gap qualifies)", src)

    # ---- second benchmark: TREC DL 2019 / 2020 (19) ----
    R3 = os.path.join(args.results, "g-j3")
    logp = os.path.join(R3, "19_dl_benchmark.log")
    if os.path.exists(logp):
        src = "results/g-j3/19_dl_benchmark.log"
        lines = open(logp).read().splitlines()
        diffs, kappa_ok = [], 0
        for i, line in enumerate(lines):
            if line.startswith("[1b]"):
                for row in lines[i + 2:i + 6]:
                    c = row.split()
                    diffs.append(abs(float(c[-1])))
                    kappa_ok += c[1] == c[2] and c[3] == c[4]
        add("dlFarziWithin", str(sum(d <= 0.01 for d in diffs)), "Farzi and Dietz taus reproduced within 0.01 (DL19 + DL20)", src)
        add("dlFarziOf", str(len(diffs)), "Farzi and Dietz taus compared", src)
        add("dlFarziWorst", f3(max(diffs)), "largest |reproduced - published tau| (Farzi and Dietz)", src)
        add("dlKappaExact", str(kappa_ok), "Farzi and Dietz configs whose two kappas reproduce exactly", src)
        # the one tau outside 0.01, and the same configuration with its labels re-parsed from the model output
        worst = None
        for i, line in enumerate(lines):
            if line.startswith("[1b]"):
                block = [r.split() for r in lines[i + 2:i + 7]]
                for c in block[:4]:
                    if worst is None or abs(float(c[-1])) > abs(float(worst[0][-1])):
                        reparsed = next((r for r in block[4:] if r[:2] == ["70B", "re-parsed"]), None)
                        worst = (c, reparsed)
        c, rp = worst
        add("dlFarziWorstOurs", f3(float(c[-3])), f"reproduced tau of the worst config ({c[0]})", src)
        add("dlFarziWorstPub", f3(float(c[-2])), f"published tau of the worst config ({c[0]})", src)
        if rp is not None and c[0].startswith("llama-3.3-70b"):
            add("dlFarziWorstReparsed", f3(float(rp[-3])), "the same config with labels re-parsed from the model output", src)
        for year, yw in (("19", "Nineteen"), ("20", "Twenty")):
            sp = read_tsv(os.path.join(R3, f"19_dl{year}_separability.tsv"))
            lb = read_tsv(os.path.join(R3, f"19_dl{year}_leaderboard.tsv"))
            top = [r for r in sp if r["comparison"] != "adjacent"]
            sep_b = [abs(float(r["delta"])) for r in top if float(r["p_boot_holm"]) < 0.05]
            add(f"dl{yw}Judges", str(sum(r["included"] == "1" for r in lb)), f"DL{year} judge configurations", src)
            add(f"dl{yw}TopSepBoot", str(len(sep_b)), f"DL{year}: judges separable from the top judge, bootstrap test after Holm", src)
            add(f"dl{yw}TopSepPerm", str(sum(float(r["p_perm_holm"]) < 0.05 for r in top)),
                f"DL{year}: judges separable from the top judge, permutation test after Holm", src)
            add(f"dl{yw}TopSepBootMinGap", f3(min(sep_b)) if sep_b else "--",
                f"DL{year}: smallest gap separable from the top judge by the bootstrap test after Holm", src)
            adj = [r for r in sp if r["comparison"] == "adjacent"]
            add(f"dl{yw}MaxAdjGap", f3(max(abs(float(r["delta"])) for r in adj)), f"DL{year}: largest adjacent gap", src)

    # ---- eight new LLM judges on all 82 DL 2023 topics (28-30) ----
    path = os.path.join(args.results, "llm-judges", "summary.tsv")
    if os.path.exists(path):
        src = "results/llm-judges/summary.tsv"
        S = {r["name"]: r["value"] for r in read_tsv(path)}
        thousands = lambda k: f"{int(S[k]):,}"  # noqa: E731
        add("njN", S["n_judges"], "new LLM judges (Claude Haiku x3, Sonnet, four open-weight)", src)
        add("njPairs", S["n_pairs_judges"], "pairs of new judges", src)
        add("njPairsNist", thousands("n_pairs_nist"), "NIST-judged pairs of the 82 DL 2023 topics, all labelled", src)
        add("njTopicsHeld", S["n_topics_heldout"], "held-out topics (DL 2023, not in LLMJudge)", src)
        add("njReps", S["reps"], "replicates per power value (step 29)", src)
        add("njMinLabelled", thousands(min((k for k in S if k.startswith("labelled_")), key=lambda k: int(S[k]))),
            "fewest pairs labelled by any new judge", src)
        for pool, pw in (("test", "Test"), ("heldout", "Held"), ("test-full", "TestFull")):
            add(f"njTauMin{pw}", f3(float(S[f"tau_min_{pool}"])), f"lowest new-judge tau, pool {pool}", src)
            add(f"njTauMax{pw}", f3(float(S[f"tau_max_{pool}"])), f"highest new-judge tau, pool {pool}", src)
        add("njOrderKendall", f2(float(S["order_kendall_test_heldout"])),
            "Kendall tau_b between the new judges' orders on the 25 LLMJudge and the held-out topics", src)
        add("njOrderFlips", S["order_flips_test_heldout"], "judge pairs ordered oppositely, LLMJudge vs held-out topics", src)
        add("njOrderKendallFull", f2(float(S["order_kendall_test-full_heldout"])),
            "Kendall tau_b between judge orders, 25 topics with all NIST labels vs held-out topics", src)
        add("njOrderFlipsFull", S["order_flips_test-full_heldout"], "pairs ordered oppositely, test-full vs held-out", src)
        add("njMaxGapHeld", f3(float(S["max_gap_heldout"])), "largest gap in tau between new judges, held-out topics", src)
        add("njMaxGapTest", f3(float(S["max_gap_test"])), "largest gap in tau between new judges, LLMJudge topics", src)
        for n, nw in (("25", "TwentyFive"), ("100", "Hundred"), ("200", "TwoHundred")):
            add(f"njMaxPowHeld{nw}", f2(float(S[f"heldout_max_power_{n}"])), f"largest held-out power at n={n}", src)
            add(f"njNPowEightyHeld{nw}", S[f"heldout_n_power80_{n}"], f"pairs with held-out power >= 0.8 at n={n}", src)
            add(f"njErr{nw}", f2(float(S[f"pred_err_test_{n}"])), f"mean predicted (LLMJudge topics) minus held-out power, n={n}", src)
            add(f"njCorr{nw}", f2(float(S[f"pred_corr_test_{n}"])), f"correlation of predicted and held-out power, n={n}", src)
            add(f"njErrFull{nw}", f2(float(S[f"pred_err_test-full_{n}"])),
                f"mean predicted (25 topics, all NIST labels) minus held-out power, n={n}", src)
        add("njTestLLMJudge", thousands("n_test_llmjudge"), "LLMJudge human-judged pairs on its 25 topics", src)
        add("njTestNist", thousands("n_test_nist"), "NIST-judged pairs on the same 25 topics", src)
        add("njTestOmitted", thousands("n_test_omitted"), "NIST pairs on the 25 topics absent from LLMJudge", src)
        add("njTestOmittedNonrel", thousands("n_test_omitted_nonrel"), "of those, NIST label 0", src)
        add("njTestNistNonrel", thousands("n_test_nist_nonrel"), "NIST label-0 pairs on the 25 topics", src)

    # ---- coverage under the matched design: 25 topics drawn with replacement from the 25 (15 --replace) ----
    path = os.path.join(R2, "15_real_judge_coverage_matched.tsv")
    if os.path.exists(path):
        src = "results/g-j2/15_real_judge_coverage_matched.tsv"
        rows = [r for r in read_tsv(path) if r["kind"] == "tau"]
        add("matchJudges", str(len(rows)), "real judges in the matched coverage design", src)
        for m, mw in (("percentile", "Perc"), ("bca", "BCa"), ("studentized", "Stud")):
            v = np.array([float(r[f"{m}_cov"]) for r in rows])
            add(f"match{mw}Min", f2(v.min()), f"matched design, {m}: lowest coverage over judges", src)
            add(f"match{mw}Med", f2(np.median(v)), f"matched design, {m}: median coverage over judges", src)
            add(f"match{mw}Max", f2(v.max()), f"matched design, {m}: highest coverage over judges", src)
            add(f"match{mw}Nominal", str(int((v >= 0.95).sum())), f"matched design, {m}: judges with coverage >= 0.95", src)

    # ---- run-to-run variation of two Claude judges on the 25 LLMJudge topics (31) ----
    path = os.path.join(args.results, "llm-judges", "retest.tsv")
    if os.path.exists(path):
        src = "results/llm-judges/retest.tsv"
        R = {r["name"]: r["value"] for r in read_tsv(path)}
        add("njRetestDiffMin", f3(float(R["min_tau_diff"])), "smaller run-to-run difference in tau of the two rerun judges", src)
        add("njRetestDiffMax", f3(float(R["max_tau_diff"])), "larger run-to-run difference in tau of the two rerun judges", src)
        add("njRetestBasicRankFirst", R["rank_first_haiku-basic"], "haiku-basic's rank among the new judges, released run", src)
        add("njRetestBasicRankSecond", R["rank_second_haiku-basic"], "haiku-basic's rank among the new judges, second run", src)
        add("njRetestAgreeBasic", pct(float(R["agree_haiku-basic"])), "label agreement of haiku-basic's two runs", src)
        add("njRetestAgreeLLMJudge", pct(float(R["agree_haiku-llmjudge"])), "label agreement of haiku-llmjudge's two runs", src)
        add("njRetestOrderMin", f2(float(R["order_kendall_min_runs"])),
            "lowest judge-order agreement, 25 vs held-out topics, over the choice of runs", src)
        add("njRetestOrderMax", f2(float(R["order_kendall_max_runs"])),
            "highest judge-order agreement, 25 vs held-out topics, over the choice of runs", src)
        add("njRetestFlipsMin", R["order_flips_min_runs"], "fewest reversed judge pairs over the choice of runs", src)
        add("njRetestFlipsMax", R["order_flips_max_runs"], "most reversed judge pairs over the choice of runs", src)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "numbers.tex"), "w") as f:
        f.write("% Generated by judge-significance/scripts/13_make_numbers.py. Do not edit by hand.\n")
        f.write("% Meanings and sources: numbers.md\n")
        for name, value, _, _ in M:
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    with open(os.path.join(args.outdir, "numbers.md"), "w") as f:
        f.write("# Paper numbers\n\nGenerated by `judge-significance/scripts/13_make_numbers.py`; do not edit by hand. "
                "Use the macro in the text, e.g. `\\nAdjSepHolm{}`.\n\n| Macro | Value | Meaning | Source |\n|---|---|---|---|\n")
        for name, value, meaning, source in M:
            f.write(f"| `\\{name}` | {value} | {meaning} | `{source}` |\n")
    print(f"wrote {len(M)} macros to {os.path.relpath(os.path.join(args.outdir, 'numbers.tex'), ROOT)}")


if __name__ == "__main__":
    main()
