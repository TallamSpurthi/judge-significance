"""Step 20. LaTeX table comparing the three judge benchmarks (LLMJudge / DL 2023, DL 2019, DL 2020).

For each benchmark and each test (P: permutation, null of per-topic exchangeability; B: bootstrap
z-test, null of equal tau) it reports: adjacent pairs separable at p < 0.05 (uncorrected; none is
after Holm), comparisons of the top judge separable after Holm, the median rejection rate at 100
topics for pairs of distinct judges with exactly equal tau (empirical size), and the smallest gap
that 80% of the pairs at or above it detect with 80% power at 100 topics. Near-duplicate judges
(labels differ on < 1% of pairs) are left out of the power columns.

Inputs: results/g-j2/18_equal_tau_tests.tsv, 18_label_overlap.tsv, 16_real_judge_power{,_bootstrap}.tsv
        results/g-j3/19_dl{19,20}_{leaderboard,separability,power}.tsv
Usage: python scripts/20_make_replication_table.py [--out paper/manuscript/tables/replication.tex] [--n 100]
"""
import argparse
import csv
import os
import re

import numpy as np

from judge_significance.permutation import holm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.environ.get("PAPER_ROOT") or os.path.join(ROOT, "paper")  # where generated paper files go (README, Usage)
R2, R3 = os.path.join(ROOT, "results", "g-j2"), os.path.join(ROOT, "results", "g-j3")

CAPTION = (r"Three benchmarks under the permutation (P) and bootstrap $z$ (B) tests; $T$ topics, $S$ systems, "
           r"$J$ judges. Adj.: adjacent pairs separable at $p<0.05$, uncorrected (none after Holm). Top: comparisons "
           r"of the top judge separable after Holm. At {n} topics (200 samples per pair): median rejection rate for "
           r"pairs with exactly equal $\tau$ (number of pairs) and the resolution at nominal $\alpha$ (Section~\ref{{sec:method}}). "
           r"--: no tied pairs.")


def read_tsv(path):
    with open(path) as f:
        meta = ""
        body = []
        for line in f:
            if line.startswith("#"):
                meta = meta or line[1:].strip()
            else:
                body.append(line)
    return meta, list(csv.DictReader(body, delimiter="\t"))


def mde(gaps, power):
    """Smallest threshold g such that at least 80% of pairs with gap >= g have power >= 0.8."""
    for g in np.unique(gaps):
        if (power[gaps >= g - 1e-12] >= 0.8).mean() >= 0.8:
            return g
    return None


def summarise_power(rows, key_p, key_b, n):
    gaps = np.array([float(r["abs_delta"]) for r in rows])
    out = {}
    tied = gaps < 1e-9  # exactly equal tau
    for t, key in (("P", key_p), ("B", key_b)):
        pw = np.array([float(r[key.format(n=n)]) for r in rows])
        out[f"size{t}"] = float(np.median(pw[tied])) if tied.any() else None
        out[f"mde{t}"] = mde(gaps, pw)
    out["near"] = int(tied.sum())
    return out


def llmjudge(n):
    _, e = read_tsv(os.path.join(R2, "18_equal_tau_tests.tsv"))
    adj = [r for r in e if r["kind"] == "adjacent"]
    top = [r for r in e if r["kind"] == "top"]
    res = dict(name="LLMJudge (DL 2023)", T=25, S=31, J=33)
    res["adjP"] = sum(float(r["p_perm"]) < 0.05 for r in adj)
    res["adjB"] = sum(float(r["p_boot"]) < 0.05 for r in adj)
    res["topP"] = int((holm([float(r["p_perm"]) for r in top]) < 0.05).sum())
    res["topB"] = sum(float(r["p_boot_holm"]) < 0.05 for r in top)
    res["nadj"], res["ntop"] = len(adj), len(top)
    _, ov = read_tsv(os.path.join(R2, "18_label_overlap.tsv"))
    dup = {frozenset((r["judge_a"], r["judge_b"])) for r in ov if float(r["label_diff_share"]) < 0.01}
    _, pp = read_tsv(os.path.join(R2, "16_real_judge_power.tsv"))
    _, pb = read_tsv(os.path.join(R2, "16_real_judge_power_bootstrap.tsv"))
    key = lambda r: frozenset((r["judge_a"], r["judge_b"]))
    pb = {key(r): r for r in pb}
    rows = [dict(abs_delta=r["abs_delta"], perm=r[f"power_{n}"], boot=pb[key(r)][f"power_{n}"])
            for r in pp if key(r) not in dup]
    res.update(summarise_power(rows, "perm", "boot", n))
    res["dup"] = len(dup)
    return res


def dl(year, n):
    meta, lb = read_tsv(os.path.join(R3, f"19_dl{year}_leaderboard.tsv"))
    kv = dict(re.findall(r"(\w+)=(\d+)", meta))  # e.g. "dl19: runs=37 topics=43; rank 0 = excluded"
    _, sep = read_tsv(os.path.join(R3, f"19_dl{year}_separability.tsv"))
    adj = [r for r in sep if r["comparison"] == "adjacent"]
    top = [r for r in sep if r["comparison"] != "adjacent"]
    res = dict(name=f"DL 20{year}", T=int(kv["topics"]), S=int(kv["runs"]),
               J=sum(r["included"] == "1" for r in lb))
    res["adjP"] = sum(float(r["p_perm"]) < 0.05 for r in adj)
    res["adjB"] = sum(float(r["p_boot"]) < 0.05 for r in adj)
    res["adjHolm"] = sum(float(r["p_perm_holm"]) < 0.05 or float(r["p_boot_holm"]) < 0.05 for r in adj)
    res["topP"] = sum(float(r["p_perm_holm"]) < 0.05 for r in top)
    res["topB"] = sum(float(r["p_boot_holm"]) < 0.05 for r in top)
    res["nadj"], res["ntop"] = len(adj), len(top)
    _, pw = read_tsv(os.path.join(R3, f"19_dl{year}_power.tsv"))
    rows = [r for r in pw if r["near_duplicate"] == "0"]
    res.update(summarise_power(rows, "perm_{n}", "boot_{n}", n))
    res["dup"] = len(pw) - len(rows)
    return res


def fmt(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(PAPER, "manuscript", "tables", "replication.tex"))
    args = ap.parse_args()
    rows = [llmjudge(args.n), dl("19", args.n), dl("20", args.n)]
    for r in rows:
        print(r)
    assert all(r.get("adjHolm", 0) == 0 for r in rows), "caption says no adjacent pair is separable after Holm"
    lines = [
        "% Generated by judge-significance/scripts/20_make_replication_table.py. Do not edit by hand; regenerate.",
        r"\begin{table}[t]",
        r"\caption{" + CAPTION.format(n=args.n) + "}",
        r"\label{tab:replication}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrcrrrr@{}}",
        r"\toprule",
        r"Benchmark & $T$ & $S$ & $J$ & Test & Adj. & Top & Rej., $\Delta\tau=0$ & Resolution \\",
        r"\midrule",
    ]
    for k, r in enumerate(rows):
        for t in ("P", "B"):
            head = f"{r['name']} & {r['T']} & {r['S']} & {r['J']}" if t == "P" else " & & &"
            size = (fmt(r["size" + t]) + (f" ({r['near']})" if t == "P" else "")) if r["near"] else "--"
            lines.append(f"{head} & {t} & {r['adj' + t]}/{r['nadj']} & {r['top' + t]}/{r['ntop']} & {size} & "
                         f"{fmt(r['mde' + t]) if r['mde' + t] is not None else 'none'} \\\\")
        if k < len(rows) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
