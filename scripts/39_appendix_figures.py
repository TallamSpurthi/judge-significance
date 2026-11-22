"""Step 39. Two appendix figures for the ACM AI Letters version, drawn from released results (nothing recomputed):

  power-gap.pdf      per-pair power of the bootstrap z-test against the gap in published tau, at 200 and 400 topics
                     (results/g-j2/16_real_judge_power_bootstrap.tsv; near-duplicate pairs left out as in step 16),
                     with the 80% line and the nominal-alpha resolution of step 32 (results/g-j2/32_size_adjusted_resolution.tsv)
  heldout-power.pdf  per-pair power predicted from the 25 LLMJudge topics against power on the 57 held-out topics,
                     at 100 and 200 topics, for the 28 pairs of new judges (results/llm-judges/29_power_bootstrap.tsv)

The script re-derives the numbers the paper states about these data and refuses to save if any differs:
the resolution and its pair count at 200 and 400 topics (step 32's rule), the pairs at least 0.10 apart that
all include one of the three lowest-ranked judges, and the mean prediction error and largest held-out power.
Style and fonts as Figures 1-3 (scripts/12_make_figures.py), so reruns are byte-stable.

Usage: python scripts/39_appendix_figures.py [--outdir paper/venues/acm-ai-letters/figures]
"""
import argparse
import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.environ.get("PAPER_ROOT") or os.path.join(ROOT, "paper")  # where generated paper files go (README, Usage)
_spec12 = importlib.util.spec_from_file_location("fig12", os.path.join(HERE, "12_make_figures.py"))
F = importlib.util.module_from_spec(_spec12)
_spec12.loader.exec_module(F)

LOW = "#0d366b"    # dark end of the paper's blue ramp: pairs with one of the three lowest-ranked judges
OTHER = "#9a9994"  # the grey of Figure 1's thin intervals: all other pairs


def resolution(gaps, power):
    """Step 32's rule: smallest gap g such that 80% of the pairs at least g apart reach 80% power."""
    for g in np.unique(gaps):
        tail = gaps >= g - 1e-12
        if (power[tail] >= 0.8).mean() >= 0.8:
            return g, int(tail.sum())
    return None, 0


def power_gap_data():
    dup = F.near_duplicates()
    rows = [r for r in F.read_tsv(os.path.join(ROOT, "results", "g-j2", "16_real_judge_power_bootstrap.tsv"))
            if frozenset((r["judge_a"], r["judge_b"])) not in dup]
    tau = {}
    for r in rows:
        tau[r["judge_a"]], tau[r["judge_b"]] = float(r["tau_a"]), float(r["tau_b"])
    low3 = sorted(tau, key=lambda j: (tau[j], j))[:3]
    assert tau[low3[2]] < sorted(tau.values())[3], "the three lowest-ranked judges are not unique"
    gaps = np.array([float(r["abs_delta"]) for r in rows])
    low = np.array([r["judge_a"] in low3 or r["judge_b"] in low3 for r in rows])
    power = {n: np.array([float(r[f"power_{n}"]) for r in rows]) for n in (200, 400)}
    ref = {int(r["n"]): r for r in F.read_tsv(os.path.join(ROOT, "results", "g-j2", "32_size_adjusted_resolution.tsv"))}
    res = {}
    for n in (200, 400):
        g, cnt = resolution(gaps, power[n])
        assert g is not None and abs(g - float(ref[n]["nominal_res"])) < 5e-4 and cnt == int(ref[n]["nominal_npairs"]), \
            (n, g, cnt, ref[n]["nominal_res"], ref[n]["nominal_npairs"])
        res[n] = (g, cnt)
    big = gaps >= 0.10 - 1e-12
    assert low[big].all(), "a pair at least 0.10 apart has none of the three lowest-ranked judges"
    print(f"power-gap: {len(rows)} pairs ({len(dup)} near-duplicate pairs left out); lowest three {low3}; "
          f"{int(big.sum())} pairs >= 0.10 apart, all with one of them; {int(low.sum())} pairs with one of them in all; "
          + "; ".join(f"resolution at {n} topics {g:.3f} ({c} pairs)" for n, (g, c) in res.items()))
    return gaps, low, power, res, len(rows)


def power_gap_figure(out):
    gaps, low, power, res, npairs = power_gap_data()
    with F.paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(F.WIDTH, 1.95), sharey=True)
        res_labels = []
        for ax, n in zip(axes, (200, 400)):
            ax.axhline(0.8, color=F.PAPER_REF, lw=0.7, zorder=2)
            g, cnt = res[n]
            ax.axvline(g, color=F.PAPER_BLACK, lw=0.8, ls="--", zorder=2)
            res_labels.append((ax, n, ax.text(g + 0.003, 0.03, f"{g:.3f}\n({cnt})", ha="left", va="bottom",
                                          fontsize=F.PAPER_SIZE["small"])))
            ax.plot(gaps[~low], power[n][~low], ls="none", marker="o", ms=2.4, mfc="none", mec=OTHER, mew=0.5, zorder=3,
                    label=f"Other pairs ({int((~low).sum())})")
            ax.plot(gaps[low], power[n][low], ls="none", marker="^", ms=2.8, mfc=LOW, mec=LOW, mew=0.3, zorder=4,
                    label=f"With one of the 3 lowest judges ({int(low.sum())})")
            ax.set_xlim(-0.004, 0.145)
            ax.set_xticks([0, 0.05, 0.10])
            ax.set_xticklabels(["0", "0.05", "0.10"])
            ax.set_ylim(0, 1.02)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
            ax.grid(axis="y", color=F.GRID, lw=0.4, zorder=0)
            ax.set_title(f"{n} topics", fontsize=F.PAPER_SIZE["tick"], color=F.PAPER_BLACK, pad=2)
            ax.set_xlabel("Gap in published $\\tau$")
        axes[0].set_ylabel("Power ($\\alpha = 0.05$)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   handlelength=0.8, handletextpad=0.3, columnspacing=1.0, title=f"Judge pairs ({npairs})")
        fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0, 1, 0.80))
        fig.canvas.draw()
        # Refuse to save if a resolution label covers a data point (with a margin of one marker radius).
        for ax, n, t in res_labels:
            bb = t.get_window_extent().expanded(1.0, 1.0)
            xy = ax.transData.transform(np.column_stack([gaps, power[n]]))
            pad = 3.0 * fig.dpi / 72
            inside = ((xy[:, 0] > bb.x0 - pad) & (xy[:, 0] < bb.x1 + pad) & (xy[:, 1] > bb.y0 - pad) & (xy[:, 1] < bb.y1 + pad))
            if inside.any():
                raise SystemExit(f"resolution label at {n} topics covers {int(inside.sum())} points")
            if bb.x1 > ax.get_window_extent().x1:
                raise SystemExit(f"resolution label at {n} topics runs past the panel edge")
        print("resolution labels clear of every point and inside their panels")
        F.save_paper_figure(fig, out)


def heldout_data():
    rows = F.read_tsv(os.path.join(ROOT, "results", "llm-judges", "29_power_bootstrap.tsv"))
    P = {(r["pool"], r["judge_a"], r["judge_b"], int(r["n"])): float(r["power"]) for r in rows}
    pairs = sorted({(a, b) for (pool, a, b, n) in P if pool == "heldout"})
    assert len(pairs) == 28, len(pairs)
    reps = {r["reps"] for r in rows}
    assert len(reps) == 1, reps
    out = {}
    for n in (100, 200):
        pred = np.array([P[("test", a, b, n)] for a, b in pairs])
        meas = np.array([P[("heldout", a, b, n)] for a, b in pairs])
        out[n] = (pred, meas)
    # the paper's numbers (numbers.tex: \njErrHundred 0.07, \njErrTwoHundred, \njMaxPowHeldTwoHundred 0.70)
    summ = {r["name"]: r["value"] for r in F.read_tsv(os.path.join(ROOT, "results", "llm-judges", "summary.tsv"))} \
        if "name" in F.read_tsv(os.path.join(ROOT, "results", "llm-judges", "summary.tsv"))[0] else None
    for n in (100, 200):
        pred, meas = out[n]
        err = float(np.mean(pred - meas))
        if summ is not None:
            assert abs(err - float(summ[f"pred_err_test_{n}"])) < 5e-5, (n, err, summ[f"pred_err_test_{n}"])
        print(f"heldout-power: n={n}: mean predicted - held-out {err:+.4f}; predicted higher for "
              f"{int((pred > meas).sum())} of 28 pairs, equal {int((pred == meas).sum())}; largest held-out {meas.max():.3f}")
    if summ is not None:
        assert abs(out[200][1].max() - float(summ["heldout_max_power_200"])) < 5e-5
    return out, int(next(iter(reps)))


def heldout_figure(out):
    data, reps = heldout_data()
    with F.paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(F.WIDTH, 1.9), sharey=True)
        for ax, n in zip(axes, (100, 200)):
            pred, meas = data[n]
            ax.plot([0, 1], [0, 1], color=F.PAPER_REF, lw=0.7, zorder=2)
            ax.axvline(0.8, color=F.PAPER_REF, lw=0.7, ls=":", zorder=2)
            ax.plot(meas, pred, ls="none", marker="o", ms=3.0, mfc=LOW, mec="white", mew=0.4, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            ax.set_xticks(ticks)
            ax.set_xticklabels(["0", "", "0.4", "", "0.8", ""])
            ax.set_yticks(ticks)
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
            ax.grid(color=F.GRID, lw=0.4, zorder=0)
            ax.set_title(f"{n} topics", fontsize=F.PAPER_SIZE["tick"], color=F.PAPER_BLACK, pad=2)
            ax.set_xlabel("Held-out power")
        axes[0].set_ylabel("Predicted power")
        fig.tight_layout(pad=0.3, w_pad=1.2)
        F.save_paper_figure(fig, out)
    return reps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default=os.path.join(PAPER, "venues", "acm-ai-letters", "figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    power_gap_figure(os.path.join(args.outdir, "power-gap.pdf"))
    reps = heldout_figure(os.path.join(args.outdir, "heldout-power.pdf"))
    print(f"held-out powers: {reps} samples per pair and size")


if __name__ == "__main__":
    main()
