"""Step 25. Forest plot of the LLMJudge leaderboard: each judge's gap to the top judge with bootstrap
intervals (normal intervals from the bootstrap SE, i.e. the intervals the bootstrap z-test inverts):
thick dark bars uncorrected 95%, thin grey bars simultaneous over the 32 comparisons (Bonferroni). A judge
whose 95% interval excludes 0 has a filled marker; the others are hollow. A legend in the empty upper
right explains markers and lines; the script refuses to save if it would cover an interval.

Writes paper/manuscript/figures/leaderboard.pdf and a TSV with the gaps, SEs and intervals.
--plot-only redraws the figure from that TSV (six decimals) without recomputing anything.

Usage: python scripts/25_leaderboard_figure.py DATA_DIR [--boot 10000] [--seed 0] [--tsv FILE] [--out PDF]
       python scripts/25_leaderboard_figure.py --plot-only --tsv results/g-j2/25_leaderboard_intervals.tsv [--out PDF]

--label-size and --height default to the two-column figure (6 pt judge labels and legend, 3.1 in tall). The ACM AI
Letters version uses --label-size 7 --height 3.6 --legend-above (the 7 pt floor; Nov 21, 2026):
       python scripts/25_leaderboard_figure.py --plot-only --tsv results/g-j2/25_leaderboard_intervals.tsv \
           --label-size 7 --height 3.6 --legend-above --out paper/venues/acm-ai-letters/figures/leaderboard.pdf
"""
import argparse
import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from judge_significance.simulation import stat_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.environ.get("PAPER_ROOT") or os.path.join(os.path.dirname(HERE), "paper")  # where generated paper files go (README, Usage)
_spec = importlib.util.spec_from_file_location("cov15", os.path.join(HERE, "15_real_judge_coverage.py"))
_cov15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cov15)
_spec12 = importlib.util.spec_from_file_location("fig12", os.path.join(HERE, "12_make_figures.py"))
_fig12 = importlib.util.module_from_spec(_spec12)
_spec12.loader.exec_module(_fig12)


def compute(data_dir, boot, seed):
    """Gaps to the top judge with bootstrap SEs and normal intervals; returns (top, top tau, rows, m)."""
    H, J, systems, topics = _cov15.real_population(data_dir)
    pub = _cov15.D.read_published(_cov15.PUBLISHED)
    order = sorted(J, key=lambda j: (-pub[j]["tau"], j))
    top = order[0]
    rng = np.random.default_rng(seed)
    k = H.shape[1]
    idx = rng.integers(0, k, size=(boot, k))
    hb = H[:, idx].mean(2).T
    tb = {j: stat_rows(hb, J[j][:, idx].mean(2).T) for j in order}
    full = np.arange(k)[None]
    tau = {j: float(stat_rows(H[:, full].mean(2).T, J[j][:, full].mean(2).T)[0]) for j in order}
    m = len(order) - 1
    z1, zs = norm.ppf(0.975), norm.ppf(1 - 0.025 / m)
    rows = []
    for j in order[1:]:
        d = tau[top] - tau[j]
        se = float(np.std(tb[top] - tb[j], ddof=1))
        rows.append(dict(judge=j, tau=tau[j], gap=d, se=se, lo95=d - z1 * se, hi95=d + z1 * se,
                         lo_sim=d - zs * se, hi_sim=d + zs * se))
    return top, tau[top], rows, m


def read_rows(path):
    """Rows and header fields of a TSV written by this script (values at six decimals)."""
    with open(path) as f:
        head = f.readline().lstrip("# ").split()
        meta = dict(x.split("=", 1) for x in head)
        cols = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(cols, [line.split("\t")[0]] + [float(v) for v in line.rstrip("\n").split("\t")[1:]]))
                for line in f if line.strip()]
    top_tau = rows[0]["tau"] + rows[0]["gap"]  # gap = tau(top) - tau(judge)
    return meta["top"], top_tau, rows


def draw(rows, top_tau, out, label_size=6, height=3.1, legend_above=False):
    """Figure 1, in the paper style of step 12 (Biolinum, black text, fixed page size)."""
    with _fig12.paper_style():
        fig, ax = plt.subplots(figsize=(_fig12.WIDTH, height))
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color=_fig12.PAPER_BLACK, lw=0.8, zorder=2.5)  # above the gridline at 0 (gridlines sit at z 1.5)
        for yi, r in zip(y, rows):
            ax.plot([r["lo_sim"], r["hi_sim"]], [yi, yi], color="#9a9994", lw=0.8, solid_capstyle="butt", zorder=2)
            ax.plot([r["lo95"], r["hi95"]], [yi, yi], color="#1a1a19", lw=1.6, solid_capstyle="butt", zorder=3)
            excl = r["lo95"] > 0 or r["hi95"] < 0
            ax.plot([r["gap"]], [yi], marker="o", ms=3.4, mew=0.8, mec=_fig12.PAPER_BLACK,
                    mfc=_fig12.PAPER_BLACK if excl else "white", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['judge']} ({r['tau']:.3f})" for r in rows], fontsize=label_size)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.set_ylim(-0.8, len(rows) - 0.2)
        ax.set_xlabel(f"Gap in $\\tau$ to the top judge ($\\tau$ = {top_tau:.3f})", fontsize=_fig12.PAPER_SIZE["tick"])
        ax.grid(axis="x", color=_fig12.GRID, lw=0.4, zorder=0)
        ax.set_xticks(np.round(np.arange(-0.3, 0.41, 0.1), 1))  # the tick positions the figure had before the restyle
        ax.tick_params(axis="x", labelsize=_fig12.PAPER_SIZE["small"])
        from matplotlib.lines import Line2D
        dot = dict(ls="none", marker="o", ms=3.4, mew=0.8, mec=_fig12.PAPER_BLACK)
        handles = [Line2D([], [], mfc=_fig12.PAPER_BLACK, **dot), Line2D([], [], mfc="white", **dot),
                   Line2D([], [], color="#1a1a19", lw=1.6, solid_capstyle="butt"),
                   Line2D([], [], color="#9a9994", lw=0.8, solid_capstyle="butt")]
        labels = ["95% excludes 0", "95% includes 0", "95%, unadjusted", "Bonferroni"]
        if legend_above:
            # Above the axes in two columns (AILET, 7 pt): at 7 pt the upper-right legend would cover intervals.
            leg = ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=label_size,
                            handlelength=1.1, handletextpad=0.4, labelspacing=0.2, columnspacing=1.2, borderpad=0.2,
                            borderaxespad=0.3, frameon=False)
            fig.tight_layout(pad=0.3)
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            _fig12.save_paper_figure(fig, out)
            print("legend above the axes, outside the data area")
            return [r["judge"] for r in rows if r["lo95"] > 0 or r["hi95"] < 0]
        leg = ax.legend(handles, labels,
                        loc="upper right", fontsize=label_size, handlelength=1.1, handletextpad=0.4, labelspacing=0.2,
                        borderpad=0.3, borderaxespad=0.2, frameon=True, framealpha=1.0, facecolor="white",
                        edgecolor="none")
        leg.set_zorder(5)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        # The legend sits in the empty upper right; refuse to save if it covers any interval or marker.
        bb = leg.get_window_extent().transformed(ax.transData.inverted())
        hit = [r["judge"] for yi, r in zip(y, rows)
               if bb.y0 - 0.35 <= yi <= bb.y1 + 0.35 and max(r["hi_sim"], r["hi95"], r["gap"]) >= bb.x0]
        if hit:
            raise SystemExit(f"legend overlaps the intervals of {hit}")
        covered = [r["judge"] for yi, r in zip(y, rows) if bb.y0 - 0.35 <= yi <= bb.y1 + 0.35]
        print(f"legend box (data coordinates): x {bb.x0:.3f} to {bb.x1:.3f}, y {bb.y0:.2f} to {bb.y1:.2f}; "
              f"beside rows {covered[0]} to {covered[-1]} (largest interval end there "
              f"{max(max(r['hi_sim'], r['hi95']) for r in rows if r['judge'] in covered):.3f}); no overlap")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        _fig12.save_paper_figure(fig, out)
    return [r["judge"] for r in rows if r["lo95"] > 0 or r["hi95"] < 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir", nargs="?")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tsv")
    ap.add_argument("--plot-only", action="store_true", help="redraw from --tsv; recompute nothing")
    ap.add_argument("--label-size", type=float, default=6, help="judge labels and legend, pt (AILET: 7)")
    ap.add_argument("--height", type=float, default=3.1, help="figure height, in (AILET: 3.6)")
    ap.add_argument("--legend-above", action="store_true", help="legend above the axes in two columns (AILET)")
    ap.add_argument("--out", default=os.path.join(PAPER, "manuscript", "figures", "leaderboard.pdf"))
    args = ap.parse_args()
    if args.plot_only:
        if not args.tsv:
            ap.error("--plot-only needs --tsv")
        top, top_tau, rows = read_rows(args.tsv)
        filled = draw(rows, top_tau, args.out, args.label_size, args.height, args.legend_above)
        print(f"redrawn from {os.path.relpath(args.tsv, os.path.join(HERE, '..'))} (top {top}); 95% interval excludes 0: {filled}")
        return
    if not args.data_dir:
        ap.error("DATA_DIR is required unless --plot-only")
    top, top_tau, rows, m = compute(args.data_dir, args.boot, args.seed)
    n_excl95 = sum(r["lo95"] > 0 for r in rows)
    n_excl_sim = sum(r["lo_sim"] > 0 for r in rows)
    print(f"top judge {top} (tau {top_tau:.4f}); {m} comparisons; B={args.boot}")
    print(f"  uncorrected 95% intervals excluding 0: {n_excl95}/{m}; simultaneous (Bonferroni) intervals excluding 0: {n_excl_sim}/{m}")
    print(f"  largest gap {max(r['gap'] for r in rows):.4f}; median SE {np.median([r['se'] for r in rows]):.4f}")
    draw(rows, top_tau, args.out, args.label_size, args.height, args.legend_above)
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(f"# top={top} B={args.boot} seed={args.seed} excl95={n_excl95} excl_sim={n_excl_sim} m={m}\n"
                    "judge\ttau\tgap\tse\tlo95\thi95\tlo_sim\thi_sim\n")
            for r in rows:
                f.write("\t".join([r["judge"]] + [f"{r[c]:.6f}" for c in ("tau", "gap", "se", "lo95", "hi95", "lo_sim", "hi_sim")]) + "\n")


if __name__ == "__main__":
    main()
