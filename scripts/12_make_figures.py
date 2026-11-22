"""Step 12. Paper figures from the simulation results (single-column ACM sigconf, PDF).

  coverage.pdf            real judges: coverage of nominal 95% bootstrap intervals for each judge's
                          25-topic tau when 25 topics are drawn with replacement, the matched design
                          (15_real_judge_coverage_matched.tsv)
  coverage_without_replacement.pdf
                          the same when 10, 15 or 20 topics are drawn without replacement
                          (15_real_judge_coverage.tsv; not in the paper)
  power.pdf               real judges: median rejection rate vs number of topics by gap bin, one
                          panel per test (16_real_judge_power.tsv, 16_real_judge_power_bootstrap.tsv)
  coverage_synthetic.pdf  synthetic judges on 82 topics, 25-topic samples (09_coverage*.tsv)
  power_synthetic.pdf     synthetic judge pairs, permutation test (10_power.tsv)

Identity never relies on colour alone: each series also has its own marker and line style, and
there is a legend, so the figures survive grayscale printing. Fonts are embedded as TrueType;
Figures 1-3 (coverage.pdf, power.pdf and step 25's leaderboard.pdf) use Linux Biolinum (paper_style()).
Requires matplotlib (pip install -e .[figures]).

Usage: python scripts/12_make_figures.py [--outdir paper/manuscript/figures] (see --help for input paths)
"""
import argparse
import contextlib
import csv
import os
import subprocess
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.environ.get("PAPER_ROOT") or os.path.join(ROOT, "paper")  # where generated paper files go (README, Usage)

# Reference categorical palette, first three slots (validated all-pairs, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
LINESTYLES = ["-", "--", ":"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
WIDTH = 3.33  # sigconf column width, inches

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
})

# ---- Print style for the three paper figures (Figures 1-3) ----------------------------------------------------
# Linux Biolinum, the template's sans font (TeX Live's libertine package), so figure text matches the paper.
# Its x-height is 0.432 em against DejaVu Sans's 0.547, so sizes are 1.27x the old DejaVu sizes to keep text from
# looking smaller. Math is set in Biolinum too (mathtext.fontset "custom", no fallback font), and any
# missing-glyph warning is raised as an error. Biolinum ships with CFF outlines; they are converted to TrueType
# at run time so pdf.fonttype 42 embeds a TrueType subset. The other figures keep the module-wide style.
PAPER_BLACK = "#000000"
PAPER_RAMP = ["#5a95dc", "#1c5cab", "#0d366b"]  # one blue hue, light -> dark (ordinal; light end 3.1:1 on white)
PAPER_SIZE = {"tick": 8.2, "label": 8.9, "small": 7.6}  # pt, Biolinum; equal x-height to DejaVu 6.5 / 7.0 / 6.0
PAPER_REF = "#6b6b6b"  # Figure 3's 80% and alpha reference lines: solid mid-grey, so dash styles mark only data series


def _otf_to_ttf(src, dst):
    """Convert a CFF-flavoured OpenType font to TrueType outlines (fontTools cu2qu), so that matplotlib's
    pdf.fonttype 42 embeds a real TrueType subset (with CFF outlines poppler reports a font-type mismatch)."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    font = TTFont(src, recalcTimestamp=False)
    gs, order = font.getGlyphSet(), font.getGlyphOrder()
    glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, {}
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf[name] = pen.glyph()
    font["glyf"], font["loca"] = glyf, newTable("loca")
    maxp = font["maxp"] = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, k, 0)
    maxp.maxZones = 1
    font["head"].glyphDataFormat = 0
    post = font["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    del font["CFF "]
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(dst)


def _biolinum():
    """TrueType copies (in a temporary folder) of the Biolinum regular and italic OTFs that TeX Live ships, or None."""
    import tempfile
    out = []
    tmp = tempfile.mkdtemp(prefix="biolinum-")
    for name in ("LinBiolinum_R", "LinBiolinum_RI"):
        try:
            src = subprocess.run(["kpsewhich", name + ".otf"], capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            src = ""
        if not src or not os.path.exists(src):
            return None
        dst = os.path.join(tmp, name + ".ttf")
        _otf_to_ttf(src, dst)
        out.append(dst)
    return out


@contextlib.contextmanager
def paper_style():
    """Context for Figures 1-3: Biolinum text and math, black text, glyph warnings as errors."""
    from matplotlib import font_manager
    fonts = _biolinum()
    rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "text.color": PAPER_BLACK, "axes.labelcolor": PAPER_BLACK,
          "xtick.labelcolor": PAPER_BLACK, "ytick.labelcolor": PAPER_BLACK, "axes.unicode_minus": True,
          "font.size": PAPER_SIZE["label"], "axes.labelsize": PAPER_SIZE["label"],
          "xtick.labelsize": PAPER_SIZE["tick"], "ytick.labelsize": PAPER_SIZE["tick"],
          "legend.fontsize": PAPER_SIZE["tick"], "legend.title_fontsize": PAPER_SIZE["tick"]}
    if fonts:
        for f in fonts:
            font_manager.fontManager.addfont(f)
        fam = font_manager.FontProperties(fname=fonts[0]).get_name()
        rc.update({"font.family": fam, "mathtext.fontset": "custom", "mathtext.rm": fam,
                   "mathtext.it": f"{fam}:italic", "mathtext.bf": fam, "mathtext.fallback": None})
    else:
        print("note: Linux Biolinum not found (kpsewhich LinBiolinum_R.otf); figures use DejaVu Sans")
    with plt.rc_context(rc), warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*[Gg]lyph.*missing")
        warnings.filterwarnings("error", message=r".*substituting with a dummy symbol")
        yield


def save_paper_figure(fig, out):
    """Fixed page size (the figsize; no bbox_inches='tight'), no creation date, so reruns are byte-stable."""
    fig.savefig(out, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, ROOT)}")  # relative to the repository: logs carry no local paths


def read_tsv(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def judge_label(name):
    model, eps = name.split("-")
    return f"{model.capitalize()} {float(eps):.1f}"


def coverage_figure(paths, out, n=25):
    """One row per synthetic judge, pooled over the judge sets in `paths` (set A, set B, ...)."""
    rows, judges = [], []
    for k, path in enumerate(paths):
        tag = "AB"[k] if len(paths) > 1 else ""
        for r in sorted(read_tsv(path), key=lambda r: r["target"]):
            if r["kind"] == "tau" and int(r["n"]) == n:
                r = dict(r, label=(f"{tag}: " if tag else "") + judge_label(r["target"]))
                rows.append(r)
                judges.append(r["label"])
    methods = [("percentile", "Percentile"), ("bca", "BCa"), ("studentized", "Studentized")]
    fig, ax = plt.subplots(figsize=(WIDTH, 0.6 + 0.17 * len(judges)))
    y = np.arange(len(judges))[::-1]
    ax.axvline(0.95, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=2)
    ax.set_xlim(0.0, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.95])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", ".95"])
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    for k, (key, label) in enumerate(methods):
        xs = [float(r[f"{key}_cov"]) for r in rows]
        ax.scatter(xs, y + (1 - k) * 0.22, s=14, marker=MARKERS[k], color=SERIES[k],
                   edgecolor="white", linewidth=0.4, zorder=3, label=label)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(y)
    ax.set_yticklabels(judges)
    ax.set_ylim(-0.6, len(judges) - 0.4)
    ax.set_xlabel(f"Coverage of nominal 95% interval for $\\tau$ ({n} topics)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.normpath(out)}")


def power_figure(path, out, targets):
    rows = read_tsv(path)
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    ax.axhline(0.8, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=2)
    ax.axhline(0.05, color=INK, lw=0.7, ls=(0, (1, 1.5)), zorder=2)
    ax.axvline(25, color=GRID, lw=0.8, zorder=0)
    sizes = sorted({int(r["n"]) for r in rows})
    for k, t in enumerate(targets):
        pairs = []
        for r in rows:
            if abs(float(r["target"]) - t) < 1e-9 and (r["judge_a"], r["judge_b"]) not in pairs:
                pairs.append((r["judge_a"], r["judge_b"]))
        deltas = []
        for i, (a, b) in enumerate(pairs):
            pts = sorted((int(r["n"]), float(r["power"]), float(r["delta82"])) for r in rows
                         if abs(float(r["target"]) - t) < 1e-9 and r["judge_a"] == a and r["judge_b"] == b)
            deltas.append(pts[0][2])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[k], ls=LINESTYLES[k], lw=1.0,
                    marker=MARKERS[k], ms=3.2, mec="white", mew=0.4, zorder=3,
                    label=(f"$\\Delta\\tau \\approx {t:g}$" if i == 0 else None))
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", ".2", ".4", ".6", ".8", "1"])
    for yv, txt in ((0.8, "80%"), (0.05, "$\\alpha$")):
        ax.text(1.01, yv, txt, fontsize=6, color=INK2, va="center", transform=ax.get_yaxis_transform())
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    ax.yaxis.get_gridlines()[4].set_visible(False)  # the 0.8 reference line replaces this gridline
    ax.set_xlabel("Number of topics (log scale)")
    ax.set_ylabel("Rejection rate ($\\alpha = 0.05$)")
    ax.text(27, 0.31, "25 topics\n(LLMJudge)", fontsize=6, color=INK2, va="center")
    ax.legend(loc="center left", bbox_to_anchor=(1.06, 0.5), frameon=False, handlelength=2.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.normpath(out)}")


METHODS = [("percentile", "Percentile"), ("bca", "BCa"), ("studentized", "Studentized")]


def real_coverage_figure(path, out, sizes=(10, 15, 20)):
    """Three panels (topic-set sizes); in each, one row per method and one dot per real judge."""
    rows = [r for r in read_tsv(path) if r["kind"] == "tau"]
    fig, axes = plt.subplots(1, len(sizes), figsize=(WIDTH, 1.25), sharey=True)
    rng = np.random.default_rng(0)
    for ax, n in zip(axes, sizes):
        sub = [r for r in rows if int(r["n"]) == n]
        ax.axvline(0.95, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=2)
        for k, (key, label) in enumerate(METHODS):
            xs = np.array([float(r[f"{key}_cov"]) for r in sub])
            xs = xs[np.isfinite(xs)]
            yk = len(METHODS) - 1 - k
            ax.scatter(xs, yk + rng.uniform(-0.18, 0.18, len(xs)), s=6, marker=MARKERS[k], color=SERIES[k],
                       edgecolor="white", linewidth=0.3, zorder=3, alpha=0.9)
            med = float(np.median(xs))
            ax.plot([med, med], [yk - 0.32, yk + 0.32], color=INK, lw=1.1, zorder=4)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 0.95])
        ax.set_xticklabels(["0", ".5", ".95"])
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.set_title(f"{n} of 25 topics", fontsize=6.5, color=INK, pad=2)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(METHODS)))
    axes[0].set_yticklabels([m[1] for m in METHODS][::-1])
    axes[0].set_ylim(-0.6, len(METHODS) - 0.4)
    fig.supxlabel("Coverage of the published $\\tau$ (dots: judges; bars: median)", fontsize=7, y=0.02)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.normpath(out)}")


def matched_coverage_figure(path, out, n=25, height=1.3):
    """Figure 2. One panel for the matched design: one row per method, one dot per real judge.
    Rows are labelled, so all three share one near-black colour; shapes still differ. 1.3 in tall with smaller
    markers and wider vertical jitter, so the BCa and studentized clusters do not overplot."""
    from matplotlib import patheffects
    rows = [r for r in read_tsv(path) if r["kind"] == "tau" and int(r["n"]) == n]
    with paper_style():
        fig, ax = plt.subplots(figsize=(WIDTH, height))
        rng = np.random.default_rng(0)
        ax.axvline(0.95, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=2)
        for k, (key, label) in enumerate(METHODS):
            xs = np.array([float(r[f"{key}_cov"]) for r in rows])
            xs = xs[np.isfinite(xs)]
            yk = len(METHODS) - 1 - k
            ax.scatter(xs, yk + rng.uniform(-0.28, 0.28, len(xs)), s=8.5, marker=MARKERS[k], color="#2b2b2a",
                       edgecolor="white", linewidth=0.3, zorder=3, alpha=0.6)
            med = float(np.median(xs))
            ax.plot([med, med], [yk - 0.4, yk + 0.4], color=PAPER_BLACK, lw=2.0, zorder=4, solid_capstyle="butt",
                    path_effects=[patheffects.withStroke(linewidth=3.4, foreground="white")])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 0.95])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "0.95"])
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([m[1] for m in METHODS][::-1])
        ax.set_ylim(-0.6, len(METHODS) - 0.4)
        ax.set_xlabel("Coverage of the published $\\tau$", labelpad=1.0)
        fig.tight_layout(pad=0.3)
        save_paper_figure(fig, out)


GAP_BINS = [((0.0, 1e-9), "$\\Delta\\tau = 0$"), ((1e-9, 0.025), "$0 < |\\Delta\\tau| < 0.025$"), ((0.10, 1.0), "$|\\Delta\\tau| \\geq 0.10$")]


NEAR_DUPLICATE = 0.01  # judge pairs whose labels differ on fewer than 1% of pairs are left out


def near_duplicates(path=os.path.join(ROOT, "results", "g-j2", "18_label_overlap.tsv")):
    return {frozenset((r["judge_a"], r["judge_b"])) for r in read_tsv(path)
            if float(r["label_diff_share"]) < NEAR_DUPLICATE}


def power_by_bin(path, bins):
    dup = near_duplicates()
    rows = [r for r in read_tsv(path) if frozenset((r["judge_a"], r["judge_b"])) not in dup]
    sizes = sorted(int(k.split("_")[1]) for k in rows[0] if k.startswith("power_"))
    out = []
    for (lo, hi), label in bins:
        sub = [r for r in rows if lo <= float(r["abs_delta"]) < hi]
        P = np.array([[float(r[f"power_{n}"]) for n in sizes] for r in sub])
        out.append((label, len(sub), sizes, np.median(P, 0), np.quantile(P, 0.25, 0), np.quantile(P, 0.75, 0)))
    return out


DODGE = [0.93, 1.0, 1.075]  # x offsets of the three series on the log axis (multiplicative)


def real_power_figure(perm_path, boot_path, out, bins=GAP_BINS):
    """Figure 3. Median rejection rate (line) and interquartile range over judge pairs (bar) per gap bin.
    Bars sit only at the measured sizes (no band between them); the three series are offset by a few percent
    along the log x axis (DODGE) so their bars do not hide each other.
    The bins are ordered by gap, so they take one blue hue from light to dark: |delta tau| >= 0.10, the
    headline series, is darkest in colour and in grayscale. Shapes and dash styles still differ.
    The 80% and alpha reference lines are solid mid-grey and labelled directly, in the gap between the panels;
    they sit below the data lines, bars and markers (zorder 3+). The script refuses to save if a label would
    touch a data line, marker, bar, panel edge or tick."""
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.75), sharey=True)
        for ax, (path, title) in zip(axes, ((perm_path, "Permutation test"), (boot_path, "Bootstrap $z$-test"))):
            ax.axhline(0.8, color=PAPER_REF, lw=0.7, zorder=2)
            ax.axhline(0.05, color=PAPER_REF, lw=0.7, zorder=2)
            for k, (label, npairs, sizes, med, q1, q3) in enumerate(power_by_bin(path, bins)):
                xs = np.array(sizes, dtype=float) * DODGE[k]
                ax.errorbar(xs, med, yerr=[med - q1, q3 - med], fmt="none", ecolor=PAPER_RAMP[k], elinewidth=0.8,
                            capsize=1.6, capthick=0.8, zorder=2.5 + k)
                ax.plot(xs, med, color=PAPER_RAMP[k], ls=LINESTYLES[k], lw=1.6, marker=MARKERS[k], ms=3.9,
                        mec="white", mew=0.4, zorder=3 + k, label=f"{label} ({npairs})")
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(n) for n in sizes])
            ax.minorticks_off()
            ax.set_ylim(0, 1.02)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
            ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
            ax.set_title(title, fontsize=PAPER_SIZE["tick"], color=PAPER_BLACK, pad=2)
            ax.set_xlabel("Topics (log scale)")
        axes[0].set_ylabel("Rejection rate ($\\alpha = 0.05$)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.8, handletextpad=0.4, columnspacing=0.7, title="Gap in published $\\tau$ (pairs)")
        # w_pad: keeps "400" and "25" apart, and leaves room for the reference-line labels between the panels
        fig.tight_layout(pad=0.3, w_pad=2.0, rect=(0, 0, 1, 0.78))
        label_reference_lines(fig, axes, {0.8: "80%", 0.05: "$\\alpha$"})
        save_paper_figure(fig, out)


def label_reference_lines(fig, axes, levels):
    """Label Figure 3's reference lines in the gap between the panels, centred between the left panel's edge
    and the tips of the right panel's y ticks (which point into the gap). Refuse to save on any overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    a0, a1 = axes[0].get_window_extent(), axes[1].get_window_extent()
    tick = plt.rcParams["ytick.major.size"] * fig.dpi / 72
    to_fig = fig.transFigure.inverted()
    x_mid = to_fig.transform(((a0.x1 + a1.x0 - tick) / 2, 0))[0]
    labels = [fig.text(x_mid, to_fig.transform(axes[0].transData.transform((1, y)))[1], text,
                       fontsize=PAPER_SIZE["tick"], color=PAPER_REF, ha="center", va="center")
              for y, text in levels.items()]
    fig.canvas.draw()
    pt = 72 / fig.dpi
    for lab in labels:
        bb = lab.get_window_extent(renderer)
        hits = []
        if bb.x0 <= a0.x1 or bb.x1 >= a1.x0 - tick:
            hits.append("panel edge or tick")
        for ax in axes:
            for ln in ax.get_lines():
                if ln.get_color() == PAPER_REF:
                    continue  # the reference lines end at the panel edges
                xy = ln.get_xydata()
                t = np.linspace(0, len(xy) - 1, 400)
                xs = np.exp(np.interp(t, np.arange(len(xy)), np.log(xy[:, 0])))
                ys = np.interp(np.log(xs), np.log(xy[:, 0]), xy[:, 1])
                p = ax.transData.transform(np.c_[xs, ys])
                r = (ln.get_linewidth() + ln.get_markersize()) * fig.dpi / 72 / 2
                if np.any((p[:, 0] > bb.x0 - r) & (p[:, 0] < bb.x1 + r) & (p[:, 1] > bb.y0 - r) & (p[:, 1] < bb.y1 + r)):
                    hits.append(f"line {ln.get_label()}")
            grid = np.c_[[(x, y) for x in np.linspace(bb.x0, bb.x1, 12) for y in np.linspace(bb.y0, bb.y1, 8)]]
            for coll in ax.collections:
                if any(p.transformed(ax.transData).contains_points(grid).any() for p in coll.get_paths()):
                    hits.append("band")
        if hits:
            raise SystemExit(f"reference-line label {lab.get_text()!r} overlaps: {sorted(set(hits))}")
        print(f"label {lab.get_text()!r}: x {bb.x0 * pt:.1f}-{bb.x1 * pt:.1f} pt, y {bb.y0 * pt:.1f}-{bb.y1 * pt:.1f} pt; "
              f"clearance {(bb.x0 - a0.x1) * pt:.1f} pt to the left panel, {(a1.x0 - tick - bb.x1) * pt:.1f} pt to the "
              f"right panel's tick tips; no overlap")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coverage", nargs="+", default=[os.path.join(ROOT, "results", "g-j2", "09_coverage.tsv"),
                                                      os.path.join(ROOT, "results", "g-j2", "09_coverage_seed1.tsv")],
                    help="coverage TSVs, one per synthetic judge set (labelled A, B)")
    ap.add_argument("--power", default=os.path.join(ROOT, "results", "g-j2", "10_power.tsv"))
    ap.add_argument("--targets", type=float, nargs="+", default=[0.0, 0.05, 0.10],
                    help="delta-tau groups to draw in the power figure (at most 3)")
    ap.add_argument("--real-coverage", default=os.path.join(ROOT, "results", "g-j2", "15_real_judge_coverage.tsv"),
                    help="real-judge coverage, topics drawn without replacement (coverage_without_replacement.pdf)")
    ap.add_argument("--real-coverage-matched", default=os.path.join(ROOT, "results", "g-j2", "15_real_judge_coverage_matched.tsv"),
                    help="real-judge coverage, matched design (coverage.pdf, Figure 2)")
    ap.add_argument("--real-power", nargs=2, default=[os.path.join(ROOT, "results", "g-j2", "16_real_judge_power.tsv"),
                                                     os.path.join(ROOT, "results", "g-j2", "16_real_judge_power_bootstrap.tsv")],
                    help="real-judge power TSVs: permutation test, bootstrap z-test")
    ap.add_argument("--outdir", default=os.path.join(PAPER, "manuscript", "figures"))
    ap.add_argument("--only", nargs="*", choices=["coverage_synthetic", "power_synthetic", "coverage_without_replacement",
                                                  "coverage", "power"],
                    help="write only these figures (default: all)")
    args = ap.parse_args()
    want = (lambda name: not args.only or name in args.only)
    assert len(args.targets) <= 3, "at most three groups: the validated palette has three all-pairs slots"
    os.makedirs(args.outdir, exist_ok=True)
    if want("coverage_synthetic"):
        coverage_figure(args.coverage, os.path.join(args.outdir, "coverage_synthetic.pdf"))
    if want("power_synthetic"):
        power_figure(args.power, os.path.join(args.outdir, "power_synthetic.pdf"), args.targets)
    if want("coverage_without_replacement"):
        real_coverage_figure(args.real_coverage, os.path.join(args.outdir, "coverage_without_replacement.pdf"))
    if want("coverage"):
        matched_coverage_figure(args.real_coverage_matched, os.path.join(args.outdir, "coverage.pdf"))
    if want("power"):
        real_power_figure(*args.real_power, os.path.join(args.outdir, "power.pdf"))


if __name__ == "__main__":
    main()
