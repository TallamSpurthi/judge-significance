"""Step 36. Figure: the eight new judges' tau on the 25 LLMJudge topics and on the 57 held-out DL 2023 topics.

Reads step 29's results/llm-judges/29_power_bootstrap.tsv (each pair row carries both judges' tau on each topic
pool; no new simulation) and step 32's resolution at 200 topics, and writes
  results/llm-judges/36_tau_by_pool.tsv   judge, tau on the 25 LLMJudge topics, tau on the held-out topics
  heldout.pdf                             a slope chart of those values (not ranks), each line labelled at both
                                          ends, with a bar as long as the bootstrap test's resolution at 200
                                          topics at nominal alpha
No per-judge interval is drawn: the paper finds single-judge bootstrap intervals for tau under-cover (step 15).
Paper style of step 12 (Linux Biolinum, black text, fixed page size, byte-stable reruns).

Usage: python scripts/36_heldout_figure.py [--power results/llm-judges/29_power_bootstrap.tsv]
           [--resolution results/g-j2/32_size_adjusted_resolution.tsv] [--tsv results/llm-judges/36_tau_by_pool.tsv]
           [--out paper/manuscript/figures/heldout.pdf]

--width WIDE (inches) lays the figure out from the measured label widths, so that no label touches the y axis or the
resolution bar, and refuses to save if one does. Without it the figure is the original 3.33 in column version. The
ACM AI Letters version:
       python scripts/36_heldout_figure.py --width 4.2 --out paper/venues/acm-ai-letters/figures/heldout.pdf
"""
import argparse
import importlib.util
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.environ.get("PAPER_ROOT") or os.path.join(ROOT, "paper")  # where generated paper files go (README, Usage)
_spec12 = importlib.util.spec_from_file_location("fig12", os.path.join(HERE, "12_make_figures.py"))
_fig12 = importlib.util.module_from_spec(_spec12)
_spec12.loader.exec_module(_fig12)

LABELS = {  # the paper's names: Haiku with three prompts, the other models with UMBRELA
    "haiku-umbrela": "Haiku, UMBRELA", "haiku-basic": "Haiku, basic", "haiku-llmjudge": "Haiku, LLMJudge",
    "sonnet-umbrela": "Sonnet", "gemma4b-umbrela": "Gemma 3 4B", "llama8b-umbrela": "Llama 3.1 8B",
    "scout-umbrela": "Llama 4 Scout", "qwen30b-umbrela": "Qwen3 30B",
}
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]


def tau_by_pool(path, pools=("test", "heldout")):
    """{pool: {judge: tau}} from the pair rows; every row must agree on a judge's tau within a pool."""
    seen = {}
    for r in _fig12.read_tsv(path):
        if r["pool"] not in pools:
            continue
        for j, t in ((r["judge_a"], r["tau_a"]), (r["judge_b"], r["tau_b"])):
            seen.setdefault((r["pool"], j), set()).add(float(t))
    bad = {k: v for k, v in seen.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"inconsistent tau across rows: {bad}")
    judges = sorted({j for _, j in seen})
    return {p: {j: next(iter(seen[(p, j)])) for j in judges} for p in pools}


def spread(values, gap):
    """Label positions: the values, pushed apart to at least `gap` while keeping their order (centred shifts)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for _ in range(200):
        moved = False
        for k in range(1, len(pos)):
            d = pos[k] - pos[k - 1]
            if d < gap - 1e-12:
                shift = (gap - d) / 2
                pos[k - 1] -= shift
                pos[k] += shift
                moved = True
        if not moved:
            break
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def measured_layout(fig, ax, left, right, note, size):
    """Bar position and x limits (data units) that leave 6 pt between the labels, the y axis and the bar."""
    r = fig.canvas.get_renderer()
    w = lambda t: t.get_window_extent(r).width * 72 / fig.dpi  # noqa: E731
    probe = ax.text(0, 0, note, fontsize=size, linespacing=1.1)
    wl, wr, wn = max(map(w, left)), max(map(w, right)), w(probe)
    probe.remove()
    gap = 6.0
    axis_pt = fig.get_figwidth() * 72 - 38.0  # the y label and tick labels take about 38 pt on the left
    fixed = wl + wr + wn + 3 * gap + 2.0      # three text widths, their gaps, and a margin at the right edge
    unit = (axis_pt - fixed) / (1 + 0.11 + 0.11 + 0.09)  # points per data unit
    if unit < 30:
        raise SystemExit(f"figure too narrow: {unit:.0f} pt left for the slopes")
    bx = 1 + 0.11 + (wr + gap) / unit
    return bx, (-0.11 - (wl + gap) / unit, bx + 0.09 + (wn + 2.0) / unit)


def check_layout(fig, ax, left, right, bar, note):
    """Refuse to save if a label touches the y axis, the resolution bar, the bar's note or the figure edge."""
    fig.canvas.draw()
    pad = 2.0 * fig.dpi / 72
    axis_x, bar_x = ax.get_window_extent().x0, ax.transData.transform((bar.get_xdata()[0], 0))[0]
    cap = ax.transData.transform((bar.get_xdata()[0] + 0.05, 0))[0] - bar_x
    for t in left:
        if t.get_window_extent().x0 < axis_x + pad:
            raise SystemExit(f"label touches the y axis: {t.get_text()}")
    for t in right:
        if t.get_window_extent().x1 > bar_x - cap - pad:
            raise SystemExit(f"label touches the resolution bar: {t.get_text()}")
    if note.get_window_extent().x0 < bar_x + pad or note.get_window_extent().x1 > fig.bbox.x1:
        raise SystemExit("the bar's note touches the bar or runs off the figure")
    print("labels clear of the y axis, the resolution bar and the figure edge")


def draw(T, res200, out, width=None):
    judges = sorted(T["test"], key=lambda j: -T["test"][j])
    with _fig12.paper_style():
        fig, ax = plt.subplots(figsize=(width or _fig12.WIDTH, 2.35))
        size = _fig12.PAPER_SIZE["small"]
        lo = min(min(T["test"].values()), min(T["heldout"].values()))
        hi = max(max(T["test"].values()), max(T["heldout"].values()))
        pad = max(0.0, (res200 - (hi - lo)) / 2) + 0.004  # the axis must hold the whole resolution bar
        ylo, yhi = lo - pad, hi + pad
        gap = 0.0068 * (yhi - ylo) / 0.12  # label spacing scaled to the axis range
        for k, j in enumerate(judges):
            ax.plot([0, 1], [T["test"][j], T["heldout"][j]], color="#3a3a38", lw=0.8, zorder=2)
            ax.plot([0, 1], [T["test"][j], T["heldout"][j]], ls="none", marker=MARKERS[k], ms=3.6, mew=0.6,
                    mec=_fig12.PAPER_BLACK, mfc="white", zorder=3)
        left, right = [], []
        for side, x, ha, dx in (("test", 0, "right", -0.11), ("heldout", 1, "left", 0.11)):
            ys = spread([T[side][j] for j in judges], gap)
            for j, yl in zip(judges, ys):
                t = ax.text(x + dx, yl, LABELS.get(j, j), ha=ha, va="center", fontsize=size)
                (left if side == "test" else right).append(t)
                if abs(yl - T[side][j]) > 1e-9:  # leader line when the label moved
                    ax.plot([x + dx * 0.4, x + dx * 0.9], [T[side][j], yl], color="#9a9994", lw=0.4, zorder=1)
        bx, xlim = 2.1, (-1.1, 2.8)  # the resolution bar, centred on the axis range
        note = f"{res200:.2f}: resolution\nat 200 topics"
        if width:
            bx, xlim = measured_layout(fig, ax, left, right, note, size)
        mid = (ylo + yhi) / 2
        bar, = ax.plot([bx, bx], [mid - res200 / 2, mid + res200 / 2], color=_fig12.PAPER_BLACK, lw=1.4,
                       solid_capstyle="butt")
        for yy in (mid - res200 / 2, mid + res200 / 2):
            ax.plot([bx - 0.05, bx + 0.05], [yy, yy], color=_fig12.PAPER_BLACK, lw=0.8)
        note = ax.text(bx + 0.09, mid, note, ha="left", va="center", fontsize=size, linespacing=1.1)
        ax.set_xlim(*xlim)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["25 LLMJudge\ntopics", "57 held-out\ntopics"], fontsize=_fig12.PAPER_SIZE["tick"])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(r"Kendall's $\tau$", fontsize=_fig12.PAPER_SIZE["tick"])
        ax.tick_params(axis="y", labelsize=size)
        ax.spines["bottom"].set_visible(False)
        fig.tight_layout(pad=0.3)
        if width:
            check_layout(fig, ax, left, right, bar, note)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        _fig12.save_paper_figure(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--power", default=os.path.join(ROOT, "results", "llm-judges", "29_power_bootstrap.tsv"))
    ap.add_argument("--resolution", default=os.path.join(ROOT, "results", "g-j2", "32_size_adjusted_resolution.tsv"))
    ap.add_argument("--tsv", default=os.path.join(ROOT, "results", "llm-judges", "36_tau_by_pool.tsv"))
    ap.add_argument("--out", default=os.path.join(PAPER, "manuscript", "figures", "heldout.pdf"))
    ap.add_argument("--width", type=float, help="figure width, in; measured layout (AILET: 4.2)")
    args = ap.parse_args()
    T = tau_by_pool(args.power)
    res200 = next(float(r["nominal_res"]) for r in _fig12.read_tsv(args.resolution) if int(r["n"]) == 200)
    judges = sorted(T["test"])
    flips = sum((T["test"][a] - T["test"][b]) * (T["heldout"][a] - T["heldout"][b]) < 0
                for a, b in itertools.combinations(judges, 2))
    first = max(judges, key=lambda j: T["heldout"][j])
    last = min(judges, key=lambda j: T["test"][j])
    print(f"{len(judges)} judges; tau on the 25 topics {min(T['test'].values()):.4f}-{max(T['test'].values()):.4f}, "
          f"held-out {min(T['heldout'].values()):.4f}-{max(T['heldout'].values()):.4f}; "
          f"{flips} of {len(judges) * (len(judges) - 1) // 2} pairwise orders reverse; "
          f"last on the 25 topics ({last}) is first held out: {first == last}; resolution bar {res200:.4f}")
    with open(args.tsv, "w") as f:
        f.write("# tau of each new judge on the 25 LLMJudge topics (test) and the 57 held-out topics, from step 29\n")
        f.write("judge\ttau_test\ttau_heldout\n")
        for j in sorted(judges, key=lambda j: -T["test"][j]):
            f.write(f"{j}\t{T['test'][j]:.6f}\t{T['heldout'][j]:.6f}\n")
    print(f"wrote {os.path.relpath(args.tsv, ROOT)}")
    draw(T, res200, args.out, args.width)


if __name__ == "__main__":
    main()
