"""
Figures for the write-up, generated from committed results only.

No DATA value is typed into this file. Every quantity plotted or annotated is
read from results/study/summary.json, results/ceiling/ceiling.json and
results/stride/stride.json, so a figure cannot drift from the evidence it claims
to show. Styling constants - sizes, colours, line widths - are of course
hardcoded; the distinction is that none of them is a measurement.

Output: docs/figures/*.pdf (vector, for LaTeX) and *.png (for markdown)
Run:    .venv/bin/python make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

HORIZONS = (5, 10, 20)
PROTOCOLS = ("random_split", "chronological", "purged_embargoed")
LABEL = {
    "random_split": "random split",
    "chronological": "chronological",
    "purged_embargoed": "purged + embargoed",
}
# One accent for the leaky protocol, greys for the honest ones. The figure
# should make the reader look at the same thing the argument does.
COLOUR = {"random_split": "#c0392b", "chronological": "#7f8c8d", "purged_embargoed": "#2c3e50"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 160,
    }
)


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote docs/figures/{name}.pdf / .png")


def fig_excess_over_baseline() -> None:
    """Main result: only the leaky protocol clears its own baseline."""
    cells = json.loads((ROOT / "results/study/summary.json").read_text())["cells"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    width = 0.26

    for i, p in enumerate(PROTOCOLS):
        xs, ys, errs = [], [], []
        for j, h in enumerate(HORIZONS):
            c = cells[f"{p}|h{h}"]
            acc = c["val_acc"]["mean"]
            base = c["val_majority"]["mean"]
            lo = c["val_acc"].get("ci95_of_mean_low", acc)
            hi = c["val_acc"].get("ci95_of_mean_high", acc)
            xs.append(j + (i - 1) * width)
            ys.append(acc - base)
            errs.append([acc - lo, hi - acc])
        errs = list(zip(*errs))
        ax.bar(xs, ys, width, label=LABEL[p], color=COLOUR[p],
               yerr=errs, capsize=2, error_kw={"lw": 0.7})

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"h = {h}" for h in HORIZONS])
    ax.set_ylabel("accuracy − majority baseline")
    ax.set_title("Only the leaky protocol beats chance, and only at longer horizons",
                 fontsize=9, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.99, 0.03, "data with no signal · 20 seeds · 95% CI", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#666")
    save(fig, "fig1-excess-over-baseline")


def fig_stride() -> None:
    """The remedy: predicted ceiling and measured oracle collapse together."""
    rows = json.loads((ROOT / "results/stride/stride.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)

    for ax, h in zip(axes, HORIZONS):
        sel = sorted([r for r in rows if r["horizon"] == h], key=lambda r: r["stride"])
        s = [r["stride"] for r in sel]
        ax.plot(s, [r["ceiling"] for r in sel], "-", color="#2c3e50", lw=1.4,
                label="closed-form ceiling")
        ax.plot(s, [r["oracle_twin_acc"] for r in sel], "o", ms=3.5, color="#c0392b",
                label="measured oracle")
        ax.plot(s, [r["majority"] for r in sel], "--", color="#7f8c8d", lw=1.0,
                label="majority baseline")
        ax.axvline(h, color="#999", lw=0.8, ls=":")
        # axis-fraction coords, so it cannot fall outside the drawn range
        ax.annotate("stride = horizon", xy=(h, 0.97), xycoords=("data", "axes fraction"),
                    fontsize=6.5, rotation=90, ha="right", va="top", color="#666")
        ax.set_title(f"horizon = {h}", fontsize=9)
        ax.set_xlabel("stride")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 30])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    axes[0].set_ylabel("accuracy")
    # One legend for the figure, below the panels: inside any panel it collides
    # with the ceiling curve, which is the line the reader most needs to see.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.28)
    fig.suptitle("Leakage is predictable, and it vanishes once stride reaches the horizon",
                 fontsize=9.5, y=1.04)
    save(fig, "fig2-stride")


def fig_available_vs_taken() -> None:
    """Availability is not exploitation."""
    cells = json.loads((ROOT / "results/study/summary.json").read_text())["cells"]
    ceil = json.loads((ROOT / "results/ceiling/ceiling.json").read_text())
    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    xs = range(len(HORIZONS))
    base = [cells[f"random_split|h{h}"]["val_majority"]["mean"] for h in HORIZONS]
    cnn = [cells[f"random_split|h{h}"]["val_acc"]["mean"] for h in HORIZONS]
    top = [ceil["analytic"][f"h{h}"]["ceiling"] for h in HORIZONS]
    orc = [ceil["empirical"][f"random_split|h{h}"]["oracle_twin_acc"] for h in HORIZONS]

    ax.fill_between(xs, base, top, color="#c0392b", alpha=0.10,
                    label="leak made available by the split")
    ax.plot(xs, top, "-", color="#c0392b", lw=1.3, label="closed-form ceiling")
    ax.plot(xs, orc, "s", ms=4, color="#c0392b", label="oracle memoriser")
    ax.plot(xs, cnn, "o-", color="#2c3e50", lw=1.3, ms=4, label="trained CNN")
    ax.plot(xs, base, "--", color="#7f8c8d", lw=1.0, label="majority baseline")

    for i, h in enumerate(HORIZONS):
        frac = (cnn[i] - base[i]) / (top[i] - base[i])
        ax.annotate(f"{frac:.0%} taken" if frac > 0 else "none taken",
                    xy=(i, (cnn[i] + base[i]) / 2), fontsize=7,
                    ha="left" if i == 0 else "center",
                    xytext=(6 if i == 0 else 0, 0), textcoords="offset points",
                    color="#2c3e50")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"h = {h}" for h in HORIZONS])
    ax.set_ylabel("accuracy")
    ax.set_title("How much leak exists, and how much a model actually takes",
                 fontsize=9, pad=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.26)
    save(fig, "fig3-available-vs-taken")


if __name__ == "__main__":
    print("generating figures from committed results:")
    fig_excess_over_baseline()
    fig_stride()
    fig_available_vs_taken()
