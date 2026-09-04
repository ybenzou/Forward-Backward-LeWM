#!/usr/bin/env python3
"""Aggregate 10-group k-depth bars: k=0 / 5 / 10 / dynamic / 15.

k=0 is official LeWM and dynamic is HAS, both from the main multiseed
matrix (same starts). k=5, k=10, and k=15 come from k_ablation/multiseed/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from plot_iclr_multiseed import (  # noqa: E402
    MULTI_ROOT,
    PALETTE,
    SEEDS,
    TASK_TITLES,
    TASKS,
    FigureStyle,
    apply_publication_style,
    mean_std,
)

K_ROOT = ROOT / "outputs" / "diag" / "k_ablation" / "multiseed"
PREV = ROOT / "paper" / "interpretability_preview"
OFFSETS = (75, 100)
# Fixed depths first, then HAS. Internal key "dynamic" is the HAS rows.
CONDS = (
    ("k0", r"$k{=}0$", PALETTE["red_strong"]),
    ("k5", r"$k{=}5$", "#E9A6A1"),
    ("k10", r"$k{=}10$", PALETTE["teal"]),
    ("k15", r"$k{=}15$", "#767676"),
    ("dynamic", "HAS", PALETTE["blue_main"]),
)


def _rate_from_summary(path: Path, mode: str, offset: int) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if not line.startswith("mode="):
            continue
        parts = dict(x.split("=", 1) for x in line.split() if "=" in x)
        if parts.get("mode") == mode and int(parts["offset"]) == offset:
            return float(parts["success_rate"])
    return None


def collect() -> dict:
    out: dict[str, dict[str, dict[int, list[float]]]] = {}
    for task in TASKS:
        out[task] = {name: {o: [] for o in OFFSETS} for name, _, _ in CONDS}
        for seed in SEEDS:
            base = MULTI_ROOT / task / f"seed_{seed}" / "summary.txt"
            for o in OFFSETS:
                v0 = _rate_from_summary(base, "official", o)
                vd = _rate_from_summary(base, "forward", o)
                if v0 is not None:
                    out[task]["k0"][o].append(v0)
                if vd is not None:
                    out[task]["dynamic"][o].append(vd)
            for name in ("k5", "k10", "k15"):
                summary = K_ROOT / task / name / f"seed_{seed}" / "summary.txt"
                for o in OFFSETS:
                    v = _rate_from_summary(summary, "forward", o)
                    if v is not None:
                        out[task][name][o].append(v)
    return out


def write_table(data: dict, dest: Path) -> None:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Task & $o$ & $k{=}0$ & $k{=}5$ & $k{=}10$ & $k{=}15$ & HAS \\",
        r"\midrule",
    ]
    for i, task in enumerate(TASKS):
        for o in OFFSETS:
            cells = []
            for name, _, _ in CONDS:
                xs = data[task][name][o]
                if not xs:
                    cells.append("---")
                elif len(xs) == 1:
                    cells.append(f"{xs[0]:.1f}")
                else:
                    mu, sd = mean_std(np.array(xs, dtype=float))
                    cells.append(f"{mu:.1f}$\\pm${sd:.1f}")
            o_tex = r"$75$" if o == 75 else r"$100$"
            lines.append(f"{TASK_TITLES[task]} & {o_tex} & " + " & ".join(cells) + r" \\")
        if i != len(TASKS) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines))
    print(f"wrote {dest}")


def plot_bars(data: dict, dest: Path) -> None:
    apply_publication_style(
        FigureStyle(
            font_size=10,
            axes_linewidth=1.0,
            font_family=("Liberation Sans", "DejaVu Sans", "sans-serif"),
        )
    )
    fig = plt.figure(figsize=(7.4, 2.55))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.00, 0.22),
        hspace=0.42,
        left=0.08,
        right=0.995,
        top=0.90,
        bottom=0.04,
    )
    gs = outer[0].subgridspec(1, 3, wspace=0.12)
    axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i], sharey=axes[0] if axes else None)
        if i:
            ax.tick_params(labelleft=False)
        axes.append(ax)
    x = np.arange(len(OFFSETS))
    w = 0.15
    handles = []
    for ax, task in zip(axes, TASKS):
        for i, (name, lab, col) in enumerate(CONDS):
            mus, sds = [], []
            for o in OFFSETS:
                xs = np.array(data[task][name][o], dtype=float)
                if len(xs) == 0:
                    mus.append(np.nan)
                    sds.append(0.0)
                elif len(xs) == 1:
                    mus.append(float(xs[0]))
                    sds.append(0.0)
                else:
                    mu, sd = mean_std(xs)
                    mus.append(mu)
                    sds.append(sd)
            xpos = x + (i - 2.0) * w
            bar = ax.bar(
                xpos,
                mus,
                width=w,
                color=col,
                label=lab,
                edgecolor="white",
                linewidth=0.4,
                yerr=sds,
                error_kw={"ecolor": "#4D4D4D", "elinewidth": 0.8, "capsize": 2},
            )
            if ax is axes[0]:
                handles.append(bar)
        ax.set_xticks(x, [r"$o{=}75$", r"$o{=}100$"])
        ax.set_title(TASK_TITLES[task], fontsize=10, color="#2B2B2B", pad=4)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(length=2.5)
    axes[0].set_ylabel("Success rate (%)")
    leg_ax = fig.add_subplot(outer[1])
    leg_ax.set_axis_off()
    leg_ax.legend(
        handles,
        [lab for _, lab, _ in CONDS],
        loc="center",
        ncol=len(CONDS),
        handlelength=1.1,
        columnspacing=1.2,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        out = dest.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"wrote {out}")
    plt.close(fig)


def main() -> int:
    data = collect()
    payload = {
        task: {
            name: {str(o): [float(v) for v in data[task][name][o]] for o in OFFSETS}
            for name, _, _ in CONDS
        }
        for task in TASKS
    }
    PREV.mkdir(parents=True, exist_ok=True)
    (PREV / "k_depth_multiseed.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_table(data, PREV / "tab_k_depth_multiseed.tex")
    plot_bars(data, PREV / "fig_k_depth_multiseed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
