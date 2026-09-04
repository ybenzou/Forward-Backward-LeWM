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
from matplotlib.lines import Line2D
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
K_FIXED = (
    ("k0", 0),
    ("k5", 5),
    ("k10", 10),
    ("k15", 15),
)
# Internal key "dynamic" is HAS. Table still lists fixed k, then HAS.
CONDS = (
    ("k0", r"$k{=}0$", PALETTE["red_strong"]),
    ("k5", r"$k{=}5$", "#E9A6A1"),
    ("k10", r"$k{=}10$", PALETTE["teal"]),
    ("k15", r"$k{=}15$", "#767676"),
    ("dynamic", "HAS", PALETTE["blue_main"]),
)
OFFSET_STYLE = (
    (75, PALETTE["red_strong"], "o", "-"),
    (100, PALETTE["teal"], "s", "--"),
)
HAS_X = {75: 17.4, 100: 18.6}


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


def _mu_sd(xs: list[float]) -> tuple[float, float]:
    arr = np.array(xs, dtype=float)
    if len(arr) == 0:
        return float("nan"), 0.0
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return mean_std(arr)


def plot_bars(data: dict, dest: Path) -> None:
    apply_publication_style(
        FigureStyle(
            font_size=10,
            axes_linewidth=1.0,
            font_family=("Liberation Sans", "DejaVu Sans", "sans-serif"),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.55), sharey=True)
    rng = np.random.default_rng(0)
    ks = np.array([k for _, k in K_FIXED], dtype=float)
    handles: list = []
    for ax, task in zip(axes, TASKS):
        ax.axvline(16.2, color="#CFCECE", lw=0.8, ls=":", zorder=0)
        for o, color, marker, ls in OFFSET_STYLE:
            mus, sds = [], []
            for name, k in K_FIXED:
                xs = data[task][name][o]
                mu, sd = _mu_sd(xs)
                mus.append(mu)
                sds.append(sd)
                if xs:
                    jitter = rng.uniform(-0.45, 0.45, size=len(xs))
                    ax.scatter(
                        np.full(len(xs), k) + jitter,
                        np.array(xs, dtype=float),
                        s=11,
                        color=color,
                        alpha=0.28,
                        linewidths=0,
                        zorder=2,
                    )
            mus = np.array(mus)
            sds = np.array(sds)
            line = ax.plot(
                ks,
                mus,
                color=color,
                lw=1.6,
                ls=ls,
                marker=marker,
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=rf"$o{{=}}{o}$",
                zorder=3,
            )[0]
            ax.fill_between(ks, mus - sds, mus + sds, color=color, alpha=0.10, linewidth=0)
            has_xs = data[task]["dynamic"][o]
            has_mu, _ = _mu_sd(has_xs)
            hx = HAS_X[o]
            ax.scatter(
                [hx],
                [has_mu],
                s=36,
                color=color,
                marker="D",
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            if has_xs:
                jitter = rng.uniform(-0.22, 0.22, size=len(has_xs))
                ax.scatter(
                    np.full(len(has_xs), hx) + jitter,
                    np.array(has_xs, dtype=float),
                    s=11,
                    color=color,
                    alpha=0.28,
                    linewidths=0,
                    zorder=2,
                )
            if ax is axes[0]:
                handles.append(line)
        ax.set_title(TASK_TITLES[task], fontsize=10, color="#2B2B2B", pad=4)
        ax.set_xlim(-1.2, 19.4)
        ax.set_xticks([0, 5, 10, 15, 18.0], ["0", "5", "10", "15", "HAS"])
        ax.set_ylim(-2, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(length=2.5)
        if ax is axes[1]:
            ax.set_xlabel(r"Fixed Forward depth $k$")
    axes[0].set_ylabel("Success rate (%)")
    has_handle = Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor="#4D4D4D",
        markeredgecolor="white",
        markersize=6,
        linestyle="None",
    )
    fig.legend(
        [*handles, has_handle],
        [r"$o{=}75$", r"$o{=}100$", "HAS"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.54, 1.02),
        handlelength=1.6,
        columnspacing=1.4,
        borderaxespad=0.15,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.18, top=0.82, wspace=0.14)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        out = dest.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
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
