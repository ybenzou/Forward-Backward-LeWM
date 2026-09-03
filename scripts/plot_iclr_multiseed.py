#!/usr/bin/env python3
"""Publication figures for ICLR-bar 10-seed Official vs HAS vs long-CEM LeWM.

Reads outputs/diag/iclr_bar/multiseed/{task}/seed_*/summary.txt
  and outputs/diag/iclr_bar/longcem/h10/{task}/seed_*/summary.txt
Writes outputs/figures/fig_iclr_multiseed.png
     and paper/figures/fig_iclr_multiseed.png
     plus the matching delta figures.
Does not overwrite fig_*_pred_goal*.png or fig_*_backward_zz.png.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MULTI_ROOT = ROOT / "outputs" / "diag" / "iclr_bar" / "multiseed"
LONGCEM_ROOT = ROOT / "outputs" / "diag" / "iclr_bar" / "longcem"
FIG_DIR = ROOT / "outputs" / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"
OFFSETS = (25, 50, 75, 100)
TASKS = ("pusht", "tworoom", "reacher")
LONGCEM_TASKS = ("pusht", "tworoom")
OPTIONAL_LONGCEM_TASKS = ("reacher",)
TASK_TITLES = {"pusht": "PushT", "tworoom": "TwoRoom", "reacher": "Reacher"}
SEEDS = tuple(range(42, 52))
LONGCEM_HORIZON = 10

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "red_2": "#E9A6A1",
    "teal": "#42949E",
    "neutral": "#CFCECE",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    font_family: tuple[str, ...] = ("DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    style = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.family": list(style.font_family),
            "font.size": style.font_size,
            "axes.titlesize": style.font_size + 2,
            "axes.labelsize": style.font_size,
            "xtick.labelsize": style.font_size - 1,
            "ytick.labelsize": style.font_size - 1,
            "legend.fontsize": style.font_size - 1,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": style.axes_linewidth,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=0.35):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if formats is None:
        formats = [out_path.suffix.lstrip(".") or "png"]
    saved = []
    fig.tight_layout(pad=pad)
    for fmt in formats:
        dest = out_path.with_suffix(f".{fmt}")
        fig.savefig(dest, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        saved.append(dest)
        print(f"wrote {dest}")
    if close:
        plt.close(fig)
    return saved


def load_matrix(root: Path) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """task -> mode -> offset -> (n_seeds,) success rates."""
    out: dict[str, dict[str, dict[int, dict[int, float]]]] = {}
    n_files = 0
    for path in sorted(root.glob("*/seed_*/summary.txt")):
        n_files += 1
        task = path.parts[-3]
        seed = int(path.parts[-2].split("_")[1])
        for line in path.read_text().splitlines():
            if not line.startswith("mode="):
                continue
            parts = dict(x.split("=", 1) for x in line.split() if "=" in x)
            mode = parts["mode"]
            offset = int(parts["offset"])
            out.setdefault(task, {}).setdefault(mode, {}).setdefault(offset, {})[
                seed
            ] = float(parts["success_rate"])
    expected = len(TASKS) * len(SEEDS)
    if n_files != expected:
        raise SystemExit(
            f"expected {expected} summary.txt, found {n_files} under {root}"
        )
    stacked: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for task in TASKS:
        if task not in out:
            raise SystemExit(f"missing task {task}")
        stacked[task] = {}
        for mode in ("official", "forward"):
            stacked[task][mode] = {}
            for offset in OFFSETS:
                by_seed = out[task][mode][offset]
                missing = [s for s in SEEDS if s not in by_seed]
                if missing:
                    raise SystemExit(f"{task} {mode} offset={offset} missing {missing}")
                stacked[task][mode][offset] = np.array(
                    [by_seed[s] for s in SEEDS], dtype=float
                )
    return stacked


def load_longcem_official(
    root: Path, horizon: int = LONGCEM_HORIZON
) -> dict[str, dict[int, np.ndarray]]:
    """task -> offset -> (n_seeds,) Official success at the given CEM horizon."""
    sub = root / f"h{horizon}"
    raw: dict[str, dict[int, dict[int, float]]] = {}
    n_files = 0
    for path in sorted(sub.glob("*/seed_*/summary.txt")):
        n_files += 1
        task = path.parts[-3]
        seed = int(path.parts[-2].split("_")[1])
        for line in path.read_text().splitlines():
            if not line.startswith("mode="):
                continue
            parts = dict(x.split("=", 1) for x in line.split() if "=" in x)
            if parts.get("mode") != "official":
                continue
            offset = int(parts["offset"])
            raw.setdefault(task, {}).setdefault(offset, {})[seed] = float(
                parts["success_rate"]
            )
    expected = len(LONGCEM_TASKS) * len(SEEDS)
    n_kept = sum(
        1
        for path in sub.glob("*/seed_*/summary.txt")
        if path.parts[-3] in LONGCEM_TASKS
    )
    if n_kept != expected:
        raise SystemExit(
            f"expected {expected} longcem h{horizon} summary.txt for "
            f"{LONGCEM_TASKS}, found {n_kept} under {sub}. "
            "Run: python scripts/run_iclr_longcem.py --horizons 10"
        )
    stacked: dict[str, dict[int, np.ndarray]] = {}
    for task in LONGCEM_TASKS:
        if task not in raw:
            raise SystemExit(f"missing longcem task {task} under {sub}")
        stacked[task] = {}
        for offset in OFFSETS:
            by_seed = raw[task].get(offset, {})
            missing = [s for s in SEEDS if s not in by_seed]
            if missing:
                raise SystemExit(
                    f"longcem h{horizon} {task} offset={offset} missing {missing}"
                )
            stacked[task][offset] = np.array([by_seed[s] for s in SEEDS], dtype=float)
    for task in OPTIONAL_LONGCEM_TASKS:
        if task not in raw:
            continue
        complete = True
        arrays: dict[int, np.ndarray] = {}
        for offset in OFFSETS:
            by_seed = raw[task].get(offset, {})
            if any(s not in by_seed for s in SEEDS):
                complete = False
                break
            arrays[offset] = np.array([by_seed[s] for s in SEEDS], dtype=float)
        if complete:
            stacked[task] = arrays
    return stacked


def mean_std(xs: np.ndarray) -> tuple[float, float]:
    return float(xs.mean()), float(xs.std(ddof=1))


def aggregate(data, longcem=None) -> dict:
    rec: dict = {"n_seeds": len(SEEDS), "seeds": list(SEEDS), "n_eval": 50, "tasks": {}}
    for task in TASKS:
        rec["tasks"][task] = {}
        for mode in ("official", "forward"):
            rec["tasks"][task][mode] = {}
            for off in OFFSETS:
                xs = data[task][mode][off]
                mu, sd = mean_std(xs)
                rec["tasks"][task][mode][str(off)] = {
                    "mean": round(mu, 2),
                    "std": round(sd, 2),
                    "min": float(xs.min()),
                    "max": float(xs.max()),
                    "seeds": [float(v) for v in xs],
                }
        rec["tasks"][task]["delta"] = {}
        for off in OFFSETS:
            d = data[task]["forward"][off] - data[task]["official"][off]
            mu, sd = mean_std(d)
            rec["tasks"][task]["delta"][str(off)] = {
                "mean": round(mu, 2),
                "std": round(sd, 2),
                "min": float(d.min()),
                "max": float(d.max()),
                "seeds": [float(v) for v in d],
            }
        if longcem is not None and task in longcem:
            rec["tasks"][task]["official_h10"] = {}
            rec["tasks"][task]["delta_vs_h10"] = {}
            for off in OFFSETS:
                xs = longcem[task][off]
                mu, sd = mean_std(xs)
                rec["tasks"][task]["official_h10"][str(off)] = {
                    "mean": round(mu, 2),
                    "std": round(sd, 2),
                    "min": float(xs.min()),
                    "max": float(xs.max()),
                    "seeds": [float(v) for v in xs],
                }
                d = data[task]["forward"][off] - xs
                mu, sd = mean_std(d)
                rec["tasks"][task]["delta_vs_h10"][str(off)] = {
                    "mean": round(mu, 2),
                    "std": round(sd, 2),
                    "min": float(d.min()),
                    "max": float(d.max()),
                    "seeds": [float(v) for v in d],
                }
    return rec


def _series(data, task: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([mean_std(data[task][mode][o])[0] for o in OFFSETS])
    stds = np.array([mean_std(data[task][mode][o])[1] for o in OFFSETS])
    return means, stds


def plot_success(data, dest: Path, longcem=None) -> None:
    apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.0))
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.55), sharey=True)
    x = np.array(OFFSETS, dtype=float)
    rng = np.random.default_rng(0)
    panel = {"pusht": "a", "tworoom": "b", "reacher": "c"}
    series = [
        ("official", PALETTE["red_strong"], r"LeWM ($H{=}25$)", 2, "s", "-"),
        ("forward", PALETTE["blue_main"], r"HAS ($H{=}25$)", 4, "o", "-"),
    ]
    if longcem is not None:
        series.insert(
            1,
            ("official_h10", PALETTE["teal"], r"Long-CEM ($H{=}50$)", 3, "^", "--"),
        )

    handles = labels = None
    for ax, task in zip(axes, TASKS):
        for key, color, label, z, marker, ls in series:
            if key == "official_h10":
                if longcem is None or task not in longcem:
                    continue
                mu = np.array([mean_std(longcem[task][o])[0] for o in OFFSETS])
                sd = np.array([mean_std(longcem[task][o])[1] for o in OFFSETS])
                per_off = longcem[task]
            else:
                mu, sd = _series(data, task, key)
                per_off = data[task][key]
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.14, linewidth=0)
            ax.plot(
                x,
                mu,
                color=color,
                lw=2.6,
                ls=ls,
                marker=marker,
                markersize=8.5,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=label,
                zorder=z,
            )
            for off in OFFSETS:
                ys = per_off[off]
                jitter = rng.uniform(-2.2, 2.2, size=len(ys))
                ax.scatter(
                    np.full(len(ys), off) + jitter,
                    ys,
                    s=16,
                    color=color,
                    alpha=0.24,
                    linewidths=0,
                    zorder=z - 1,
                )

        ax.set_title(f"({panel[task]})  {TASK_TITLES[task]}", pad=8)
        ax.set_xlabel(r"Goal offset $o$")
        ax.set_xticks(list(OFFSETS))
        ax.set_xlim(17, 108)
        ax.set_ylim(-2, 104)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.axvline(25, color=PALETTE["neutral"], lw=1.0, ls="--", zorder=0)
        if task == "pusht":
            ax.set_ylabel("Success rate (%)")
            ax.text(
                29.5,
                99.5,
                r"$k=0$",
                fontsize=11,
                color="#767676",
                va="top",
                ha="left",
            )
        handles, labels = ax.get_legend_handles_labels()

    ncol = 3 if longcem is not None else 2
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.54, 1.03),
        handlelength=2.2,
        columnspacing=1.4,
    )
    finalize_figure(fig, dest, formats=["png", "pdf", "svg"], dpi=300, pad=1.15)


def plot_delta(data, dest: Path, longcem=None) -> None:
    apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.0))
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.55), sharey=False)
    x = np.arange(len(OFFSETS), dtype=float)
    rng = np.random.default_rng(1)
    panel = {"pusht": "a", "tworoom": "b", "reacher": "c"}
    grouped = longcem is not None
    width = 0.36 if grouped else 0.62

    for ax, task in zip(axes, TASKS):
        series = [("official", PALETTE["blue_main"], r"HAS $-$ LeWM ($H{=}25$)")]
        if grouped and longcem is not None and task in longcem:
            series.append(("official_h10", PALETTE["teal"], r"HAS $-$ Long-CEM ($H{=}50$)"))
        n_s = len(series)
        for s_i, (key, color, label) in enumerate(series):
            means = []
            stds = []
            shift = (s_i - (n_s - 1) / 2.0) * width if grouped else 0.0
            for j, off in enumerate(OFFSETS):
                base = (
                    longcem[task][off] if key == "official_h10" else data[task]["official"][off]
                )
                d = data[task]["forward"][off] - base
                mu, sd = mean_std(d)
                means.append(mu)
                stds.append(sd)
                jitter = rng.uniform(-0.06, 0.06, size=len(d))
                ax.scatter(
                    np.full(len(d), x[j] + shift) + jitter,
                    d,
                    s=18,
                    color=color,
                    alpha=0.28,
                    linewidths=0,
                    zorder=3,
                )
            means = np.array(means)
            stds = np.array(stds)
            ax.bar(
                x + shift,
                means,
                width=width * 0.92,
                color=color,
                edgecolor="black",
                linewidth=1.15,
                zorder=2,
                label=label,
            )
            ax.errorbar(
                x + shift,
                means,
                yerr=stds,
                fmt="none",
                ecolor="#272727",
                elinewidth=1.3,
                capsize=3.4,
                capthick=1.3,
                zorder=4,
            )
            for xi, mu, sd in zip(x + shift, means, stds):
                ax.text(
                    xi,
                    mu + sd + (3.6 if task == "tworoom" else 2.0),
                    f"{mu:+.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=color,
                )
        ax.axhline(0.0, color="#4D4D4D", lw=1.15, zorder=1)
        ax.set_title(f"({panel[task]})  {TASK_TITLES[task]}", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(o) for o in OFFSETS])
        ax.set_xlabel(r"Goal offset $o$")
        if task == "pusht":
            ax.set_ylabel("Success gap (pp)")
            ax.set_ylim(-18, 44)
            ax.set_yticks([-10, 0, 10, 20, 30, 40])
            ax.legend(loc="upper right", handlelength=1.2)
        elif task == "reacher":
            ax.set_ylim(-10, 28)
            ax.set_yticks([-10, 0, 10, 20])
        else:
            ax.set_ylim(-18, 86)
            ax.set_yticks([0, 20, 40, 60, 80])

    finalize_figure(fig, dest, formats=["png", "pdf", "svg"], dpi=300, pad=1.15)


def main() -> int:
    data = load_matrix(MULTI_ROOT)
    longcem = load_longcem_official(LONGCEM_ROOT, LONGCEM_HORIZON)
    stats = aggregate(data, longcem=longcem)
    stats_path = MULTI_ROOT / "aggregate.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {stats_path}")
    long_stats = LONGCEM_ROOT / "h10" / "aggregate.json"
    long_stats.parent.mkdir(parents=True, exist_ok=True)
    long_stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {long_stats}")
    sidecar = FIG_DIR / "fig_iclr_multiseed_stats.json"
    sidecar.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {sidecar}")

    print("\nmean ± std  (n=10 seeds, 50 episodes each)")
    for task in TASKS:
        print(f"  {TASK_TITLES[task]}")
        for mode, name in (
            ("official", "LeWM H=25"),
            ("forward", "HAS  H=25"),
        ):
            bits = []
            for off in OFFSETS:
                mu, sd = mean_std(data[task][mode][off])
                bits.append(f"{off}:{mu:.1f}±{sd:.1f}")
            print(f"    {name:10}  " + "  ".join(bits))
        if longcem is not None and task in longcem:
            bits = []
            for off in OFFSETS:
                mu, sd = mean_std(longcem[task][off])
                bits.append(f"{off}:{mu:.1f}±{sd:.1f}")
            print(f"    {'Long-CEM':10}  " + "  ".join(bits))
        bits = []
        for off in OFFSETS:
            d = data[task]["forward"][off] - data[task]["official"][off]
            mu, sd = mean_std(d)
            bits.append(f"{off}:{mu:+.1f}±{sd:.1f}")
            print(f"    {'Δ vs H=25':10}  " + "  ".join(bits))
        if longcem is not None and task in longcem:
            bits = []
            for off in OFFSETS:
                d = data[task]["forward"][off] - longcem[task][off]
                mu, sd = mean_std(d)
                bits.append(f"{off}:{mu:+.1f}±{sd:.1f}")
            print(f"    {'Δ vs H=50':10}  " + "  ".join(bits))

    for dest_dir in (FIG_DIR, PAPER_FIG_DIR):
        plot_success(data, dest_dir / "fig_iclr_multiseed", longcem=longcem)
        plot_delta(data, dest_dir / "fig_iclr_multiseed_delta", longcem=longcem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
