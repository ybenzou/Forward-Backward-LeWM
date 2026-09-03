#!/usr/bin/env python3
"""Aggregate paired CIs, depth diagnostics, and planner cost for the paper.

Reads the existing 10-group Official / HAS / Long-CEM summaries and
diagnostics. Writes JSON for writing lookup and a compact mechanism figure.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from plot_iclr_multiseed import (  # noqa: E402
    FIG_DIR,
    LONGCEM_ROOT,
    MULTI_ROOT,
    OFFSETS,
    PAPER_FIG_DIR,
    PALETTE,
    SEEDS,
    TASK_TITLES,
    TASKS,
    FigureStyle,
    apply_publication_style,
    finalize_figure,
    load_longcem_official,
    load_matrix,
    mean_std,
)

DEPTHS = (1, 5, 10, 15)
N_BOOT = 10_000
BOOT_SEED = 0


def bootstrap_mean_ci(xs: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    draws = rng.choice(xs, size=(n_boot, len(xs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(xs.mean()), float(lo), float(hi)


def parse_summaries(root: Path, mode_filter: str | None = None):
    rows = []
    for path in sorted(root.glob("*/seed_*/summary.txt")):
        task = path.parts[-3]
        seed = int(path.parts[-2].split("_")[1])
        for line in path.read_text().splitlines():
            if not line.startswith("mode="):
                continue
            parts = dict(x.split("=", 1) for x in line.split() if "=" in x)
            if mode_filter is not None and parts["mode"] != mode_filter:
                continue
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "mode": parts["mode"],
                    "offset": int(parts["offset"]),
                    "success_rate": float(parts["success_rate"]),
                    "seconds": float(parts["seconds"]),
                    "n_success": int(parts["n_success"].split("/")[0]),
                }
            )
    return rows


def collect_depth_stats(root: Path) -> dict:
    rec: dict[str, dict[str, list[float]]] = {}
    for path in sorted(root.glob("*/seed_*/diagnostics.json")):
        task = path.parts[-3]
        payload = json.loads(path.read_text())
        rec.setdefault(task, {"norm": {k: [] for k in DEPTHS}, "dist": {k: [] for k in DEPTHS}})
        for k in DEPTHS:
            rec[task]["norm"][k].append(float(payload["forward_norms"][str(k)]))
            rec[task]["dist"][k].append(float(payload["forward_dist_to_seed"][str(k)]))
    out = {}
    for task, payload in rec.items():
        out[task] = {}
        for key in ("norm", "dist"):
            out[task][key] = {}
            for k in DEPTHS:
                xs = np.array(payload[key][k], dtype=float)
                mu, sd = mean_std(xs)
                out[task][key][str(k)] = {
                    "mean": round(mu, 3),
                    "std": round(sd, 3),
                    "seeds": [float(v) for v in xs],
                }
    return out


def paired_report(has: np.ndarray, base: np.ndarray) -> dict:
    delta = has - base
    mu, lo, hi = bootstrap_mean_ci(delta)
    wins = int((delta > 0).sum())
    ties = int((delta == 0).sum())
    losses = int((delta < 0).sum())
    return {
        "mean": round(float(delta.mean()), 2),
        "std": round(float(delta.std(ddof=1)), 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "n_groups": int(len(delta)),
        "seeds": [float(v) for v in delta],
    }


def wallclock_report(rows: list[dict], mode: str) -> dict:
    out = {}
    for task in TASKS:
        out[task] = {}
        for off in OFFSETS:
            xs = [
                r["seconds"]
                for r in rows
                if r["task"] == task and r["mode"] == mode and r["offset"] == off
            ]
            if not xs:
                continue
            arr = np.array(xs, dtype=float)
            mu, sd = mean_std(arr)
            out[task][str(off)] = {
                "mean_s": round(mu, 1),
                "std_s": round(sd, 1),
            }
    return out


def plot_mechanism(stats: dict, dest: Path) -> None:
    apply_publication_style(FigureStyle(font_size=14, axes_linewidth=2.0))
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.35))
    x = np.arange(len(OFFSETS), dtype=float)
    width = 0.36

    for ax, task in zip(axes, TASKS):
        vs5 = [stats["tasks"][task]["delta_vs_lewm"][str(o)] for o in OFFSETS]
        means = np.array([r["mean"] for r in vs5])
        yerr = np.array(
            [
                [r["mean"] - r["ci95"][0] for r in vs5],
                [r["ci95"][1] - r["mean"] for r in vs5],
            ]
        )
        ax.bar(
            x - width / 2,
            means,
            width=width,
            color=PALETTE["blue_main"],
            edgecolor="black",
            linewidth=1.05,
            label=r"HAS $-$ LeWM ($H{=}25$)",
            zorder=2,
        )
        ax.errorbar(
            x - width / 2,
            means,
            yerr=yerr,
            fmt="none",
            ecolor="#272727",
            elinewidth=1.15,
            capsize=3.0,
            zorder=3,
        )
        if "delta_vs_longcem" in stats["tasks"][task]:
            vs10 = [stats["tasks"][task]["delta_vs_longcem"][str(o)] for o in OFFSETS]
            means10 = np.array([r["mean"] for r in vs10])
            yerr10 = np.array(
                [
                    [r["mean"] - r["ci95"][0] for r in vs10],
                    [r["ci95"][1] - r["mean"] for r in vs10],
                ]
            )
            ax.bar(
                x + width / 2,
                means10,
                width=width,
                color=PALETTE["teal"],
                edgecolor="black",
                linewidth=1.05,
                label=r"HAS $-$ Long-CEM ($H{=}50$)",
                zorder=2,
            )
            ax.errorbar(
                x + width / 2,
                means10,
                yerr=yerr10,
                fmt="none",
                ecolor="#272727",
                elinewidth=1.15,
                capsize=3.0,
                zorder=3,
            )
        ax.axhline(0.0, color="#4D4D4D", lw=1.05, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$o={o}$" for o in OFFSETS])
        ax.set_title(TASK_TITLES[task], pad=6)
        if task == "pusht":
            ax.set_ylabel("Paired success gap (pp)")
            ax.legend(loc="upper right", fontsize=10, handlelength=1.15)
            ax.set_ylim(-12, 42)
        elif task == "tworoom":
            ax.set_ylim(-12, 78)
        else:
            ax.set_ylim(-12, 52)
    finalize_figure(fig, dest, formats=["png", "pdf", "svg"], dpi=300, pad=0.9)


def plot_depth(depth: dict, dest: Path) -> None:
    apply_publication_style(FigureStyle(font_size=14, axes_linewidth=2.0))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.15))
    ks = np.array(DEPTHS, dtype=float)
    for task in TASKS:
        mu = np.array([depth[task]["dist"][str(k)]["mean"] for k in DEPTHS])
        sd = np.array([depth[task]["dist"][str(k)]["std"] for k in DEPTHS])
        axes[0].plot(
            ks,
            mu,
            marker="o",
            lw=2.3,
            label=TASK_TITLES[task],
        )
        axes[0].fill_between(ks, mu - sd, mu + sd, alpha=0.14, linewidth=0)
        nmu = np.array([depth[task]["norm"][str(k)]["mean"] for k in DEPTHS])
        nsd = np.array([depth[task]["norm"][str(k)]["std"] for k in DEPTHS])
        axes[1].plot(ks, nmu, marker="o", lw=2.3, label=TASK_TITLES[task])
        axes[1].fill_between(ks, nmu - nsd, nmu + nsd, alpha=0.14, linewidth=0)
    axes[0].set_title("Displacement from seed latent")
    axes[0].set_xlabel("Forward depth $k$")
    axes[0].set_ylabel(r"$\|F^{k}(z)-z\|_2$")
    axes[0].set_xticks(DEPTHS)
    axes[1].set_title("Imagined latent norm")
    axes[1].set_xlabel("Forward depth $k$")
    axes[1].set_ylabel(r"$\|F^{k}(z)\|_2$")
    axes[1].set_xticks(DEPTHS)
    axes[0].legend(loc="best", fontsize=11)
    finalize_figure(fig, dest, formats=["png", "pdf", "svg"], dpi=300, pad=0.9)


def main() -> int:
    data = load_matrix(MULTI_ROOT)
    longcem = load_longcem_official(LONGCEM_ROOT)
    multi_rows = parse_summaries(MULTI_ROOT)
    long_rows = parse_summaries(LONGCEM_ROOT / "h10", mode_filter="official")
    depth = collect_depth_stats(MULTI_ROOT)

    rec = {
        "n_groups": len(SEEDS),
        "groups": list(SEEDS),
        "n_eval": 50,
        "note": (
            "Groups 42-51 are paired evaluation starts, not training seeds. "
            "Each task uses one epoch-10 checkpoint."
        ),
        "tasks": {},
        "depth": depth,
        "wallclock": {
            "lewm_h25": wallclock_report(multi_rows, "official"),
            "has_h25": wallclock_report(multi_rows, "forward"),
            "longcem_h50": wallclock_report(long_rows, "official"),
        },
    }

    for task in TASKS:
        rec["tasks"][task] = {
            "delta_vs_lewm": {},
            "lewm": {},
            "has": {},
        }
        if task in longcem:
            rec["tasks"][task]["delta_vs_longcem"] = {}
            rec["tasks"][task]["longcem"] = {}
        for off in OFFSETS:
            has = data[task]["forward"][off]
            lewm = data[task]["official"][off]
            rec["tasks"][task]["has"][str(off)] = {
                "mean": round(float(has.mean()), 1),
                "std": round(float(has.std(ddof=1)), 1),
            }
            rec["tasks"][task]["lewm"][str(off)] = {
                "mean": round(float(lewm.mean()), 1),
                "std": round(float(lewm.std(ddof=1)), 1),
            }
            rec["tasks"][task]["delta_vs_lewm"][str(off)] = paired_report(has, lewm)
            if task in longcem:
                base = longcem[task][off]
                rec["tasks"][task]["longcem"][str(off)] = {
                    "mean": round(float(base.mean()), 1),
                    "std": round(float(base.std(ddof=1)), 1),
                }
                rec["tasks"][task]["delta_vs_longcem"][str(off)] = paired_report(
                    has, base
                )

    out_json = FIG_DIR / "fig_iclr_analysis_stats.json"
    out_json.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {out_json}")

    for dest_dir in (FIG_DIR, PAPER_FIG_DIR):
        plot_mechanism(rec, dest_dir / "fig_has_paired_delta")
        plot_depth(depth, dest_dir / "fig_has_forward_depth")

    print("\nPaired HAS-LeWM mean [95% CI], wins/ties/losses")
    for task in TASKS:
        print(f"  {TASK_TITLES[task]}")
        for off in OFFSETS:
            r = rec["tasks"][task]["delta_vs_lewm"][str(off)]
            print(
                f"    o={off}: {r['mean']:+.1f} [{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}] "
                f"{r['wins']}/{r['ties']}/{r['losses']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
