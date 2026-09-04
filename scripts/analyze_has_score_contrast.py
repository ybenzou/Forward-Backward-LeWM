#!/usr/bin/env python3
"""Score-contrast figure: LeWM saturation vs HAS restored contrast.

Held-out dataset windows (all 10 eval-manifest episodes excluded).
Same held-out window draw as fig_has_latents_distance (default: 400
windows, one draw). Three remaining gaps after H=25:

  k=5  compares z_{t+25} with z_{t+50}
  k=10 compares z_{t+25} with z_{t+75}
  k=15 compares z_{t+25} with z_{t+100}

Does not run CEM. Writes interpretability_preview/fig_has_score_contrast.*.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import stable_worldmodel as swm
import torch

from checkpoint_utils import load_fblewm_checkpoint
from eval_fblewm_matrix import img_transform
from analyze_has_latents import (
    TASKS,
    apply_forward,
    compose_eval_cfg,
    encode_rows,
    episode_rows,
    excluded_episodes,
    pairwise_l2,
)
from plot_iclr_multiseed import PALETTE, FigureStyle, apply_publication_style

BLOCK = 5
HORIZON = 25
MAX_OFFSET = 100
ALIGNMENTS = (
    (5, 50),
    (10, 75),
    (15, 100),
)
N_WINDOWS = 400
OUT_ROOT = ROOT / "outputs" / "diag" / "has_score_contrast" / "v1"
PREV = ROOT / "paper" / "interpretability_preview"


def eligible_windows(rows_by_ep: dict[int, np.ndarray], exclude: set[int]) -> list[np.ndarray]:
    need = MAX_OFFSET + 1
    out = []
    for ep, rows in rows_by_ep.items():
        if ep in exclude or len(rows) < need:
            continue
        for t in range(0, len(rows) - MAX_OFFSET, BLOCK):
            out.append(rows[t : t + MAX_OFFSET + 1 : BLOCK])
    if not out:
        raise RuntimeError("no held-out windows with length >= 100")
    return out


def sample_windows(windows: list[np.ndarray], n_windows: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(n_windows, len(windows))
    pick = rng.choice(len(windows), size=n, replace=False)
    return [windows[int(i)] for i in pick]


@torch.no_grad()
def run_task(task: str, n_windows: int, seed: int, batch_size: int, cache_dir: Path) -> dict:
    spec = TASKS[task]
    cfg = compose_eval_cfg(spec["config"])
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )
    transform = img_transform(int(cfg.eval.img_size))
    exclude = excluded_episodes(task)
    windows = sample_windows(eligible_windows(episode_rows(dataset), exclude), n_windows, seed)
    unique_rows = np.unique(np.concatenate(windows))
    print(
        f"{task}: windows={len(windows)} excluded={len(exclude)} "
        f"unique_rows={len(unique_rows)}",
        flush=True,
    )

    model = load_fblewm_checkpoint(spec["policy"], cache_dir=str(cache_dir))
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    embs = encode_rows(model, dataset, unique_rows, transform, batch_size)
    emb_by_row = {int(r): e for r, e in zip(unique_rows, embs)}

    h_idx = HORIZON // BLOCK
    payload = {"n_windows": len(windows), "alignments": {}}
    arrays = {}
    z_seq = np.stack([np.stack([emb_by_row[int(r)] for r in rows], axis=0) for rows in windows])
    z_h = z_seq[:, h_idx]
    for k, offset in ALIGNMENTS:
        z_g = z_seq[:, offset // BLOCK]
        z_f = apply_forward(model, z_h, k, batch_size)
        red = pairwise_l2(z_h, z_g)
        blue = pairwise_l2(z_f, z_g)
        arrays[(k, offset)] = {"lewm": red, "has": blue}
        payload["alignments"][str(k)] = {
            "offset": offset,
            "n": int(len(red)),
            "frac_has_closer": float(np.mean(blue < red)),
        }
    del model
    torch.cuda.empty_cache()

    task_dir = OUT_ROOT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        task_dir / "distances.npz",
        **{
            f"k{k}_o{off}_{name}": arrays[(k, off)][name]
            for k, off in ALIGNMENTS
            for name in ("lewm", "has")
        },
    )
    (task_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload["alignments"].items()}, indent=2), flush=True)
    return arrays


def load_saved(task: str) -> dict:
    blob = np.load(OUT_ROOT / task / "distances.npz")
    return {
        (k, off): {"lewm": blob[f"k{k}_o{off}_lewm"], "has": blob[f"k{k}_o{off}_has"]}
        for k, off in ALIGNMENTS
    }


def plot_contrast(task_data: dict[str, dict], dest: Path) -> None:
    apply_publication_style(
        FigureStyle(
            font_size=9,
            axes_linewidth=0.8,
            font_family=("Liberation Sans", "DejaVu Sans", "sans-serif"),
        )
    )
    names = [n for n in ("pusht", "tworoom", "reacher") if n in task_data]
    nrows, ncols = len(names), len(ALIGNMENTS)
    fig = plt.figure(figsize=(7.4, 4.35))
    gs = fig.add_gridspec(
        nrows,
        ncols,
        left=0.10,
        right=0.995,
        top=0.90,
        bottom=0.08,
        wspace=0.10,
        hspace=0.16,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            sharey = axes[r, 0] if c else None
            axes[r, c] = fig.add_subplot(gs[r, c], sharey=sharey)
            if c:
                axes[r, c].tick_params(labelleft=False)
    for r, task in enumerate(names):
        row_hi = 0.0
        for k, off in ALIGNMENTS:
            rec = task_data[task][(k, off)]
            row_hi = max(row_hi, float(rec["lewm"].max()), float(rec["has"].max()))
        xmax = max(1.0, np.ceil(row_hi / 5.0) * 5.0)
        for c, (k, off) in enumerate(ALIGNMENTS):
            ax = axes[r, c]
            rec = task_data[task][(k, off)]
            bins = np.linspace(0.0, xmax, 25)
            ax.hist(
                rec["lewm"],
                bins=bins,
                range=(0.0, xmax),
                alpha=0.55,
                color=PALETTE["red_strong"],
                edgecolor="none",
            )
            ax.hist(
                rec["has"],
                bins=bins,
                range=(0.0, xmax),
                alpha=0.55,
                color=PALETTE["blue_main"],
                edgecolor="none",
            )
            ax.set_xlim(0.0, xmax)
            if r == 0:
                ax.set_title(rf"$k{{=}}{k}$,  $o{{=}}{off}$", pad=2)
            if c == 0:
                ax.annotate(
                    TASKS[task]["title"],
                    xy=(-0.24, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9,
                    color="#2B2B2B",
                )
            if r == nrows - 1:
                ax.set_xlabel("Terminal score")
            ax.tick_params(length=2.5)
    axes[nrows // 2, 0].set_ylabel("Count")
    handles = [
        Patch(facecolor=PALETTE["red_strong"], alpha=0.55, label=r"LeWM  $\|z_{e+H}-z_g\|$"),
        Patch(facecolor=PALETTE["blue_main"], alpha=0.55, label=r"HAS  $\|F^{k}(z_{e+H})-z_g\|$"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.55, 0.905),
        handlelength=1.2,
        columnspacing=1.8,
        borderaxespad=0.0,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        out = dest.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
        print(f"wrote {out}")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="pusht,tworoom,reacher")
    p.add_argument("--n-windows", type=int, default=N_WINDOWS)
    p.add_argument("--seed", type=int, default=0, help="Which 400 windows to draw")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-encode", action="store_true")
    args = p.parse_args()

    cache_dir = Path(os.environ.get("STABLEWM_HOME", ROOT / ".stable-wm"))
    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_data = {}
    for task in names:
        if args.skip_encode:
            task_data[task] = load_saved(task)
        else:
            task_data[task] = run_task(task, args.n_windows, args.seed, args.batch_size, cache_dir)
    plot_contrast(task_data, PREV / "fig_has_score_contrast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
