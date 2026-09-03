#!/usr/bin/env python3
"""Held-out Forward latent geometry: retrieval, PCA, t-SNE, goal distances.

Does not run CEM. Encodes dataset frames and applies F^k on projector latents.
Evaluation-manifest episodes are excluded.
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
import numpy as np
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from checkpoint_utils import load_fblewm_checkpoint
from eval_fblewm_matrix import img_transform
from plot_iclr_multiseed import (
    FIG_DIR,
    MULTI_ROOT,
    PAPER_FIG_DIR,
    PALETTE,
    SEEDS,
    FigureStyle,
    apply_publication_style,
    finalize_figure,
)

BLOCK = 5
HORIZON = 25
GOAL_OFFSET = 75
MAX_BLOCKS = GOAL_OFFSET // BLOCK
DEPTHS = (1, 5, 10, 15)
OUT_ROOT = ROOT / "outputs" / "diag" / "has_latents" / "v1"

TASKS = {
    "pusht": {
        "config": "pusht",
        "policy": "fblewm/weights_epoch_10.pt",
        "title": "PushT",
    },
    "tworoom": {
        "config": "tworoom",
        "policy": "fblewm_tworoom/weights_epoch_10.pt",
        "title": "TwoRoom",
    },
    "reacher": {
        "config": "reacher",
        "policy": "fblewm_reacher_v1/weights_epoch_10.pt",
        "title": "Reacher",
    },
}


def compose_eval_cfg(name: str):
    config_dir = str((ROOT / "config" / "eval").resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=name)


def excluded_episodes(task: str) -> set[int]:
    ids: set[int] = set()
    for seed in SEEDS:
        path = MULTI_ROOT / task / f"seed_{seed}" / "starts_manifest.json"
        rec = json.loads(path.read_text())
        ids.update(int(x) for x in rec["episodes"])
    return ids


def episode_rows(dataset) -> dict[int, np.ndarray]:
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = np.asarray(dataset.get_col_data(col))
    step = np.asarray(dataset.get_col_data("step_idx"))
    out: dict[int, np.ndarray] = {}
    for episode in np.unique(ep):
        idx = np.nonzero(ep == episode)[0]
        order = np.argsort(step[idx])
        rows = idx[order]
        steps = step[idx][order]
        if len(steps) < 2:
            continue
        if np.any(np.diff(steps) != 1):
            continue
        out[int(episode)] = rows
    return out


def sample_windows(
    rows_by_ep: dict[int, np.ndarray],
    exclude: set[int],
    n_windows: int,
    seed: int,
) -> list[tuple[int, int, np.ndarray]]:
    eligible = []
    need = GOAL_OFFSET + 1
    for ep, rows in rows_by_ep.items():
        if ep in exclude:
            continue
        if len(rows) < need:
            continue
        for t in range(0, len(rows) - GOAL_OFFSET, BLOCK):
            eligible.append((ep, t, rows[t : t + GOAL_OFFSET + 1 : BLOCK]))
    if not eligible:
        raise RuntimeError("no held-out windows with length >= 75")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(eligible), size=min(n_windows, len(eligible)), replace=False)
    return [eligible[int(i)] for i in pick]


def load_pixels(dataset, row_ids: np.ndarray, transform) -> torch.Tensor:
    frames = []
    for row_id in row_ids:
        pix = dataset.get_row_data(int(row_id))["pixels"]
        frames.append(transform(pix))
    stacked = torch.stack(frames, dim=0)
    if stacked.ndim == 4:
        stacked = stacked.unsqueeze(1)
    return stacked


@torch.no_grad()
def encode_rows(model, dataset, row_ids: np.ndarray, transform, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    embs = []
    for start in range(0, len(row_ids), batch_size):
        chunk = row_ids[start : start + batch_size]
        pixels = load_pixels(dataset, chunk, transform).to(device)
        emb = model.encode({"pixels": pixels})["emb"][:, -1].float().cpu().numpy()
        embs.append(emb)
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def apply_forward(model, latents: np.ndarray, steps: int, batch_size: int) -> np.ndarray:
    if steps == 0:
        return np.asarray(latents)
    device = next(model.parameters()).device
    out = []
    z_all = torch.from_numpy(np.asarray(latents)).float()
    for start in range(0, len(z_all), batch_size):
        z = z_all[start : start + batch_size].to(device)
        imagined = model.imagine_forward(z, steps)
        out.append(imagined.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def compute_metrics(z_seq: np.ndarray, f_from_t: dict[int, np.ndarray], f_from_h: dict[int, np.ndarray]) -> dict:
    # z_seq: [N, 16, D] at t+5j
    n, n_j, _ = z_seq.shape
    retrieval = {}
    for k, pred in f_from_t.items():
        d = np.linalg.norm(pred[:, None, :] - z_seq, axis=-1)
        hat = d.argmin(axis=1)
        retrieval[str(k)] = {
            "accuracy": float((hat == k).mean()),
            "mean_abs_block_error": float(np.abs(hat - k).mean()),
            "n": int(n),
        }
    z_h = z_seq[:, HORIZON // BLOCK]
    z_g = z_seq[:, GOAL_OFFSET // BLOCK]
    dist = {
        "lewm_h_to_g": float(pairwise_l2(z_h, z_g).mean()),
        "has_f10_h_to_g": float(pairwise_l2(f_from_h[10], z_g).mean()),
        "n": int(n),
    }
    return {"retrieval": retrieval, "distance": dist, "n_windows": int(n), "n_times": int(n_j)}


def _scatter(ax, points: np.ndarray, labels: np.ndarray, title: str) -> None:
    colors = {
        "z_H": PALETTE["red_strong"],
        "F^{10}(z_H)": PALETTE["blue_main"],
        "z_g": PALETTE["teal"],
    }
    markers = {"z_H": "o", "F^{10}(z_H)": "^", "z_g": "s"}
    for name in ("z_H", "F^{10}(z_H)", "z_g"):
        sel = labels == name
        ax.scatter(
            points[sel, 0],
            points[sel, 1],
            s=14,
            alpha=0.55,
            c=colors[name],
            marker=markers[name],
            linewidths=0,
            label=name,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_main(task_data: dict[str, dict], dest: Path) -> None:
    apply_publication_style(FigureStyle(font_size=12, axes_linewidth=1.2))
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    names = [n for n in ("pusht", "tworoom", "reacher") if n in task_data]
    for i, task in enumerate(names):
        rec = task_data[task]
        _scatter(axes[0, i], rec["pca"], rec["labels"], TASKS[task]["title"])
        ks = [int(k) for k in rec["metrics"]["retrieval"]]
        acc = [rec["metrics"]["retrieval"][str(k)]["accuracy"] for k in ks]
        axes[1, i].plot(ks, acc, color=PALETTE["blue_main"], marker="o", lw=2.0)
        axes[1, i].set_ylim(-0.05, 1.05)
        axes[1, i].set_xlabel(r"depth $k$")
        if i == 0:
            axes[1, i].set_ylabel("retrieval accuracy")
        axes[1, i].set_xticks(ks)
    axes[0, 0].legend(loc="best", fontsize=8, frameon=False)
    finalize_figure(fig, dest, formats=["png", "pdf", "svg"], pad=0.25)


def plot_tsne(task_data: dict[str, dict], dest: Path) -> None:
    apply_publication_style(FigureStyle(font_size=12, axes_linewidth=1.2))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    names = [n for n in ("pusht", "tworoom", "reacher") if n in task_data]
    for i, task in enumerate(names):
        _scatter(axes[i], task_data[task]["tsne"], task_data[task]["labels"], TASKS[task]["title"])
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    finalize_figure(fig, dest, formats=["png", "pdf", "svg"], pad=0.25)


def plot_distance(task_data: dict[str, dict], dest: Path) -> None:
    apply_publication_style(FigureStyle(font_size=12, axes_linewidth=1.2))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), sharey=True)
    names = [n for n in ("pusht", "tworoom", "reacher") if n in task_data]
    for i, task in enumerate(names):
        rec = task_data[task]
        axes[i].hist(rec["d_lewm"], bins=24, alpha=0.55, color=PALETTE["red_strong"], label=r"$\|z_H-z_g\|$")
        axes[i].hist(rec["d_has"], bins=24, alpha=0.55, color=PALETTE["blue_main"], label=r"$\|F^{10}(z_H)-z_g\|$")
        axes[i].set_title(TASKS[task]["title"])
        axes[i].set_xlabel("Euclidean distance")
        if i == 0:
            axes[i].set_ylabel("count")
    axes[0].legend(fontsize=8, frameon=False)
    finalize_figure(fig, dest, formats=["png", "pdf", "svg"], pad=0.25)


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
    rows_by_ep = episode_rows(dataset)
    windows = sample_windows(rows_by_ep, exclude, n_windows, seed)
    unique_rows = np.unique(np.concatenate([w[2] for w in windows]))
    print(
        f"{task}: windows={len(windows)} excluded={len(exclude)} "
        f"unique_rows={len(unique_rows)}",
        flush=True,
    )

    model = load_fblewm_checkpoint(spec["policy"], cache_dir=str(cache_dir))
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    emb_by_row = {}
    embs = encode_rows(model, dataset, unique_rows, transform, batch_size)
    for row_id, emb in zip(unique_rows, embs):
        emb_by_row[int(row_id)] = emb

    z_seq = np.stack(
        [np.stack([emb_by_row[int(r)] for r in rows], axis=0) for _, _, rows in windows],
        axis=0,
    )
    z_t = z_seq[:, 0]
    z_h = z_seq[:, HORIZON // BLOCK]
    z_g = z_seq[:, GOAL_OFFSET // BLOCK]
    f_from_t = {k: apply_forward(model, z_t, k, batch_size) for k in DEPTHS}
    f_from_h = {k: apply_forward(model, z_h, k, batch_size) for k in DEPTHS}
    metrics = compute_metrics(z_seq, f_from_t, f_from_h)

    stack = np.concatenate([z_h, f_from_h[10], z_g], axis=0)
    labels = np.array(["z_H"] * len(z_h) + ["F^{10}(z_H)"] * len(z_h) + ["z_g"] * len(z_g))
    pca = PCA(n_components=2, random_state=seed).fit_transform(stack)
    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(stack)

    task_dir = OUT_ROOT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        task_dir / "latents.npz",
        z_seq=z_seq,
        z_h=z_h,
        z_g=z_g,
        f10_h=f_from_h[10],
        pca=pca,
        tsne=tsne,
        labels=labels,
        d_lewm=pairwise_l2(z_h, z_g),
        d_has=pairwise_l2(f_from_h[10], z_g),
    )
    payload = {
        "task": task,
        "excluded_episode_count": len(exclude),
        "n_windows": len(windows),
        "depths": list(DEPTHS),
        "metrics": metrics,
    }
    (task_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return {
        "pca": pca,
        "tsne": tsne,
        "labels": labels,
        "metrics": metrics,
        "d_lewm": pairwise_l2(z_h, z_g),
        "d_has": pairwise_l2(f_from_h[10], z_g),
    }


def load_saved(task: str) -> dict:
    task_dir = OUT_ROOT / task
    blob = np.load(task_dir / "latents.npz", allow_pickle=True)
    metrics = json.loads((task_dir / "metrics.json").read_text())["metrics"]
    return {
        "pca": blob["pca"],
        "tsne": blob["tsne"],
        "labels": blob["labels"],
        "metrics": metrics,
        "d_lewm": blob["d_lewm"],
        "d_has": blob["d_has"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="pusht,tworoom,reacher")
    p.add_argument("--n-windows", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
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

    for dest_dir in (FIG_DIR, PAPER_FIG_DIR):
        plot_main(task_data, dest_dir / "fig_has_latents")
        plot_tsne(task_data, dest_dir / "fig_has_latents_tsne")
        plot_distance(task_data, dest_dir / "fig_has_latents_distance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
