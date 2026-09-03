#!/usr/bin/env python3
"""Record paired LeWM/HAS rollouts and draw a timeline process figure.

Re-runs starts that the seed-42 matrix marked as HAS-success / LeWM-fail.
Because a smaller env batch changes CEM samples, only pairs that still
contrast after the re-run are used in the figure.
"""

from __future__ import annotations

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

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import stable_worldmodel.world.world as world_mod
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from checkpoint_utils import load_fblewm_checkpoint
from eval_fblewm_matrix import BUDGETS, build_process, img_transform
from policy import FBWorldModelPolicy
from plot_iclr_multiseed import (
    FIG_DIR,
    PAPER_FIG_DIR,
    PALETTE,
    FigureStyle,
    apply_publication_style,
)

BLOCK = 5
OFFSET = 75
BLOCK_TIMES = tuple(range(0, OFFSET + 1, BLOCK))
CONTACT_TIMES = (0, 25, 50, 75)
MAJOR_TIMES = {0, 25, 50, 75}

TASKS = {
    "pusht": {
        "config": "pusht",
        "policy": "fblewm/weights_epoch_10.pt",
        "title": "PushT",
        "offset": OFFSET,
        "candidates": None,
        "preferred_local_j": None,
        "times": BLOCK_TIMES,
    },
    "tworoom": {
        "config": "tworoom",
        "policy": "fblewm_tworoom/weights_epoch_10.pt",
        "title": "TwoRoom",
        "offset": OFFSET,
        "candidates": None,
        "preferred_local_j": 6,
        "times": BLOCK_TIMES,
    },
    "reacher": {
        "config": "reacher",
        "policy": "fblewm_reacher_v1/weights_epoch_10.pt",
        "title": "Reacher",
        "offset": OFFSET,
        "candidates": None,
        "preferred_local_j": 1,
        "times": BLOCK_TIMES,
    },
}

_ORIG_SAVE_PANEL = world_mod.save_panel_videos


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    if x.ndim == 3 and x.shape[-1] == 4:
        x = x[..., :3]
    if np.issubdtype(x.dtype, np.floating):
        x = np.clip(x, 0.0, 1.0) if x.max() <= 1.5 else np.clip(x, 0.0, 255.0)
        x = (x * 255.0).round().astype(np.uint8) if x.max() <= 1.5 else x.astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    return x


def _save_raw_and_panel(video_dir, panels, fps: int = 15) -> None:
    """Save raw agent/goal arrays. Skip mp4 encoding (the slow path)."""
    del fps
    video_dir = Path(video_dir)
    raw_dir = video_dir / "raw_agent"
    raw_dir.mkdir(parents=True, exist_ok=True)
    agent = panels["agent"]
    goal = panels.get("goal")
    for i in range(len(agent)):
        np.save(raw_dir / f"env_{i}.npy", np.asarray(agent[i]))
        if goal is not None:
            np.save(raw_dir / f"goal_{i}.npy", np.asarray(goal[i]))


world_mod.save_panel_videos = _save_raw_and_panel


def contrast_indices(results_path: Path, offset: int) -> list[int]:
    rec = json.loads(results_path.read_text())
    official = forward = None
    for row in rec["results"]:
        if int(row["offset"]) != offset:
            continue
        bits = row["metrics"]["episode_successes"]
        if row["mode"] == "official":
            official = bits
        elif row["mode"] == "forward":
            forward = bits
    if official is None or forward is None:
        raise RuntimeError(f"missing official/forward rows in {results_path}")
    return [i for i, (a, b) in enumerate(zip(official, forward)) if (not a) and b]


def goal_frame(dataset, episode: int, start_step: int, offset: int) -> np.ndarray:
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = dataset.get_col_data(col)
    step = dataset.get_col_data("step_idx")
    rows = np.nonzero((ep == episode) & (step == start_step + offset))[0]
    if len(rows) == 0:
        raise RuntimeError(f"missing goal row episode={episode} step={start_step + offset}")
    return _to_uint8_rgb(dataset.get_row_data(int(rows[0]))["pixels"])


def compose_eval_cfg(name: str):
    config_dir = str((ROOT / "config" / "eval").resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=name)


def run_subset(task: str, spec: dict, out_root: Path, cache_dir: Path, all_starts: bool = False) -> dict:
    matrix_dir = ROOT / "outputs" / "diag" / "iclr_bar" / "multiseed" / task / "seed_42"
    manifest = json.loads((matrix_dir / "starts_manifest.json").read_text())
    historical = set(contrast_indices(matrix_dir / "results.json", spec["offset"]))
    if all_starts:
        want = list(range(len(manifest["start_steps"])))
    elif spec.get("candidates"):
        want = [i for i in spec["candidates"] if i in historical]
    else:
        want = sorted(historical)
    if not want:
        raise RuntimeError(f"no historical HAS-only indices remain for {task}")

    start_steps = [manifest["start_steps"][i] for i in want]
    episodes = [manifest["episodes"][i] for i in want]
    offset = spec["offset"]
    budget = BUDGETS[offset]
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = compose_eval_cfg(spec["config"])

    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )
    process = build_process(dataset, cfg.dataset.keys_to_cache)
    transform = {
        "pixels": img_transform(int(cfg.eval.img_size)),
        "goal": img_transform(int(cfg.eval.img_size)),
    }
    model = load_fblewm_checkpoint(spec["policy"], cache_dir=str(cache_dir))
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    summary = {
        "task": task,
        "offset": offset,
        "indices": want,
        "episodes": episodes,
        "start_steps": start_steps,
        "modes": {},
    }
    for mode in ("official", "forward"):
        world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
        world_cfg["num_envs"] = len(want)
        world_cfg["max_episode_steps"] = 2 * budget
        world = swm.World(**world_cfg, image_shape=(224, 224))
        plan_config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        model.set_planning_mode(mode)
        policy = FBWorldModelPolicy(
            solver=solver,
            config=plan_config,
            goal_offset=offset,
            planning_mode=mode,
            process=process,
            transform=transform,
        )
        world.set_policy(policy)
        video_dir = out_root / "videos" / mode
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"{task} {mode} o={offset} n={len(want)}", flush=True)
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=offset,
            eval_budget=budget,
            episodes_idx=episodes,
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=video_dir,
        )
        summary["modes"][mode] = {
            "success": [bool(x) for x in metrics["episode_successes"]],
            "video_dir": str(video_dir),
            "raw_dir": str(video_dir / "raw_agent"),
        }
        print(f"  success={summary['modes'][mode]['success']}", flush=True)

    pairs = []
    for j, idx in enumerate(want):
        if summary["modes"]["official"]["success"][j]:
            continue
        if not summary["modes"]["forward"]["success"][j]:
            continue
        pairs.append(
            {
                "index": idx,
                "local_j": j,
                "episode": episodes[j],
                "start_step": start_steps[j],
                "frames": {
                    "official": str(out_root / "videos" / "official" / "raw_agent" / f"env_{j}.npy"),
                    "forward": str(out_root / "videos" / "forward" / "raw_agent" / f"env_{j}.npy"),
                },
            }
        )

    if not pairs:
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise RuntimeError(f"no HAS-success/LeWM-fail pair reproduced for {task}")

    chosen = pairs[0]
    preferred = spec.get("preferred_local_j")
    if preferred is not None:
        match = [p for p in pairs if p["local_j"] == int(preferred)]
        if match:
            chosen = match[0]
    summary["pairs"] = pairs
    summary["chosen"] = chosen
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "spec": spec,
        "task": task,
        "chosen": chosen,
        "goal": goal_frame(dataset, chosen["episode"], chosen["start_step"], offset),
        "offset": offset,
    }


def _time_label(t: int, major_only: bool = True) -> str:
    if major_only and int(t) not in MAJOR_TIMES:
        return ""
    return f"$t{int(t)}$"


def plot_process(panels: list[dict], dest: Path, *, wrap: bool = False) -> None:
    apply_publication_style(
        FigureStyle(
            font_size=11,
            axes_linewidth=1.0,
            font_family=("DejaVu Sans", "sans-serif"),
        )
    )
    n_tasks = len(panels)
    times = list(panels[0]["spec"]["times"])
    method_names = ("LeWM", "HAS")
    method_colors = (PALETTE["red_strong"], PALETTE["blue_main"])

    if wrap:
        mid = (len(times) + 1) // 2
        strips = [times[:mid], times[mid:]]
        n_cols = max(len(s) for s in strips) + 1
        rows_per_task = 1 + 4
        fig_h = 3.15 * n_tasks
    else:
        strips = [times]
        n_cols = len(times) + 1
        rows_per_task = 3
        fig_h = 2.35 * n_tasks

    fig_w = min(18.0, 0.78 * n_cols + 1.7)
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(n_tasks, 1, hspace=0.38)

    for p_i, panel in enumerate(panels):
        n_method_rows = 2 * len(strips)
        height_ratios = [0.32] + [1.0] * n_method_rows
        inner = outer[p_i].subgridspec(
            1 + n_method_rows,
            n_cols + 1,
            height_ratios=height_ratios,
            width_ratios=[0.72] + [1.0] * n_cols,
            hspace=0.04,
            wspace=0.04,
        )
        header_times = strips[0]
        for c, t in enumerate(header_times):
            ax = fig.add_subplot(inner[0, c + 1])
            ax.axis("off")
            label = _time_label(t, major_only=not wrap)
            if label:
                ax.text(0.5, 0.1, label, ha="center", va="bottom", fontsize=9, color="#4D4D4D")
            elif int(t) % 25 != 0:
                ax.plot([0.5], [0.08], marker="|", color="#9A9A9A", markersize=4, transform=ax.transAxes)
        ax = fig.add_subplot(inner[0, n_cols])
        ax.axis("off")
        ax.text(0.5, 0.1, "goal", ha="center", va="bottom", fontsize=9, color="#4D4D4D")

        name_ax = fig.add_subplot(inner[1:, 0])
        name_ax.axis("off")
        name_ax.text(
            0.0,
            0.5,
            panel["spec"]["title"],
            ha="left",
            va="center",
            fontsize=13,
            color="#272727",
            transform=name_ax.transAxes,
        )
        name_ax.text(1.0, 0.76, "LeWM", ha="right", va="center", fontsize=10, color=method_colors[0])
        name_ax.text(1.0, 0.24, "HAS", ha="right", va="center", fontsize=10, color=method_colors[1])

        seqs = (panel["official_frames"], panel["has_frames"])
        for strip_i, strip in enumerate(strips):
            for r, (frames, color) in enumerate(zip(seqs, method_colors)):
                row = 1 + strip_i * 2 + r
                for c, t in enumerate(strip):
                    ax = fig.add_subplot(inner[row, c + 1])
                    ax.imshow(frames[int(t)])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.5)
                        spine.set_color("#CFCECE")
                    if c == 0:
                        ax.annotate(
                            "",
                            xy=(-0.05, 0.0),
                            xytext=(-0.05, 1.0),
                            xycoords="axes fraction",
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "color": color, "lw": 2.0},
                        )
                ax = fig.add_subplot(inner[row, n_cols])
                ax.imshow(panel["goal"])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.5)
                    spine.set_color("#767676")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf", "svg"):
        out = dest.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)
        print(f"wrote {out}")
    plt.close(fig)


def plot_contact_sheet(task: str, spec: dict, task_root: Path, dest: Path) -> None:
    summary = json.loads((task_root / "summary.json").read_text())
    pairs = summary.get("pairs") or []
    if not pairs:
        raise RuntimeError(f"no contrast pairs in {task_root / 'summary.json'}")
    apply_publication_style(
        FigureStyle(font_size=10, axes_linewidth=0.8, font_family=("DejaVu Sans", "sans-serif"))
    )
    n = len(pairs)
    n_cols = len(CONTACT_TIMES) * 2 + 1
    fig, axes = plt.subplots(n, n_cols, figsize=(1.15 * n_cols + 0.8, 1.25 * n + 0.4))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    headers = []
    for t in CONTACT_TIMES:
        headers.extend([f"LeWM $t={t}$", f"HAS $t={t}$"])
    headers.append("goal")
    for c, text in enumerate(headers):
        axes[0, c].set_title(text, fontsize=8, pad=2)
    for r, pair in enumerate(pairs):
        j = pair["local_j"]
        official = _frames_from_raw_or_video(task_root, "official", j, CONTACT_TIMES)
        has = _frames_from_raw_or_video(task_root, "forward", j, CONTACT_TIMES)
        goal = _goal_from_raw_or_video(task_root, j)
        col = 0
        for t in CONTACT_TIMES:
            for frames in (official, has):
                axes[r, col].imshow(frames[int(t)])
                axes[r, col].set_xticks([])
                axes[r, col].set_yticks([])
                col += 1
        axes[r, col].imshow(goal)
        axes[r, col].set_xticks([])
        axes[r, col].set_yticks([])
        axes[r, 0].set_ylabel(f"j={j}\nep={pair['episode']}", fontsize=7)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(dest.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
        print(f"wrote {dest.with_suffix('.' + fmt)}")
    plt.close(fig)


def _frames_from_raw_or_video(task_root: Path, mode: str, local_j: int, times: tuple[int, ...]) -> dict[int, np.ndarray]:
    raw = task_root / "videos" / mode / "raw_agent" / f"env_{local_j}.npy"
    if raw.exists():
        seq = np.load(raw)
        if getattr(seq, "ndim", 0) >= 3:
            return {
                int(t): _to_uint8_rgb(seq[min(int(t), len(seq) - 1)])
                for t in times
            }
    raise RuntimeError(f"missing raw frames {raw}")


def _goal_from_raw_or_video(task_root: Path, local_j: int) -> np.ndarray:
    for mode in ("forward", "official"):
        raw = task_root / "videos" / mode / "raw_agent" / f"goal_{local_j}.npy"
        if raw.exists():
            return _to_uint8_rgb(np.load(raw))
    raise RuntimeError(f"missing goal frames for local_j={local_j}")


def attach_frames(panel: dict, task_root: Path) -> dict:
    chosen = panel["chosen"]
    times = panel["spec"]["times"]
    j = chosen["local_j"]
    panel["official_frames"] = _frames_from_raw_or_video(task_root, "official", j, times)
    panel["has_frames"] = _frames_from_raw_or_video(task_root, "forward", j, times)
    panel["goal"] = _goal_from_raw_or_video(task_root, j)
    return panel


def load_saved_panel(task: str, spec: dict, out_root: Path, local_j: int | None) -> dict:
    summary = json.loads((out_root / task / "summary.json").read_text())
    chosen = summary["chosen"]
    if local_j is not None:
        match = [p for p in summary["pairs"] if p["local_j"] == local_j]
        if not match:
            raise RuntimeError(f"{task} local_j={local_j} is not a reproduced contrast pair")
        chosen = match[0]
    return attach_frames(
        {
            "spec": spec,
            "task": task,
            "chosen": chosen,
            "offset": spec["offset"],
        },
        out_root / task,
    )


def _parse_choose(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    out = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, value = part.split("=", 1)
        out[name.strip()] = int(value)
    return out


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--all-starts", action="store_true")
    p.add_argument("--tasks", default="pusht,tworoom,reacher")
    p.add_argument("--choose", default="", help="task=local_j pairs, e.g. pusht=2,tworoom=6")
    p.add_argument("--contact-sheet", action="store_true")
    p.add_argument("--wrap", action="store_true", help="Split 0-35 and 40-75 into two strips")
    args = p.parse_args()

    cache_dir = Path(os.environ.get("STABLEWM_HOME", ROOT / ".stable-wm"))
    out_root = ROOT / "outputs" / "diag" / "process_compare" / "v2"
    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    choose = _parse_choose(args.choose)
    panels = []
    for task in names:
        spec = TASKS[task]
        if args.skip_eval:
            panels.append(load_saved_panel(task, spec, out_root, choose.get(task)))
        else:
            panel = run_subset(task, spec, out_root / task, cache_dir, all_starts=args.all_starts)
            if task in choose:
                summary = json.loads((out_root / task / "summary.json").read_text())
                match = [p for p in summary["pairs"] if p["local_j"] == choose[task]]
                if not match:
                    raise RuntimeError(f"{task} local_j={choose[task]} not in reproduced pairs")
                panel["chosen"] = match[0]
            panels.append(attach_frames(panel, out_root / task))
        if args.contact_sheet:
            plot_contact_sheet(
                task,
                spec,
                out_root / task,
                out_root / task / f"contact_{task}",
            )

    for dest_dir in (FIG_DIR, PAPER_FIG_DIR):
        plot_process(panels, dest_dir / "fig_has_process", wrap=args.wrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
