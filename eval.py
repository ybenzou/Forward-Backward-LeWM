"""Single-mode / single-offset FBLeWM evaluation entry."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint_utils import load_fblewm_checkpoint
from policy import FBWorldModelPolicy


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def sample_or_load_starts(dataset, cfg, starts_file: Path | None):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    if starts_file is not None and starts_file.exists():
        manifest = json.loads(starts_file.read_text())
        return (
            np.array(manifest["row_indices"]),
            manifest["episodes"],
            manifest["start_steps"],
            manifest,
        )

    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    goal_offset = int(cfg.eval.goal_offset_steps)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{int(valid_mask.sum())} valid starting points (goal_offset={goal_offset}).", flush=True)
    g = np.random.default_rng(cfg.seed)
    picked = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    row_indices = np.sort(valid_indices[picked])
    eval_episodes = dataset.get_row_data(row_indices)[col_name]
    eval_start_idx = dataset.get_row_data(row_indices)["step_idx"]
    manifest = {
        "row_indices": row_indices.tolist(),
        "episodes": eval_episodes.tolist(),
        "start_steps": eval_start_idx.tolist(),
        "seed": int(cfg.seed),
        "num_eval": int(cfg.eval.num_eval),
        "goal_offset_for_sampling": goal_offset,
    }
    return row_indices, eval_episodes.tolist(), eval_start_idx.tolist(), manifest


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be <= eval_budget"

    plan_len = int(cfg.plan_config.horizon * cfg.plan_config.action_block)
    action_block = int(cfg.plan_config.action_block)
    goal_offset = int(cfg.eval.goal_offset_steps)
    if goal_offset % action_block != 0:
        raise ValueError(
            f"goal_offset={goal_offset} must be divisible by action_block={action_block}"
        )
    if plan_len % action_block != 0:
        raise ValueError(
            f"plan_len={plan_len} must be divisible by action_block={action_block}"
        )

    mode = str(cfg.get("planning_mode", "official"))
    starts_file = cfg.get("starts_file", None)
    starts_path = Path(starts_file) if starts_file else None

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy_name = cfg.get("policy", "random")
    if policy_name == "random":
        raise ValueError("FBLeWM eval requires a trained policy checkpoint")

    model = load_fblewm_checkpoint(policy_name, cache_dir=cfg.get("cache_dir"))
    model = model.to("cuda")
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    model.set_planning_mode(mode)

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = FBWorldModelPolicy(
        solver=solver,
        config=config,
        goal_offset=goal_offset,
        planning_mode=mode,
        process=process,
        transform=transform,
    )
    world.set_policy(policy)

    _, eval_episodes, eval_start_idx, manifest = sample_or_load_starts(
        dataset, cfg, starts_path
    )
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes for evaluation.")

    results_root = Path(cfg.get("hydra_run_dir") or (ROOT / "outputs" / "eval"))
    results_root.mkdir(parents=True, exist_ok=True)
    video_dir = results_root / f"videos_{mode}_offset_{goal_offset}"
    video_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"running 1/1 | mode={mode} goal_offset={goal_offset} "
        f"eval_budget={cfg.eval.eval_budget} num_eval={cfg.eval.num_eval}",
        flush=True,
    )
    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx,
        goal_offset=goal_offset,
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=eval_episodes,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=video_dir,
    )
    dt = time.time() - t0
    sr = float(metrics.get("success_rate", float("nan")))
    succ = metrics.get("episode_successes", None)
    n_ok = int(np.sum(succ)) if succ is not None else -1
    print(
        f"eval 1/1 DONE | mode={mode} offset={goal_offset} "
        f"success_rate={sr:.1f}% ({n_ok}/{cfg.eval.num_eval}) time={dt:.1f}s",
        flush=True,
    )

    out = {
        "mode": mode,
        "offset": goal_offset,
        "budget": int(cfg.eval.eval_budget),
        "success_rate": sr,
        "n_success": n_ok,
        "n_eval": int(cfg.eval.num_eval),
        "seconds": dt,
        "metrics": {
            k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()
        },
        "starts_manifest": manifest,
        "policy": policy_name,
    }
    (results_root / f"result_{mode}_offset_{goal_offset}.json").write_text(
        json.dumps(out, indent=2)
    )
    print(metrics, flush=True)
    return 0


if __name__ == "__main__":
    run()
