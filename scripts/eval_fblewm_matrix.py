#!/usr/bin/env python3
"""Evaluate one FBLeWM checkpoint across modes × offsets (12 units).

Progress: only ``eval i/12 DONE`` advances completed count for the pipeline UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from checkpoint_utils import load_fblewm_checkpoint
from planning import ALL_PLANNING_MODES, BASE_MODES, FUSION_MODES
from policy import FBWorldModelPolicy, expected_replan_depths

MODES = BASE_MODES
OFFSETS = (25, 50, 75, 100)
BUDGETS = {25: 50, 50: 100, 75: 150, 100: 200}
FUSION_EVAL_MODES = FUSION_MODES


def collect_imaginer_diagnostics(model, device: str = "cuda") -> dict:
    """Non-invasive F/B rollout diagnostics (does not change planning)."""
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(8, 192, device=device)
        out = {"forward_norms": {}, "backward_norms": {}, "schedules": {}}
        for k in (1, 5, 10, 15):
            zf = model.imagine_forward(z0, k)
            if getattr(model.backward_imaginer, "is_conditional", False):
                zb = model.imagine_backward(z0, k, z_now=z0)
            else:
                zb = model.imagine_backward(z0, k)
            out["forward_norms"][str(k)] = float(zf.float().norm(dim=-1).mean().cpu())
            out["backward_norms"][str(k)] = float(zb.float().norm(dim=-1).mean().cpu())
            # Distance of imagined latent to the seed (proxy; no GT trajectory here).
            out.setdefault("forward_dist_to_seed", {})[str(k)] = float(
                (zf - z0).float().pow(2).mean().sqrt().cpu()
            )
            out.setdefault("backward_dist_to_seed", {})[str(k)] = float(
                (zb - z0).float().pow(2).mean().sqrt().cpu()
            )
        for offset in OFFSETS:
            out["schedules"][str(offset)] = expected_replan_depths(offset)

        # Random CEM-cost probe to detect early cost collapse (B=1, S=32).
        B, S, H, A = 1, 32, 5, 10
        pixels = torch.randn(B, S, 1, 3, 224, 224, device=device)
        goal = torch.randn(B, S, 1, 3, 224, 224, device=device)
        actions = torch.randn(B, S, H, A, device=device)
        cost_stats = {}
        for mode in BASE_MODES:
            model.set_planning_mode(mode)
            info = {"pixels": pixels.clone(), "goal": goal.clone()}
            if mode == "forward":
                info["imagine_steps"] = torch.full(
                    (B, S, 1), 5, dtype=torch.int64, device=device
                )
            costs = model.get_cost(info, actions.clone())
            cost_stats[mode] = {
                "mean": float(costs.float().mean().cpu()),
                "std": float(costs.float().std(unbiased=False).cpu()),
                "shape": list(costs.shape),
            }
        out["cem_cost_probe"] = cost_stats
        model.set_planning_mode("official")
    return out


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
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


def sample_starts_for_max_offset(dataset, num_eval: int, seed: int, max_offset: int = 100):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - max_offset - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(
        f"{int(valid_mask.sum())} valid starting points (goal_offset={max_offset}).",
        flush=True,
    )
    if len(valid_indices) < num_eval + 1:
        raise ValueError(
            f"Not enough valid starts for goal_offset={max_offset}: {len(valid_indices)}"
        )
    g = np.random.default_rng(seed)
    picked = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    row_indices = np.sort(valid_indices[picked])
    episodes = dataset.get_row_data(row_indices)[col_name].tolist()
    start_steps = dataset.get_row_data(row_indices)["step_idx"].tolist()
    manifest = {
        "row_indices": row_indices.tolist(),
        "episodes": episodes,
        "start_steps": start_steps,
        "seed": int(seed),
        "num_eval": int(num_eval),
        "goal_offset_for_sampling": int(max_offset),
    }
    raw = json.dumps(manifest, sort_keys=True)
    manifest["hash"] = hashlib.sha256(raw.encode()).hexdigest()
    return manifest


def build_process(dataset, keys):
    process = {}
    for col in keys:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FBLeWM 3 modes × 4 offsets matrix eval")
    p.add_argument("--policy", required=True, help="e.g. fblewm/weights_epoch_10.pt")
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("STABLEWM_HOME", str(ROOT / ".stable-wm")),
    )
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--modes",
        default="official,forward,backward",
        help=(
            "Comma-separated planning modes. Use 'fusion' as shorthand for "
            + ",".join(FUSION_EVAL_MODES)
        ),
    )
    p.add_argument("--offsets", default="25,50,75,100")
    p.add_argument("--hydra-run-dir", default=None)
    p.add_argument("--config-name", default="pusht")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip mode/offset units already present in results.jsonl under --hydra-run-dir",
    )
    p.add_argument(
        "--starts-manifest",
        default=None,
        help="Reuse an existing starts_manifest.json (fairness across runs)",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "CEM latent horizon (default: config plan_config.horizon, usually 5 → plan_len=25). "
            "Use a smaller value e.g. 2 (plan_len=10) so offset=25 has k>0 for Forward/Backward."
        ),
    )
    p.add_argument(
        "--receding-horizon",
        type=int,
        default=None,
        help="Receding horizon in latent steps (default: same as --horizon)",
    )
    return p.parse_args(argv)


def _expand_modes(modes_arg: str) -> tuple[str, ...]:
    raw = [m.strip() for m in modes_arg.split(",") if m.strip()]
    out: list[str] = []
    for m in raw:
        if m == "fusion":
            out.extend(FUSION_EVAL_MODES)
        elif m == "all":
            out.extend(ALL_PLANNING_MODES)
        else:
            if m not in ALL_PLANNING_MODES:
                raise ValueError(
                    f"unknown mode {m!r}; expected one of {ALL_PLANNING_MODES} "
                    "or shorthand 'fusion' / 'all'"
                )
            out.append(m)
    # preserve order, drop duplicates
    seen = set()
    uniq = []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return tuple(uniq)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        modes = _expand_modes(args.modes)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    offsets = tuple(int(x.strip()) for x in args.offsets.split(",") if x.strip())
    units = [(m, o) for m in modes for o in offsets]
    n_units = len(units)
    if n_units == 0:
        print("No eval units", file=sys.stderr)
        return 2

    config_dir = str((ROOT / "config" / "eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        base_cfg = compose(config_name=args.config_name)

    base_cfg.cache_dir = args.cache_dir
    base_cfg.policy = args.policy
    base_cfg.seed = args.seed
    base_cfg.eval.num_eval = args.num_eval

    out_root = Path(args.hydra_run_dir or (ROOT / "outputs" / "eval" / "matrix"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Optional short CEM horizon so offset=25 can use F/B (k>0).
    if args.horizon is not None:
        base_cfg.plan_config.horizon = int(args.horizon)
        base_cfg.plan_config.receding_horizon = int(
            args.receding_horizon
            if args.receding_horizon is not None
            else args.horizon
        )
    elif args.receding_horizon is not None:
        base_cfg.plan_config.receding_horizon = int(args.receding_horizon)

    plan_len = int(base_cfg.plan_config.horizon) * int(base_cfg.plan_config.action_block)
    action_block = int(base_cfg.plan_config.action_block)
    print("==== FBLeWM MATRIX EVAL ====", flush=True)
    print(f"policy={args.policy}", flush=True)
    print(f"cache_dir={args.cache_dir}", flush=True)
    print(
        f"plan: horizon={base_cfg.plan_config.horizon} "
        f"receding={base_cfg.plan_config.receding_horizon} "
        f"action_block={action_block} plan_len={plan_len}",
        flush=True,
    )
    from policy import compute_imagine_steps

    for o in offsets:
        k0 = compute_imagine_steps(o, 0, plan_len, action_block)
        print(f"  schedule offset={o}: k(t=0)={k0}", flush=True)
    print(f"units={n_units} modes={list(modes)} offsets={list(offsets)}", flush=True)
    print(f"eval 0/{n_units} | loading dataset/model", flush=True)

    max_budget = max(BUDGETS[o] for o in offsets)
    dataset_path = Path(args.cache_dir)
    dataset = swm.data.HDF5Dataset(
        base_cfg.eval.dataset_name,
        keys_to_cache=base_cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    process = build_process(dataset, base_cfg.dataset.keys_to_cache)
    transform = {
        "pixels": img_transform(int(base_cfg.eval.img_size)),
        "goal": img_transform(int(base_cfg.eval.img_size)),
    }

    model = load_fblewm_checkpoint(args.policy, cache_dir=args.cache_dir)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    diag_path = out_root / "diagnostics.json"
    if not diag_path.exists():
        diag = collect_imaginer_diagnostics(model, device="cuda")
        diag_path.write_text(json.dumps(diag, indent=2))
        print(f"wrote diagnostics.json schedules={diag['schedules']}", flush=True)
    else:
        print(f"reusing existing {diag_path.name}", flush=True)

    # Shared starts valid for longest offset.
    starts_path = out_root / "starts_manifest.json"
    manifest_src = Path(args.starts_manifest) if args.starts_manifest else None
    if manifest_src is None and starts_path.exists() and args.resume:
        manifest_src = starts_path
    if manifest_src is not None and manifest_src.exists():
        manifest = json.loads(manifest_src.read_text())
        print(
            f"reusing starts manifest hash={manifest.get('hash')} from {manifest_src}",
            flush=True,
        )
    else:
        max_offset = max(offsets)
        manifest = sample_starts_for_max_offset(
            dataset, num_eval=args.num_eval, seed=args.seed, max_offset=max_offset
        )
    starts_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote starts manifest hash={manifest['hash']}", flush=True)
    print(
        f"shared start_steps (n={len(manifest['start_steps'])}): "
        f"{manifest['start_steps'][:8]}...",
        flush=True,
    )

    results = []
    jsonl_path = out_root / "results.jsonl"
    done_keys: set[tuple[str, int]] = set()
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done_keys.add((row["mode"], int(row["offset"])))
            results.append(row)
        print(f"resume: loaded {len(done_keys)} completed units from {jsonl_path}", flush=True)
    elif jsonl_path.exists() and not args.resume:
        jsonl_path.unlink()

    total_t0 = time.time()
    for i, (mode, offset) in enumerate(units, start=1):
        budget = BUDGETS[offset]
        if (mode, offset) in done_keys:
            print(
                f"eval {i}/{n_units} SKIP | mode={mode} offset={offset} "
                f"(already in results.jsonl)",
                flush=True,
            )
            # Keep progress semantics: emit DONE so UI can advance if desired.
            prev = next(r for r in results if r["mode"] == mode and int(r["offset"]) == offset)
            print(
                f"eval {i}/{n_units} DONE | mode={mode} offset={offset} "
                f"success_rate={float(prev['success_rate']):.1f}% "
                f"({prev['n_success']}/{prev['n_eval']}) time={float(prev['seconds']):.1f}s",
                flush=True,
            )
            continue
        print(
            f"eval {i}/{n_units} START | mode={mode} offset={offset} "
            f"budget={budget} (completed {i-1}/{n_units})",
            flush=True,
        )

        # Fresh world / solver / policy per unit.
        world_cfg = OmegaConf.to_container(base_cfg.world, resolve=True)
        world_cfg["num_envs"] = args.num_eval
        world_cfg["max_episode_steps"] = 2 * max(budget, max_budget)
        world = swm.World(**world_cfg, image_shape=(224, 224))

        # Reload solver with same model weights but fresh RNG.
        plan_config = swm.PlanConfig(**base_cfg.plan_config)
        solver = hydra.utils.instantiate(base_cfg.solver, model=model)
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

        video_dir = out_root / "videos" / mode / f"offset_{offset}"
        video_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=manifest["start_steps"],
            goal_offset=offset,
            eval_budget=budget,
            episodes_idx=manifest["episodes"],
            callables=OmegaConf.to_container(
                base_cfg.eval.get("callables"), resolve=True
            ),
            video=video_dir,
        )
        dt = time.time() - t0
        sr = float(metrics.get("success_rate", float("nan")))
        succ = metrics.get("episode_successes", None)
        n_ok = int(np.sum(succ)) if succ is not None else -1
        row = {
            "unit": i,
            "mode": mode,
            "offset": offset,
            "budget": budget,
            "success_rate": sr,
            "n_success": n_ok,
            "n_eval": args.num_eval,
            "seconds": dt,
            "starts_hash": manifest["hash"],
            "policy": args.policy,
            "seed": args.seed,
            "metrics": {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in metrics.items()
            },
        }
        results.append(row)
        with jsonl_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"eval {i}/{n_units} DONE | mode={mode} offset={offset} "
            f"success_rate={sr:.1f}% ({n_ok}/{args.num_eval}) time={dt:.1f}s",
            flush=True,
        )

    total_dt = time.time() - total_t0
    print("", flush=True)
    print(f"==== SUMMARY ({len(modes)} modes × {len(offsets)} offsets) ====", flush=True)
    print(
        f"{'mode':>14}  {'offset':>6}  {'budget':>6}  {'success_rate':>12}  "
        f"{'ok/n':>8}  {'seconds':>10}",
        flush=True,
    )
    for r in results:
        print(
            f"{r['mode']:>14}  {r['offset']:6d}  {r['budget']:6d}  "
            f"{r['success_rate']:11.1f}%  {r['n_success']:3d}/{r['n_eval']:<3d}  "
            f"{r['seconds']:10.1f}",
            flush=True,
        )
    print(f"total_time: {total_dt:.1f}s", flush=True)
    print(f"eval {n_units}/{n_units} | all units finished", flush=True)

    summary = {
        "policy": args.policy,
        "cache_dir": args.cache_dir,
        "seed": args.seed,
        "num_eval": args.num_eval,
        "starts_hash": manifest["hash"],
        "total_time": total_dt,
        "results": results,
    }
    (out_root / "results.json").write_text(json.dumps(summary, indent=2))
    summary_txt = out_root / "summary.txt"
    with summary_txt.open("w") as f:
        f.write("==== FBLeWM MATRIX SUMMARY ====\n")
        f.write(f"policy: {args.policy}\n")
        f.write(f"starts_hash: {manifest['hash']}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_eval: {args.num_eval}\n\n")
        for r in results:
            f.write(
                f"mode={r['mode']} offset={r['offset']} budget={r['budget']} "
                f"success_rate={r['success_rate']} "
                f"n_success={r['n_success']}/{r['n_eval']} "
                f"seconds={r['seconds']}\n"
            )
        f.write(f"\ntotal_time: {total_dt}\n")
    print(f"wrote {out_root / 'results.json'}", flush=True)
    print(f"wrote {summary_txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
