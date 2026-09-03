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

from cem_diag import CemCostTrace
from checkpoint_utils import load_fblewm_checkpoint, load_trm_head
from planning import ALL_PLANNING_MODES, BASE_MODES, FUSION_MODES
from policy import FBWorldModelPolicy, expected_replan_depths

MODES = BASE_MODES
OFFSETS = (25, 50, 75, 100)
BUDGETS = {25: 50, 50: 100, 75: 150, 100: 200}
FUSION_EVAL_MODES = FUSION_MODES


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_policy_path(policy: str, cache_dir: str) -> Path | None:
    candidates = (
        Path(policy),
        Path(cache_dir) / "checkpoints" / policy,
        ROOT / ".stable-wm" / "checkpoints" / policy,
    )
    return next((p for p in candidates if p.is_file()), None)


def collect_imaginer_diagnostics(model, device: str = "cuda") -> dict:
    """Non-invasive F/B rollout diagnostics (does not change planning)."""
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(8, 192, device=device)
        out = {"forward_norms": {}, "backward_norms": {}, "schedules": {}}
        fwd = model.forward_imaginer
        is_branch = bool(getattr(fwd, "is_branch_preserving", False))
        hist = torch.stack([z0, z0], dim=-2) if is_branch else None
        num_branches = int(getattr(fwd, "num_branches", 1)) if is_branch else 1
        for k in (1, 5, 10, 15):
            if is_branch:
                zf = model.imagine_forward_branches(hist, k)
                seed = z0.unsqueeze(-2)
                out.setdefault("branch_norms_by_k", {})[str(k)] = [
                    float(zf[:, m].float().norm(dim=-1).mean().cpu())
                    for m in range(num_branches)
                ]
                if num_branches > 1:
                    pair = zf.unsqueeze(2) - zf.unsqueeze(1)
                    dist = pair.float().norm(dim=-1)
                    eye = torch.eye(num_branches, dtype=torch.bool, device=zf.device)
                    out.setdefault("branch_pairwise_spread_by_k", {})[str(k)] = float(
                        dist[:, ~eye].mean().cpu()
                    )
                else:
                    out.setdefault("branch_pairwise_spread_by_k", {})[str(k)] = 0.0
            else:
                zf = model.imagine_forward(z0, k)
                seed = z0
            if getattr(model.backward_imaginer, "is_conditional", False):
                zb = model.imagine_backward(z0, k, z_now=z0)
            else:
                zb = model.imagine_backward(z0, k)
            out["forward_norms"][str(k)] = float(zf.float().norm(dim=-1).mean().cpu())
            out["backward_norms"][str(k)] = float(zb.float().norm(dim=-1).mean().cpu())
            # Distance of imagined latent to the seed (proxy; no GT trajectory here).
            out.setdefault("forward_dist_to_seed", {})[str(k)] = float(
                (zf - seed).float().pow(2).mean().sqrt().cpu()
            )
            out.setdefault("backward_dist_to_seed", {})[str(k)] = float(
                (zb - z0).float().pow(2).mean().sqrt().cpu()
            )
        for offset in OFFSETS:
            out["schedules"][str(offset)] = expected_replan_depths(offset)

        # Random CEM-cost probe to detect early cost collapse (B=1, S=32).
        # Action dim is frameskip-blocked (PushT=10, Cube=25), not raw env dim.
        A = int(model.action_encoder.patch_embed.in_channels)
        B, S, H = 1, 32, 5
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
        if is_branch:
            goal = torch.randn_like(z0)
            zf5 = model.imagine_forward_branches(hist, 5)
            dist_m = (zf5 - goal.unsqueeze(-2)).pow(2).sum(dim=-1)
            best = dist_m.argmin(dim=-1)
            freq = [
                float((best == m).float().mean().cpu()) for m in range(num_branches)
            ]
            out["branch_best_index_frequency"] = freq
            out["branch_cost_std"] = float(
                dist_m.min(dim=-1).values.float().std(unbiased=False).cpu()
            )
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
        "--starts-per-offset",
        action="store_true",
        help=(
            "Sample starts independently per goal offset (LeWM eval.py protocol). "
            "Official and Forward still share the same starts within one offset. "
            "Incompatible with --starts-manifest."
        ),
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
    p.add_argument(
        "--backward-depth-cap",
        type=int,
        default=None,
        help=(
            "Eval-only pred_goal recursion cap min(k, cap). "
            "Default: unset (existing checkpoints / standard modes unchanged)."
        ),
    )
    p.add_argument(
        "--forward-depth-override",
        type=int,
        default=None,
        help=(
            "Eval-only: replace Forward imagination depth with this fixed k. "
            "Default: unset (dynamic schedule). Official costs are unchanged."
        ),
    )
    p.add_argument(
        "--record-cem-cost",
        action="store_true",
        help="Write real CEM candidate-cost traces to cem_cost.jsonl (diagnostic).",
    )
    p.add_argument(
        "--eval-dir",
        default=None,
        help="Alias for --hydra-run-dir; write matrix outputs here.",
    )
    p.add_argument(
        "--trm-head",
        default=None,
        help="Standalone TRM .pt artifact; required only for trm_* modes.",
    )
    p.add_argument(
        "--trm-weight",
        type=float,
        default=1.0,
        help="Hybrid TRM coefficient (default: 1.0).",
    )
    p.add_argument(
        "--trm-eps",
        type=float,
        default=1e-8,
        help="Per-candidate z-score epsilon for trm_hybrid.",
    )
    return p.parse_args(argv)


def _expand_modes(modes_arg: str) -> tuple[str, ...]:
    raw = [m.strip() for m in modes_arg.split(",") if m.strip()]
    out: list[str] = []
    for m in raw:
        if m == "fusion":
            out.extend(FUSION_EVAL_MODES)
        elif m == "all":
            # Preserve the historical meaning of "all". TRM is opt-in only.
            out.extend(BASE_MODES + FUSION_EVAL_MODES)
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
    trm_requested = any(m in ("trm_replace", "trm_hybrid") for m in modes)
    if trm_requested and not args.trm_head:
        print("--trm-head is required for trm_replace/trm_hybrid", file=sys.stderr)
        return 2
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

    out_root = Path(
        args.eval_dir
        or args.hydra_run_dir
        or (ROOT / "outputs" / "eval" / "matrix")
    )
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
    print(f"backward_depth_cap={args.backward_depth_cap}", flush=True)
    print(f"forward_depth_override={args.forward_depth_override}", flush=True)
    print(f"starts_per_offset={bool(args.starts_per_offset)}", flush=True)
    print(f"record_cem_cost={bool(args.record_cem_cost)}", flush=True)
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
    trm_meta = None
    trm_head_hash = None
    if trm_requested:
        trm_head, trm_meta = load_trm_head(args.trm_head, cache_dir=args.cache_dir)
        artifact_path = Path(trm_meta["artifact_path"])
        trm_head_hash = _sha256_file(artifact_path)
        expected_task = trm_meta.get("task")
        if expected_task is not None and str(expected_task) != str(args.config_name):
            raise ValueError(
                f"TRM task {expected_task!r} != eval config {args.config_name!r}"
            )
        expected_base = trm_meta.get("base_checkpoint_sha256")
        policy_path = _resolve_policy_path(args.policy, args.cache_dir)
        if expected_base is not None:
            if policy_path is None:
                raise FileNotFoundError(
                    f"cannot resolve policy path for TRM hash check: {args.policy}"
                )
            actual_base = _sha256_file(policy_path)
            if str(expected_base) != actual_base:
                raise ValueError(
                    "TRM/base checkpoint hash mismatch: "
                    f"head expects {expected_base}, eval has {actual_base}"
                )
        model.set_trm_head(
            trm_head,
            weight=args.trm_weight,
            eps=args.trm_eps,
            metadata=trm_meta,
        )
    if args.backward_depth_cap is not None:
        model.set_backward_depth_cap(args.backward_depth_cap)
    if args.forward_depth_override is not None:
        model.set_forward_depth_override(args.forward_depth_override)

    diag_path = out_root / "diagnostics.json"
    if not diag_path.exists():
        diag = collect_imaginer_diagnostics(model, device="cuda")
        diag_path.write_text(json.dumps(diag, indent=2))
        print(f"wrote diagnostics.json schedules={diag['schedules']}", flush=True)
    else:
        print(f"reusing existing {diag_path.name}", flush=True)

    if args.starts_per_offset and args.starts_manifest:
        print("--starts-per-offset cannot be combined with --starts-manifest", file=sys.stderr)
        return 2

    # Shared starts (default) or one pool per offset (LeWM protocol).
    starts_path = out_root / "starts_manifest.json"
    starts_by_offset: dict[int, dict] = {}
    if args.starts_per_offset:
        offset_dir = out_root / "starts_by_offset"
        offset_dir.mkdir(parents=True, exist_ok=True)
        protocol = {
            "protocol": "per_offset",
            "seed": args.seed,
            "num_eval": args.num_eval,
            "offsets": {},
        }
        for offset in offsets:
            off_path = offset_dir / f"offset_{offset}.json"
            if args.resume and off_path.exists():
                starts_by_offset[offset] = json.loads(off_path.read_text())
                print(
                    f"reusing per-offset starts offset={offset} "
                    f"hash={starts_by_offset[offset].get('hash')}",
                    flush=True,
                )
            else:
                starts_by_offset[offset] = sample_starts_for_max_offset(
                    dataset, num_eval=args.num_eval, seed=args.seed, max_offset=offset
                )
                off_path.write_text(json.dumps(starts_by_offset[offset], indent=2))
            steps = starts_by_offset[offset]["start_steps"]
            protocol["offsets"][str(offset)] = {
                "hash": starts_by_offset[offset]["hash"],
                "path": str(off_path.relative_to(out_root)),
                "start_mean": float(np.mean(steps)),
            }
            print(
                f"per-offset starts offset={offset} n={len(steps)} "
                f"mean_start={np.mean(steps):.1f} "
                f"hash={starts_by_offset[offset]['hash'][:12]}... "
                f"head={steps[:8]}",
                flush=True,
            )
        starts_path.write_text(json.dumps(protocol, indent=2))
        manifest = {
            "hash": "per_offset",
            "protocol": "per_offset",
            "start_steps": [],
            "episodes": [],
        }
    else:
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
        for offset in offsets:
            starts_by_offset[offset] = manifest

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
        cem_trace = CemCostTrace() if args.record_cem_cost else None
        solver_kw = {"model": model}
        if cem_trace is not None:
            solver_kw["callbacks"] = [cem_trace]
        solver = hydra.utils.instantiate(base_cfg.solver, **solver_kw)
        model.set_planning_mode(mode)
        policy = FBWorldModelPolicy(
            solver=solver,
            config=plan_config,
            goal_offset=offset,
            planning_mode=mode,
            process=process,
            transform=transform,
            backward_depth_cap=args.backward_depth_cap,
            forward_depth_override=args.forward_depth_override,
        )
        world.set_policy(policy)

        video_dir = out_root / "videos" / mode / f"offset_{offset}"
        video_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        unit_starts = starts_by_offset[offset]
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=unit_starts["start_steps"],
            goal_offset=offset,
            eval_budget=budget,
            episodes_idx=unit_starts["episodes"],
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
            "starts_hash": unit_starts["hash"],
            "starts_protocol": "per_offset" if args.starts_per_offset else "shared_max_offset",
            "policy": args.policy,
            "seed": args.seed,
            "horizon": int(base_cfg.plan_config.horizon),
            "plan_len": plan_len,
            "backward_depth_cap": args.backward_depth_cap,
            "forward_depth_override": args.forward_depth_override,
            "metrics": {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in metrics.items()
            },
        }
        if mode in ("trm_replace", "trm_hybrid"):
            row["trm"] = {
                "head": str(args.trm_head),
                "head_sha256": trm_head_hash,
                "weight": float(args.trm_weight),
                "eps": float(args.trm_eps),
                "label_type": trm_meta.get("label_type") if trm_meta else None,
                "base_checkpoint_sha256": (
                    trm_meta.get("base_checkpoint_sha256") if trm_meta else None
                ),
            }
        if cem_trace is not None:
            cem_summary = cem_trace.summarize()
            row["cem_cost_summary"] = cem_summary
            cem_path = out_root / "cem_cost.jsonl"
            with cem_path.open("a") as cf:
                cf.write(
                    json.dumps(
                        {
                            "diagnostic": True,
                            "mode": mode,
                            "offset": offset,
                            "backward_depth_cap": args.backward_depth_cap,
                            "summary": cem_summary,
                            "n_records": len(cem_trace.records),
                        }
                    )
                    + "\n"
                )
            detail_path = out_root / "cem_cost" / f"{mode}_offset_{offset}.json"
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            detail_path.write_text(
                json.dumps(
                    {
                        "diagnostic": True,
                        "mode": mode,
                        "offset": offset,
                        "backward_depth_cap": args.backward_depth_cap,
                        "summary": cem_summary,
                        "records": cem_trace.records,
                    }
                )
            )
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
        "starts_hash": manifest.get("hash"),
        "starts_protocol": "per_offset" if args.starts_per_offset else "shared_max_offset",
        "starts_by_offset": (
            {str(o): starts_by_offset[o]["hash"] for o in offsets}
            if args.starts_per_offset
            else None
        ),
        "backward_depth_cap": args.backward_depth_cap,
        "forward_depth_override": args.forward_depth_override,
        "record_cem_cost": bool(args.record_cem_cost),
        "total_time": total_dt,
        "results": results,
    }
    if trm_requested:
        summary["trm"] = {
            "head": str(args.trm_head),
            "head_sha256": trm_head_hash,
            "weight": float(args.trm_weight),
            "eps": float(args.trm_eps),
            "metadata": trm_meta,
        }
    (out_root / "results.json").write_text(json.dumps(summary, indent=2))
    summary_txt = out_root / "summary.txt"
    with summary_txt.open("w") as f:
        f.write("==== FBLeWM MATRIX SUMMARY ====\n")
        f.write(f"policy: {args.policy}\n")
        f.write(
            f"starts_protocol: "
            f"{'per_offset' if args.starts_per_offset else 'shared_max_offset'}\n"
        )
        f.write(f"starts_hash: {manifest.get('hash')}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_eval: {args.num_eval}\n")
        f.write(f"horizon: {int(base_cfg.plan_config.horizon)}\n")
        f.write(f"plan_len: {plan_len}\n")
        f.write(f"backward_depth_cap: {args.backward_depth_cap}\n")
        f.write(f"forward_depth_override: {args.forward_depth_override}\n")
        f.write(f"record_cem_cost: {bool(args.record_cem_cost)}\n\n")
        if trm_requested:
            f.write(f"trm_head: {args.trm_head}\n")
            f.write(f"trm_head_sha256: {trm_head_hash}\n")
            f.write(f"trm_weight: {args.trm_weight}\n")
            f.write(f"trm_eps: {args.trm_eps}\n\n")
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
