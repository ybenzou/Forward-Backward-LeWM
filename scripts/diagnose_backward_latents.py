#!/usr/bin/env python3
"""Read-only pred_goal latent-trace probe.

Encodes real z_plan_end (t+25) and z_goal from the shared starts_manifest,
then records B_P^i(z_g) norms / displacements / shuffled-goal separation.
Does not interact with the environment and is not used for planning.
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

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir
from torchvision.transforms import v2 as transforms

from checkpoint_utils import load_fblewm_checkpoint
from policy import compute_imagine_steps


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Diagnostic latent traces for pred_goal B")
    p.add_argument("--policy", required=True)
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("STABLEWM_HOME", str(ROOT / ".stable-wm")),
    )
    p.add_argument("--config-name", default="pusht")
    p.add_argument("--starts-manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--offsets", default="50,75,100")
    p.add_argument("--plan-len", type=int, default=25)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _episode_col(dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def _lookup_rows(dataset, episodes, start_steps, delta: int) -> list[int]:
    col = _episode_col(dataset)
    ep_all = np.asarray(dataset.get_col_data(col))
    step_all = np.asarray(dataset.get_col_data("step_idx"))
    rows = []
    for ep, start in zip(episodes, start_steps):
        target = int(start) + int(delta)
        hits = np.nonzero((ep_all == ep) & (step_all == target))[0]
        if len(hits) == 0:
            raise KeyError(f"no dataset row for episode={ep} step={target}")
        rows.append(int(hits[0]))
    return rows


def _load_pixels(dataset, rows: list[int], transform) -> torch.Tensor:
    raw = dataset.get_row_data(rows)["pixels"]
    imgs = []
    for i in range(len(rows)):
        pix = raw[i]
        if torch.is_tensor(pix):
            pix = pix.cpu().numpy()
        imgs.append(transform(pix))
    return torch.stack(imgs, dim=0)


def _encode_pixels(model, pixels: torch.Tensor) -> torch.Tensor:
    info = model.encode({"pixels": pixels.unsqueeze(1)})
    return info["emb"][:, -1, :]


DEPTH_PROBE_KS = (1, 5, 10, 15)


def _trace_one_batch(model, p: torch.Tensor, z_goal: torch.Tensor, k: int) -> dict:
    norms = []
    dist_to_p = []
    step_disp = []
    prev = None
    g = z_goal
    for i in range(k + 1):
        norms.append(g.float().norm(dim=-1).cpu())
        dist_to_p.append((g - p).float().pow(2).sum(dim=-1).sqrt().cpu())
        if prev is None:
            step_disp.append(torch.zeros(g.size(0)))
        else:
            step_disp.append((g - prev).float().pow(2).sum(dim=-1).sqrt().cpu())
        prev = g
        if i < k:
            g = model.imagine_backward(g, 1, z_now=p)
    return {
        "norm": torch.stack(norms, dim=1),
        "dist_to_p": torch.stack(dist_to_p, dim=1),
        "step_disp": torch.stack(step_disp, dim=1),
        "final": g if k == 0 else model.imagine_backward(z_goal, k, z_now=p),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    offsets = tuple(int(x.strip()) for x in args.offsets.split(",") if x.strip())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.starts_manifest).read_text())
    episodes = list(manifest["episodes"])
    start_steps = list(manifest["start_steps"])

    config_dir = str((ROOT / "config" / "eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name)

    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=Path(args.cache_dir),
    )
    transform = img_transform(int(cfg.eval.img_size))
    model = load_fblewm_checkpoint(args.policy, cache_dir=args.cache_dir)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if hasattr(model, "set_backward_depth_cap"):
        model.set_backward_depth_cap(None)

    plan_rows = _lookup_rows(dataset, episodes, start_steps, args.plan_len)
    with torch.no_grad():
        p_pix = _load_pixels(dataset, plan_rows, transform).to("cuda")
        p = _encode_pixels(model, p_pix)

    g = torch.Generator(device="cuda")
    g.manual_seed(int(args.seed))
    perm = torch.randperm(p.size(0), generator=g, device=p.device)

    offset_out = {}
    for offset in offsets:
        k = compute_imagine_steps(offset, 0, args.plan_len, args.action_block)
        goal_rows = _lookup_rows(dataset, episodes, start_steps, offset)
        with torch.no_grad():
            z_goal = _encode_pixels(
                model, _load_pixels(dataset, goal_rows, transform).to("cuda")
            )
            z_shuf = z_goal[perm]
            p_shuf = p[perm]
            true_tr = _trace_one_batch(model, p, z_goal, k)
            shuf_tr = _trace_one_batch(model, p, z_shuf, k)
            p_shuf_tr = _trace_one_batch(model, p_shuf, z_goal, k)
            sep = (true_tr["final"] - shuf_tr["final"]).float().pow(2).sum(dim=-1).sqrt()
            p_sens = (
                (true_tr["final"] - p_shuf_tr["final"]).float().pow(2).sum(dim=-1).sqrt()
            )
            identity = model.imagine_backward(p, 1, z_now=p)
            identity_gap = (identity - p).float().pow(2).sum(dim=-1).sqrt()
            depth_probe = {}
            for pk in DEPTH_PROBE_KS:
                d_true = _trace_one_batch(model, p, z_goal, pk)
                d_shuf = _trace_one_batch(model, p, z_shuf, pk)
                d_sep = (
                    (d_true["final"] - d_shuf["final"]).float().pow(2).sum(dim=-1).sqrt()
                )
                depth_probe[str(pk)] = {
                    "dist_to_p": {
                        "mean": float(d_true["dist_to_p"][:, -1].float().mean().cpu()),
                        "std": float(
                            d_true["dist_to_p"][:, -1].float().std(unbiased=False).cpu()
                        ),
                    },
                    "goal_separation": {
                        "mean": float(d_sep.mean().cpu()),
                        "std": float(d_sep.std(unbiased=False).cpu()),
                    },
                }

        def _mean_std(t: torch.Tensor) -> dict:
            t = t.float()
            return {
                "mean": t.mean(dim=0).cpu().tolist(),
                "std": t.std(dim=0, unbiased=False).cpu().tolist(),
            }

        offset_out[str(offset)] = {
            "k": int(k),
            "n": int(p.size(0)),
            "norm": _mean_std(true_tr["norm"]),
            "dist_to_p": _mean_std(true_tr["dist_to_p"]),
            "step_disp": _mean_std(true_tr["step_disp"]),
            "shuffled_dist_to_p": _mean_std(shuf_tr["dist_to_p"]),
            "goal_separation_final": {
                "mean": float(sep.mean().cpu()),
                "std": float(sep.std(unbiased=False).cpu()),
            },
            "identity_gap": {
                "mean": float(identity_gap.mean().cpu()),
                "std": float(identity_gap.std(unbiased=False).cpu()),
            },
            "true_closer_than_shuffled": float(
                (true_tr["dist_to_p"][:, -1] < shuf_tr["dist_to_p"][:, -1])
                .float()
                .mean()
                .cpu()
            ),
            "goal_sensitivity": {
                "mean": float(sep.mean().cpu()),
                "std": float(sep.std(unbiased=False).cpu()),
            },
            "p_sensitivity": {
                "mean": float(p_sens.mean().cpu()),
                "std": float(p_sens.std(unbiased=False).cpu()),
            },
            "depth_probe": depth_probe,
        }
        print(
            f"offset={offset} k={k} identity_gap={identity_gap.mean():.4f} "
            f"sep={sep.mean():.4f} p_sens={p_sens.mean():.4f} true<shuf="
            f"{offset_out[str(offset)]['true_closer_than_shuffled']:.2f}",
            flush=True,
        )

    payload = {
        "diagnostic": True,
        "policy": args.policy,
        "starts_hash": manifest.get("hash"),
        "starts_manifest": str(Path(args.starts_manifest).resolve()),
        "config_name": args.config_name,
        "plan_len": args.plan_len,
        "action_block": args.action_block,
        "seed": args.seed,
        "backward_anchor": getattr(model, "backward_anchor", None),
        "offsets": offset_out,
    }
    out_path = out_dir / "latent_trace.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
