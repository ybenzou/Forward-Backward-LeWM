#!/usr/bin/env python3
"""10-group fixed-k Forward evals (seeds 42-51), paired with the main manifests.

k=0 and dynamic are not re-run: they are the official / HAS columns already
in outputs/diag/iclr_bar/multiseed/. This script fills k=5, k=10, and k=15.
Resume-safe. Seed-42 copies the existing single-group k-ablation dirs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MATRIX = ROOT / "scripts" / "eval_fblewm_matrix.py"
MULTI_ROOT = ROOT / "outputs" / "diag" / "iclr_bar" / "multiseed"
SEED42_ROOT = ROOT / "outputs" / "diag" / "k_ablation"
OUT_ROOT = ROOT / "outputs" / "diag" / "k_ablation" / "multiseed"

TASKS = {
    "pusht": "fblewm/weights_epoch_10.pt",
    "tworoom": "fblewm_tworoom/weights_epoch_10.pt",
    "reacher": "fblewm_reacher_v1/weights_epoch_10.pt",
}
CONDS = (("k5", 5), ("k10", 10), ("k15", 15))
DEFAULT_SEEDS = tuple(range(42, 52))


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _has_summary(dest: Path) -> bool:
    return (dest / "summary.txt").exists()


def _link_or_copy_seed42(task: str, cond: str, dest: Path) -> bool:
    src = SEED42_ROOT / task / cond
    if not (src / "summary.txt").exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return _has_summary(dest)
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copytree(src, dest)
    return _has_summary(dest)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="pusht,tworoom,reacher")
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--offsets", default="75,100")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / "run.log"
    jobs = [(task, cond, k, seed) for task in tasks for cond, k in CONDS for seed in seeds]
    n_total = len(jobs)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"==== K-ABLATION MULTI-SEED START {_stamp()} ====")
    log(f"jobs={n_total} tasks={tasks} seeds={seeds} conds={[c for c, _ in CONDS]}")

    for i, (task, cond, k, seed) in enumerate(jobs, start=1):
        dest = OUT_ROOT / task / cond / f"seed_{seed}"
        manifest = MULTI_ROOT / task / f"seed_{seed}" / "starts_manifest.json"
        log(f"==== RUN {i}/{n_total} {task} {cond} seed={seed} {_stamp()} ====")
        if not manifest.exists():
            log(f"FAIL missing manifest {manifest}")
            return 2
        if seed == 42 and _link_or_copy_seed42(task, cond, dest):
            log(f"SKIP seed-42 reuse {SEED42_ROOT / task / cond}")
            continue
        if _has_summary(dest):
            log(f"SKIP already has summary.txt: {dest}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [
            PY,
            str(MATRIX),
            "--policy",
            TASKS[task],
            "--config-name",
            task,
            "--modes",
            "forward",
            "--offsets",
            args.offsets,
            "--seed",
            str(seed),
            "--starts-manifest",
            str(manifest),
            "--eval-dir",
            str(dest),
            "--forward-depth-override",
            str(k),
            "--resume",
        ]
        log("$ " + " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
        if rc != 0:
            log(f"FAIL {task} {cond} seed={seed} exit={rc}")
            return rc
        log(f"==== FINISH {i}/{n_total} {task} {cond} seed={seed} ====")

    log(f"==== K-ABLATION MULTI-SEED ALL DONE {_stamp()} ====")
    plot = ROOT / "scripts" / "plot_k_depth_multiseed.py"
    if plot.exists():
        subprocess.call([PY, str(plot)], cwd=str(ROOT), env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
