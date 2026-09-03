#!/usr/bin/env python3
"""Resumable TRM evaluation using frozen FBLeWM checkpoints and manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "baselines" / "trm_compare" / "v1" / "eval"
DEFAULT_MANIFEST_ROOT = ROOT / "outputs" / "diag" / "iclr_bar" / "multiseed"
DEFAULT_SEEDS = tuple(range(42, 52))

TASKS = {
    "tworoom": {
        "policy": "fblewm_tworoom/weights_epoch_10.pt",
        "config": "tworoom",
        "mode": "trm_replace",
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate opt-in TRM costs with existing FBLeWM manifests"
    )
    parser.add_argument("--task", choices=tuple(TASKS), default="tworoom")
    parser.add_argument("--trm-head", required=True)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument("--offsets", default="25,50,75,100")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--mode", choices=("trm_replace", "trm_hybrid"), default=None)
    parser.add_argument("--trm-weight", type=float, default=1.0)
    parser.add_argument("--trm-eps", type=float, default=1e-8)
    parser.add_argument("--policy", default=None)
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("STABLEWM_HOME", str(ROOT / ".stable-wm")),
    )
    parser.add_argument("--manifest-root", default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Read completed results.json files without running evaluation.",
    )
    return parser.parse_args(argv)


def _summarize(task_root: Path) -> int:
    rows = []
    for path in sorted(task_root.glob("seed_*/results.json")):
        payload = json.loads(path.read_text())
        seed = int(path.parent.name.removeprefix("seed_"))
        for row in payload.get("results", []):
            rows.append(
                (
                    seed,
                    row["mode"],
                    int(row["offset"]),
                    float(row["success_rate"]),
                    int(row["n_success"]),
                    int(row["n_eval"]),
                )
            )
    if not rows:
        print(f"no completed results under {task_root}", file=sys.stderr)
        return 1
    print(f"{'seed':>5} {'mode':>14} {'offset':>6} {'success':>8} {'ok/n':>9}")
    for seed, mode, offset, success, n_ok, n_eval in rows:
        print(
            f"{seed:5d} {mode:>14} {offset:6d} "
            f"{success:7.1f}% {n_ok:3d}/{n_eval:<3d}"
        )
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = TASKS[args.task]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    mode = args.mode or spec["mode"]
    policy = args.policy or spec["policy"]
    trm_head = Path(args.trm_head).expanduser().resolve()
    if not trm_head.is_file():
        print(f"TRM head not found: {trm_head}", file=sys.stderr)
        return 2

    task_root = Path(args.out_root).expanduser().resolve() / args.task
    if args.summarize_only:
        return _summarize(task_root)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    task_root.mkdir(parents=True, exist_ok=True)
    log_path = task_root / "run.log"
    manifest_root = Path(args.manifest_root).expanduser().resolve()
    matrix = ROOT / "scripts" / "eval_fblewm_matrix.py"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a") as handle:
            handle.write(message + "\n")

    log(f"==== TRM {args.task} START {datetime.now().isoformat(timespec='seconds')} ====")
    log(f"policy={policy} head={trm_head} mode={mode}")
    log(
        f"seeds={seeds} offsets={args.offsets} num_eval={args.num_eval} "
        f"cache_dir={args.cache_dir}"
    )

    for index, seed in enumerate(seeds, start=1):
        manifest = manifest_root / args.task / f"seed_{seed}" / "starts_manifest.json"
        if not manifest.is_file():
            print(f"missing frozen manifest: {manifest}", file=sys.stderr)
            return 2
        destination = task_root / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        log(f"==== RUN {index}/{len(seeds)} seed={seed} ====")
        command = [
            sys.executable,
            str(matrix),
            f"--policy={policy}",
            f"--cache-dir={args.cache_dir}",
            f"--config-name={spec['config']}",
            f"--modes={mode}",
            f"--offsets={args.offsets}",
            f"--seed={seed}",
            f"--num-eval={args.num_eval}",
            f"--starts-manifest={manifest}",
            f"--eval-dir={destination}",
            f"--trm-head={trm_head}",
            f"--trm-weight={args.trm_weight}",
            f"--trm-eps={args.trm_eps}",
            "--resume",
        ]
        log("$ " + " ".join(command))
        result = subprocess.call(command, cwd=str(ROOT), env=os.environ.copy())
        if result != 0:
            log(f"FAIL seed={seed} exit={result}")
            return result
        log(f"==== FINISH {index}/{len(seeds)} seed={seed} ====")

    log(f"==== TRM {args.task} ALL DONE {datetime.now().isoformat(timespec='seconds')} ====")
    return _summarize(task_root)


if __name__ == "__main__":
    raise SystemExit(main())
