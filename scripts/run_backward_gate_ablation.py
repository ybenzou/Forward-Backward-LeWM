#!/usr/bin/env python3
"""Run pred_goal diagnostic probes + same-start cap/meet ablations.

Writes only under outputs/diag/backward_gate_v3/. Does not overwrite
official v3 eval dirs or paper figures.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "pusht": {
        "policy": "fblewm_pusht_v3/weights_epoch_10.pt",
        "config": "pusht",
        "starts": ROOT / "outputs" / "eval" / "20260817_115825_pusht" / "starts_manifest.json",
    },
    "tworoom": {
        "policy": "fblewm_tworoom_v3/weights_epoch_10.pt",
        "config": "tworoom",
        "starts": ROOT / "outputs" / "eval" / "20260817_035625_tworoom" / "starts_manifest.json",
    },
}

DIAG_ROOT = ROOT / "outputs" / "diag" / "backward_gate_v3"
VARIANTS = (
    ("eval_cap1", ["--eval-modes=backward", "--backward-depth-cap=1"]),
    ("eval_cap2", ["--eval-modes=backward", "--backward-depth-cap=2"]),
    ("eval_cap5", ["--eval-modes=backward", "--backward-depth-cap=5"]),
    ("eval_meet", ["--eval-modes=meet"]),
)


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Backward diagnostic-gate runner")
    p.add_argument("--tasks", default="pusht,tworoom")
    p.add_argument("--skip-latent", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--only-variant", default=None, help="eval_cap1|eval_cap2|eval_cap5|eval_meet")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    python = sys.executable
    cache = ROOT / ".stable-wm"

    for task in tasks:
        spec = TASKS[task]
        if not spec["starts"].exists():
            print(f"missing starts manifest: {spec['starts']}", file=sys.stderr)
            return 2
        task_root = DIAG_ROOT / task
        task_root.mkdir(parents=True, exist_ok=True)

        if not args.skip_latent:
            latent_dir = task_root / "latent"
            rc = _run(
                [
                    python,
                    str(ROOT / "scripts" / "diagnose_backward_latents.py"),
                    f"--policy={spec['policy']}",
                    f"--cache-dir={cache}",
                    f"--config-name={spec['config']}",
                    f"--starts-manifest={spec['starts']}",
                    f"--out-dir={latent_dir}",
                ]
            )
            if rc != 0:
                return rc

        if args.skip_eval:
            continue
        variants = VARIANTS
        if args.only_variant:
            variants = tuple(v for v in VARIANTS if v[0] == args.only_variant)
            if not variants:
                print(f"unknown variant {args.only_variant}", file=sys.stderr)
                return 2
        for name, extra in variants:
            eval_dir = task_root / name
            eval_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                python,
                str(ROOT / "scripts" / "run_fblewm_pipeline.py"),
                f"--task={task}",
                "--skip-deps",
                "--from-stage=eval",
                f"--policy={spec['policy']}",
                f"--starts-manifest={spec['starts']}",
                "--eval-offsets=50,75,100",
                f"--eval-dir={eval_dir}",
                "--record-cem-cost",
                *extra,
            ]
            rc = _run(cmd)
            if rc != 0:
                return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
