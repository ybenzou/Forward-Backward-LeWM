#!/usr/bin/env python3
"""Multi-seed Official vs Forward eval for ICLR-bar statistics.

Frozen encoder-B checkpoints. Each seed resamples its own starts
(no shared paper manifest). Writes only under outputs/diag/iclr_bar/multiseed/.
Does not overwrite official eval dirs.

Progress:
  RUN i/N | task a/A=pusht | seed b/B=42
  then eval_fblewm_matrix.py prints eval k/8 START|DONE
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "diag" / "iclr_bar" / "multiseed"

TASKS = {
    "pusht": {
        "policy": "fblewm/weights_epoch_10.pt",
        "config": "pusht",
    },
    "tworoom": {
        "policy": "fblewm_tworoom/weights_epoch_10.pt",
        "config": "tworoom",
    },
    "reacher": {
        "policy": "fblewm_reacher_v1/weights_epoch_10.pt",
        "config": "reacher",
    },
}

DEFAULT_SEEDS = tuple(range(42, 52))  # 42 .. 51


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ICLR-bar 10-seed Official/Forward runner")
    p.add_argument("--tasks", default="pusht,tworoom")
    p.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated eval seeds (default: 42-51)",
    )
    p.add_argument("--modes", default="official,forward")
    p.add_argument("--offsets", default="25,50,75,100")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument(
        "--out-root",
        default=str(OUT_ROOT),
        help="Root for task/seed_XX directories",
    )
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Print a table from existing summary.txt files and exit",
    )
    return p.parse_args(argv)


def summarize(out_root: Path) -> int:
    rows = []
    for p in sorted(out_root.glob("*/seed_*/summary.txt")):
        task = p.parts[-3]
        seed = p.parts[-2].split("_")[1]
        vals: dict[str, dict[int, float]] = {}
        for line in p.read_text().splitlines():
            if not line.startswith("mode="):
                continue
            parts = dict(x.split("=", 1) for x in line.split() if "=" in x)
            vals.setdefault(parts["mode"], {})[int(parts["offset"])] = float(
                parts["success_rate"]
            )
        for mode, r in vals.items():
            rows.append((task, seed, mode, r))
    if not rows:
        print(f"no summary.txt under {out_root}", file=sys.stderr)
        return 1
    print(f"{'task':8} {'seed':5} {'mode':10} {'25':>6} {'50':>6} {'75':>6} {'100':>6}")
    for task, seed, mode, r in rows:
        print(
            f"{task:8} {seed:5} {mode:10} "
            f"{r.get(25, float('nan')):6.1f} {r.get(50, float('nan')):6.1f} "
            f"{r.get(75, float('nan')):6.1f} {r.get(100, float('nan')):6.1f}"
        )
    return 0


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.summarize_only:
        return summarize(Path(args.out_root))
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for t in tasks:
        if t not in TASKS:
            print(f"unknown task {t!r}; expected {list(TASKS)}", file=sys.stderr)
            return 2

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cache = os.environ.get("STABLEWM_HOME", str(ROOT / ".stable-wm"))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    n_task = len(tasks)
    n_seed = len(seeds)
    n_total = n_task * n_seed
    n_units = len([m for m in args.modes.split(",") if m.strip()]) * len(
        [o for o in args.offsets.split(",") if o.strip()]
    )

    def log(msg: str) -> None:
        line = msg if msg.startswith("====") or msg.startswith("SKIP") or msg.startswith("$") else msg
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(f"==== MULTI-SEED START {_stamp()} ====")
    log(f"tasks={tasks} seeds={seeds}")
    log(f"modes={args.modes} offsets={args.offsets} num_eval={args.num_eval}")
    log(f"total_runs={n_total} units_per_run={n_units} (watch: RUN i/{n_total} then eval k/{n_units})")
    log(f"out_root={out_root}")

    done_n = 0
    python = sys.executable
    matrix = ROOT / "scripts" / "eval_fblewm_matrix.py"

    for t_i, task in enumerate(tasks, start=1):
        spec = TASKS[task]
        for s_i, seed in enumerate(seeds, start=1):
            done_n += 1
            dest = out_root / task / f"seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            log("")
            log(
                f"==== RUN {done_n}/{n_total} | task {t_i}/{n_task}={task} "
                f"| seed {s_i}/{n_seed}={seed} | {_stamp()} ===="
            )
            if (dest / "summary.txt").exists():
                log(f"SKIP already has summary.txt: {dest}")
                continue

            cmd = [
                python,
                str(matrix),
                f"--policy={spec['policy']}",
                f"--cache-dir={cache}",
                f"--config-name={spec['config']}",
                f"--modes={args.modes}",
                f"--offsets={args.offsets}",
                f"--seed={seed}",
                f"--num-eval={args.num_eval}",
                f"--eval-dir={dest}",
                "--resume",
            ]
            log("$ " + " ".join(cmd))
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc != 0:
                log(f"FAIL {task} seed={seed} exit={rc}")
                return rc
            log(f"==== FINISH {done_n}/{n_total} {task} seed={seed} ====")
            summary = dest / "summary.txt"
            if summary.exists():
                log(summary.read_text().rstrip())

    log(f"==== MULTI-SEED ALL DONE {_stamp()} ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
