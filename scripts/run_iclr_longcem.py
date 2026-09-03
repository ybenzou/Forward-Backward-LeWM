#!/usr/bin/env python3
"""10-seed Official long-CEM eval for the ICLR-bar control.

Reuses starts from outputs/diag/iclr_bar/multiseed/{task}/seed_XX/.
Writes only under outputs/diag/iclr_bar/longcem/h{H}/.
Does not overwrite the h=5 Official/HAS multiseed dirs.

Default: horizon=10 (plan_len=50), offsets 25/50/75/100, mode=official.
horizon=15 skips offset 25 because default budget 50 < plan_len 75.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MULTI_ROOT = ROOT / "outputs" / "diag" / "iclr_bar" / "multiseed"
OUT_ROOT = ROOT / "outputs" / "diag" / "iclr_bar" / "longcem"

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

DEFAULT_SEEDS = tuple(range(42, 52))
OFFSETS = (25, 50, 75, 100)
BUDGETS = {25: 50, 50: 100, 75: 150, 100: 200}
ACTION_BLOCK = 5


def offsets_for_horizon(horizon: int) -> tuple[int, ...]:
    plan_len = int(horizon) * ACTION_BLOCK
    return tuple(o for o in OFFSETS if BUDGETS[o] >= plan_len)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ICLR-bar 10-seed Official long-CEM runner")
    p.add_argument("--tasks", default="pusht,tworoom")
    p.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated eval seeds (default: 42-51)",
    )
    p.add_argument(
        "--horizons",
        default="10",
        help="Comma-separated CEM latent horizons (default: 10). Try 10,15.",
    )
    p.add_argument(
        "--starts-root",
        default=str(MULTI_ROOT),
        help="Existing 10-seed Official/HAS dirs whose starts_manifest.json are reused",
    )
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Print a table from existing longcem summary.txt files and exit",
    )
    return p.parse_args(argv)


def summarize(out_root: Path) -> int:
    rows = []
    for p in sorted(out_root.glob("h*/*/seed_*/summary.txt")):
        horizon = p.parts[-4]
        task = p.parts[-3]
        seed = p.parts[-2].split("_")[1]
        vals: dict[int, float] = {}
        for line in p.read_text().splitlines():
            if not line.startswith("mode="):
                continue
            parts = dict(x.split("=", 1) for x in line.split() if "=" in x)
            if parts.get("mode") != "official":
                continue
            vals[int(parts["offset"])] = float(parts["success_rate"])
        rows.append((horizon, task, seed, vals))
    if not rows:
        print(f"no summary.txt under {out_root}", file=sys.stderr)
        return 1
    print(f"{'h':4} {'task':8} {'seed':5} {'25':>6} {'50':>6} {'75':>6} {'100':>6}")
    for horizon, task, seed, r in rows:
        print(
            f"{horizon:4} {task:8} {seed:5} "
            f"{r.get(25, float('nan')):6.1f} {r.get(50, float('nan')):6.1f} "
            f"{r.get(75, float('nan')):6.1f} {r.get(100, float('nan')):6.1f}"
        )
    return 0


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_eval_cmd(
    *,
    python: str,
    matrix: Path,
    policy: str,
    config: str,
    cache: str,
    seed: int,
    dest: Path,
    starts: Path,
    horizon: int,
    offsets: tuple[int, ...],
    num_eval: int,
) -> list[str]:
    return [
        python,
        str(matrix),
        f"--policy={policy}",
        f"--cache-dir={cache}",
        f"--config-name={config}",
        "--modes=official",
        f"--offsets={','.join(str(o) for o in offsets)}",
        f"--seed={seed}",
        f"--num-eval={num_eval}",
        f"--eval-dir={dest}",
        f"--starts-manifest={starts}",
        f"--horizon={horizon}",
        f"--receding-horizon={horizon}",
        "--resume",
    ]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.summarize_only:
        return summarize(Path(args.out_root))

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    for t in tasks:
        if t not in TASKS:
            print(f"unknown task {t!r}; expected {list(TASKS)}", file=sys.stderr)
            return 2
    if not horizons:
        print("no horizons", file=sys.stderr)
        return 2

    starts_root = Path(args.starts_root)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cache = os.environ.get("STABLEWM_HOME", str(ROOT / ".stable-wm"))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"

    n_total = len(tasks) * len(seeds) * len(horizons)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"==== LONG-CEM START {_stamp()} ====")
    log(f"tasks={tasks} seeds={seeds} horizons={horizons}")
    log(f"starts_root={starts_root}")
    log(f"out_root={out_root}")
    log(f"total_runs={n_total}")

    done_n = 0
    python = sys.executable
    matrix = ROOT / "scripts" / "eval_fblewm_matrix.py"

    for horizon in horizons:
        offs = offsets_for_horizon(horizon)
        if not offs:
            log(f"FAIL horizon={horizon}: no offset has budget >= plan_len={horizon * ACTION_BLOCK}")
            return 2
        for t_i, task in enumerate(tasks, start=1):
            spec = TASKS[task]
            for s_i, seed in enumerate(seeds, start=1):
                done_n += 1
                starts = starts_root / task / f"seed_{seed}" / "starts_manifest.json"
                dest = out_root / f"h{horizon}" / task / f"seed_{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                log("")
                log(
                    f"==== RUN {done_n}/{n_total} | h={horizon} plan_len={horizon * ACTION_BLOCK} "
                    f"| task {t_i}/{len(tasks)}={task} | seed {s_i}/{len(seeds)}={seed} "
                    f"| {_stamp()} ===="
                )
                if not starts.is_file():
                    log(f"FAIL missing starts manifest (run 10-seed HAS eval first): {starts}")
                    return 2
                if (dest / "summary.txt").exists():
                    log(f"SKIP already has summary.txt: {dest}")
                    continue

                cmd = build_eval_cmd(
                    python=python,
                    matrix=matrix,
                    policy=spec["policy"],
                    config=spec["config"],
                    cache=cache,
                    seed=seed,
                    dest=dest,
                    starts=starts,
                    horizon=horizon,
                    offsets=offs,
                    num_eval=args.num_eval,
                )
                log("$ " + " ".join(cmd))
                rc = subprocess.call(cmd, cwd=str(ROOT))
                if rc != 0:
                    log(f"FAIL {task} seed={seed} h={horizon} exit={rc}")
                    return rc
                log(f"==== FINISH {done_n}/{n_total} {task} seed={seed} h={horizon} ====")
                summary = dest / "summary.txt"
                if summary.exists():
                    log(summary.read_text().rstrip())

    log(f"==== LONG-CEM ALL DONE {_stamp()} ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
