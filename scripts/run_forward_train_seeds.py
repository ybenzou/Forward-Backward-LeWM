#!/usr/bin/env python3
"""Train 3 latent-Forward seeds on PushT / TwoRoom / Reacher, then eval.

Each (task, train_seed) writes its own checkpoint dir and eval tree.
Does not write into protected paper checkpoints. Reuses frozen ICLR
starts from outputs/diag/iclr_bar/multiseed/. Resume-safe.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "run_fblewm_pipeline.py"
MATRIX = ROOT / "scripts" / "eval_fblewm_matrix.py"
MULTI_ROOT = ROOT / "outputs" / "diag" / "iclr_bar" / "multiseed"
OUT_ROOT = ROOT / "outputs" / "diag" / "train_seeds"

TASKS = ("pusht", "tworoom", "reacher")
DEFAULT_TRAIN_SEEDS = (3072, 3073, 3074)
DEFAULT_EVAL_SEEDS = tuple(range(42, 52))
PROTECTED_RUN_NAMES = frozenset(
    {
        "fblewm",
        "fblewm_bp",
        "fblewm_tworoom",
        "fblewm_cube",
        "fblewm_reacher_v1",
    }
)
PAPER_POLICY = {
    "pusht": "fblewm/weights_epoch_10.pt",
    "tworoom": "fblewm_tworoom/weights_epoch_10.pt",
    "reacher": "fblewm_reacher_v1/weights_epoch_10.pt",
}


def run_name(task: str, train_seed: int) -> str:
    name = f"fblewm_{task}_s{int(train_seed)}"
    if name in PROTECTED_RUN_NAMES:
        raise ValueError(f"refusing protected output_model_name={name!r}")
    return name


def policy_for(task: str, train_seed: int, reuse_s3072: bool) -> str:
    if reuse_s3072 and int(train_seed) == 3072:
        return PAPER_POLICY[task]
    return f"{run_name(task, train_seed)}/weights_epoch_10.pt"


def starts_path(starts_root: Path, task: str, eval_seed: int) -> Path:
    return starts_root / task / f"seed_{int(eval_seed)}" / "starts_manifest.json"


def eval_dest(out_root: Path, task: str, train_seed: int, eval_seed: int) -> Path:
    return out_root / task / f"s{int(train_seed)}" / f"seed_{int(eval_seed)}"


def ckpt_path(cache_dir: Path, policy: str) -> Path:
    return cache_dir / "checkpoints" / policy


def build_train_cmd(
    *,
    python: str,
    task: str,
    train_seed: int,
    epochs: int,
    skip_deps: bool,
) -> list[str]:
    cmd = [
        python,
        str(PIPELINE),
        f"--task={task}",
        f"--seed={int(train_seed)}",
        f"--epochs={int(epochs)}",
        f"--train-run-name={run_name(task, train_seed)}",
        "--forward-variant=latent",
        "--skip-eval",
    ]
    if skip_deps:
        cmd.append("--skip-deps")
    return cmd


def build_eval_cmd(
    *,
    python: str,
    task: str,
    policy: str,
    cache: str,
    eval_seed: int,
    dest: Path,
    starts: Path,
    offsets: str,
    num_eval: int,
) -> list[str]:
    return [
        python,
        str(MATRIX),
        f"--policy={policy}",
        f"--cache-dir={cache}",
        f"--config-name={task}",
        "--modes=forward",
        f"--offsets={offsets}",
        f"--seed={int(eval_seed)}",
        f"--num-eval={int(num_eval)}",
        f"--eval-dir={dest}",
        f"--starts-manifest={starts}",
        "--resume",
    ]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="3-train-seed latent Forward runner")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument(
        "--train-seeds",
        default=",".join(str(s) for s in DEFAULT_TRAIN_SEEDS),
        help="Training seeds (default: 3072,3073,3074)",
    )
    p.add_argument(
        "--eval-seeds",
        default=",".join(str(s) for s in DEFAULT_EVAL_SEEDS),
        help="Frozen eval groups (default: 42-51)",
    )
    p.add_argument("--offsets", default="25,50,75,100")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--starts-root",
        default=str(MULTI_ROOT),
        help="ICLR multiseed root whose starts_manifest.json are reused",
    )
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument(
        "--reuse-s3072",
        action="store_true",
        help="Skip training seed 3072; eval the existing paper checkpoints",
    )
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Print a table from existing summary.txt files and exit",
    )
    return p.parse_args(argv)


def summarize(out_root: Path) -> int:
    rows = []
    for path in sorted(out_root.glob("*/s*/seed_*/summary.txt")):
        task = path.parts[-4]
        train_seed = path.parts[-3].lstrip("s")
        eval_seed = path.parts[-2].split("_")[1]
        rates: dict[int, float] = {}
        for line in path.read_text().splitlines():
            if not line.startswith("mode="):
                continue
            parts = dict(x.split("=", 1) for x in line.split() if "=" in x)
            if parts.get("mode") != "forward":
                continue
            rates[int(parts["offset"])] = float(parts["success_rate"])
        rows.append((task, train_seed, eval_seed, rates))
    if not rows:
        print(f"no summary.txt under {out_root}", file=sys.stderr)
        return 1
    print(
        f"{'task':8} {'train':6} {'eval':5} "
        f"{'25':>6} {'50':>6} {'75':>6} {'100':>6}"
    )
    for task, train_seed, eval_seed, rates in rows:
        print(
            f"{task:8} {train_seed:6} {eval_seed:5} "
            f"{rates.get(25, float('nan')):6.1f} "
            f"{rates.get(50, float('nan')):6.1f} "
            f"{rates.get(75, float('nan')):6.1f} "
            f"{rates.get(100, float('nan')):6.1f}"
        )
    return 0


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.summarize_only:
        return summarize(Path(args.out_root))

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    train_seeds = [int(s.strip()) for s in args.train_seeds.split(",") if s.strip()]
    eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()]
    for task in tasks:
        if task not in TASKS:
            print(f"unknown task {task!r}; expected {list(TASKS)}", file=sys.stderr)
            return 2
        for seed in train_seeds:
            run_name(task, seed)

    starts_root = Path(args.starts_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cache = Path(os.environ.get("STABLEWM_HOME", str(ROOT / ".stable-wm")))

    n_pairs = len(tasks) * len(train_seeds)
    n_eval = n_pairs * len(eval_seeds)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"==== FORWARD TRAIN-SEEDS START {_stamp()} ====")
    log(f"tasks={tasks} train_seeds={train_seeds} eval_seeds={eval_seeds}")
    log(f"reuse_s3072={args.reuse_s3072} epochs={args.epochs}")
    log(f"starts_root={starts_root}")
    log(f"out_root={out_root}")
    log(f"train_pairs={n_pairs} eval_runs={n_eval}")

    python = sys.executable
    pair_i = 0
    trained_any = False
    for task in tasks:
        for train_seed in train_seeds:
            pair_i += 1
            policy = policy_for(task, train_seed, args.reuse_s3072)
            weights = ckpt_path(cache, policy)
            reuse = args.reuse_s3072 and train_seed == 3072
            log("")
            log(
                f"==== TRAIN {pair_i}/{n_pairs} | task={task} "
                f"train_seed={train_seed} | {_stamp()} ===="
            )
            if reuse:
                log(f"SKIP train reuse paper ckpt: {policy}")
            elif weights.is_file():
                log(f"SKIP already has weights: {weights}")
            else:
                cmd = build_train_cmd(
                    python=python,
                    task=task,
                    train_seed=train_seed,
                    epochs=args.epochs,
                    skip_deps=trained_any,
                )
                log("$ " + " ".join(cmd))
                rc = subprocess.call(cmd, cwd=str(ROOT))
                if rc != 0:
                    log(f"FAIL train {task} seed={train_seed} exit={rc}")
                    return rc
                trained_any = True
                if not weights.is_file():
                    log(f"FAIL missing weights after train: {weights}")
                    return 2
            log(f"==== TRAIN DONE {pair_i}/{n_pairs} {task} s{train_seed} ====")

            for e_i, eval_seed in enumerate(eval_seeds, start=1):
                starts = starts_path(starts_root, task, eval_seed)
                dest = eval_dest(out_root, task, train_seed, eval_seed)
                dest.mkdir(parents=True, exist_ok=True)
                log(
                    f"==== EVAL {task} s{train_seed} "
                    f"group {e_i}/{len(eval_seeds)}={eval_seed} | {_stamp()} ===="
                )
                if not starts.is_file():
                    log(f"FAIL missing frozen starts (do not resample): {starts}")
                    return 2
                if (dest / "summary.txt").exists():
                    log(f"SKIP already has summary.txt: {dest}")
                    continue
                cmd = build_eval_cmd(
                    python=python,
                    task=task,
                    policy=policy,
                    cache=str(cache),
                    eval_seed=eval_seed,
                    dest=dest,
                    starts=starts,
                    offsets=args.offsets,
                    num_eval=args.num_eval,
                )
                log("$ " + " ".join(cmd))
                rc = subprocess.call(cmd, cwd=str(ROOT))
                if rc != 0:
                    log(
                        f"FAIL eval {task} train={train_seed} "
                        f"eval={eval_seed} exit={rc}"
                    )
                    return rc
                summary = dest / "summary.txt"
                if summary.exists():
                    log(summary.read_text().rstrip())

    log(f"==== FORWARD TRAIN-SEEDS ALL DONE {_stamp()} ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
