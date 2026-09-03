#!/usr/bin/env python3
"""One-shot runner for the HAS interpretability plan.

Runs, in order:
  1) paired o=75 process recordings + contact sheets + process figure
  2) held-out latent PCA / t-SNE / retrieval
  3) seed-42 fixed-k Forward evals and the LaTeX table

Resume-safe: re-running skips finished eval units.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PY = sys.executable

TASKS = (
    ("pusht", "fblewm/weights_epoch_10.pt"),
    ("tworoom", "fblewm_tworoom/weights_epoch_10.pt"),
    ("reacher", "fblewm_reacher_v1/weights_epoch_10.pt"),
)
K_CONDS = (
    ("k0", "0"),
    ("k5", "5"),
    ("k15", "15"),
    ("dynamic", None),
)


def run(argv: list[str], env: dict[str, str]) -> None:
    print("\n==== RUN ====\n" + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=str(ROOT), env=env, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-process", action="store_true")
    p.add_argument("--skip-latents", action="store_true")
    p.add_argument("--skip-k-ablation", action="store_true")
    args = p.parse_args()

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not args.skip_process:
        run(
            [
                PY,
                str(SCRIPTS / "plot_has_process.py"),
                "--tasks",
                "pusht,tworoom,reacher",
                "--contact-sheet",
            ],
            env,
        )

    if not args.skip_latents:
        run([PY, str(SCRIPTS / "analyze_has_latents.py")], env)

    if not args.skip_k_ablation:
        man_root = ROOT / "outputs" / "diag" / "iclr_bar" / "multiseed"
        for task, policy in TASKS:
            manifest = man_root / task / "seed_42" / "starts_manifest.json"
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            for name, k in K_CONDS:
                cmd = [
                    PY,
                    str(SCRIPTS / "eval_fblewm_matrix.py"),
                    "--policy",
                    policy,
                    "--config-name",
                    task,
                    "--modes",
                    "forward",
                    "--offsets",
                    "75,100",
                    "--seed",
                    "42",
                    "--starts-manifest",
                    str(manifest),
                    "--eval-dir",
                    str(ROOT / "outputs" / "diag" / "k_ablation" / task / name),
                    "--resume",
                ]
                if k is not None:
                    cmd.extend(["--forward-depth-override", k])
                run(cmd, env)
        run([PY, str(SCRIPTS / "summarize_k_ablation.py")], env)

    print("\n==== ALL STAGES FINISHED ====", flush=True)
    print("process: outputs/diag/process_compare/v2/", flush=True)
    print("latents: outputs/diag/has_latents/v1/", flush=True)
    print("k-ablation table: paper/tables/tab_k_depth.tex", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
