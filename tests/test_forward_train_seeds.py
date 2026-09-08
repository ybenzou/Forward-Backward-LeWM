"""Contracts for 3-train-seed latent Forward launcher."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load(name: str, rel: str):
    path = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tasks_exclude_cube_and_run_names_are_isolated():
    mod = _load("run_forward_train_seeds", "scripts/run_forward_train_seeds.py")
    assert mod.TASKS == ("pusht", "tworoom", "reacher")
    assert "cube" not in mod.TASKS
    for task in mod.TASKS:
        for seed in mod.DEFAULT_TRAIN_SEEDS:
            name = mod.run_name(task, seed)
            assert name not in mod.PROTECTED_RUN_NAMES
            assert name == f"fblewm_{task}_s{seed}"
            assert mod.policy_for(task, seed, reuse_s3072=False) == (
                f"{name}/weights_epoch_10.pt"
            )
    assert mod.policy_for("pusht", 3072, reuse_s3072=True) == (
        "fblewm/weights_epoch_10.pt"
    )
    assert mod.policy_for("tworoom", 3072, reuse_s3072=True) == (
        "fblewm_tworoom/weights_epoch_10.pt"
    )
    assert mod.policy_for("reacher", 3072, reuse_s3072=True) == (
        "fblewm_reacher_v1/weights_epoch_10.pt"
    )
    assert mod.policy_for("pusht", 3073, reuse_s3072=True).startswith(
        "fblewm_pusht_s3073"
    )


def test_train_cmd_is_latent_and_passes_seed():
    mod = _load("run_forward_train_seeds", "scripts/run_forward_train_seeds.py")
    cmd = mod.build_train_cmd(
        python="python",
        task="tworoom",
        train_seed=3073,
        epochs=10,
        skip_deps=True,
    )
    assert "--task=tworoom" in cmd
    assert "--seed=3073" in cmd
    assert "--train-run-name=fblewm_tworoom_s3073" in cmd
    assert "--forward-variant=latent" in cmd
    assert "--skip-eval" in cmd
    assert "--skip-deps" in cmd
    first = mod.build_train_cmd(
        python="python",
        task="pusht",
        train_seed=3072,
        epochs=10,
        skip_deps=False,
    )
    assert "--skip-deps" not in first


def test_eval_cmd_forward_only_reuses_frozen_starts():
    mod = _load("run_forward_train_seeds", "scripts/run_forward_train_seeds.py")
    dest = Path("/tmp/train_seeds/pusht/s3073/seed_42")
    starts = Path("/tmp/multiseed/pusht/seed_42/starts_manifest.json")
    cmd = mod.build_eval_cmd(
        python="python",
        task="pusht",
        policy="fblewm_pusht_s3073/weights_epoch_10.pt",
        cache="/tmp/swm",
        eval_seed=42,
        dest=dest,
        starts=starts,
        offsets="25,50,75,100",
        num_eval=50,
    )
    assert "--modes=forward" in cmd
    assert "--config-name=pusht" in cmd
    assert "--seed=42" in cmd
    assert "--starts-manifest=/tmp/multiseed/pusht/seed_42/starts_manifest.json" in cmd
    assert "--policy=fblewm_pusht_s3073/weights_epoch_10.pt" in cmd
    assert "--eval-dir=/tmp/train_seeds/pusht/s3073/seed_42" in cmd
    assert "--official" not in " ".join(cmd)


def test_pipeline_train_cmd_forwards_seed():
    pipe = _load("run_fblewm_pipeline_seed", "scripts/run_fblewm_pipeline.py")
    fake = argparse.Namespace()
    fake.root = Path("/tmp")
    fake.run_id = "x"
    fake.args = argparse.Namespace(
        task="pusht",
        epochs=10,
        batch_size=128,
        seed=3074,
        train_run_name="fblewm_pusht_s3074",
        backward_target="pred",
        forward_variant="latent",
        forward_action_weight=1.0,
        forward_teacher_weight=1.0,
    )
    cmd = pipe.Pipeline._cmd_train(fake, None)
    assert "seed=3074" in cmd
    assert "output_model_name=fblewm_pusht_s3074" in cmd
    assert "loss.forward.variant=latent" in cmd
    parsed = pipe.build_parser().parse_args([])
    assert parsed.seed == 3072
