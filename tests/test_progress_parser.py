"""Progress parser: START must not advance; only DONE advances."""

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_fblewm_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_fblewm_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_start_does_not_advance_completed():
    mod = _load_pipeline_module()
    state = mod.ProgressState(current=0, total=12, brief="")
    mod.parse_progress(
        "eval 1/12 START | mode=official offset=25 budget=50 (completed 0/12)",
        state,
        "eval",
    )
    assert state.current == 0
    assert state.total == 12
    assert "running" in state.brief


def test_done_advances_completed():
    mod = _load_pipeline_module()
    state = mod.ProgressState(current=0, total=12, brief="")
    mod.parse_progress(
        "eval 1/12 DONE | mode=official offset=25 success_rate=90.0% (45/50) time=1.0s",
        state,
        "eval",
    )
    assert state.current == 1
    assert state.total == 12
    mod.parse_progress(
        "eval 12/12 DONE | mode=backward offset=100 success_rate=10.0% (5/50) time=1.0s",
        state,
        "eval",
    )
    assert state.current == 12


def test_not_100_before_unit_12():
    mod = _load_pipeline_module()
    state = mod.ProgressState(current=0, total=12, brief="")
    for i in range(1, 12):
        mod.parse_progress(
            f"eval {i}/12 DONE | mode=official offset=25 success_rate=1.0% (1/50) time=0.1s",
            state,
            "eval",
        )
    assert state.current == 11
    assert state.current < state.total


def test_error_summary_extracts_last_exception(tmp_path):
    mod = _load_pipeline_module()
    log = tmp_path / "fail.log"
    log.write_text(
        "Traceback (most recent call last):\n"
        "  File x.py, line 1\n"
        "RuntimeError: boom failed\n"
        "[exit 1]\n"
    )
    msg = mod._extract_error_summary(log)
    assert "RuntimeError" in msg or "boom" in msg


def test_backward_target_now_overrides_imaginer():
    """`now` is a third --backward-target; encoder/pred keep unary Causal B."""
    mod = _load_pipeline_module()
    assert mod.task_spec("pusht").default_backward_target == "pred"
    assert mod.task_spec("tworoom").default_backward_target == "now"
    assert mod.task_spec("cube").default_backward_target == "encoder"
    assert mod.task_spec("reacher").default_backward_target == "pred"
    assert mod.task_spec("reacher").train_data == "dmc"
    assert mod.task_spec("reacher").eval_config == "reacher"
    assert mod.task_spec("reacher").default_run_name == "fblewm_reacher_v1"
    assert "reacher.tar.zst" in mod.task_spec("reacher").zst_names

    fake = argparse.Namespace()
    fake.root = Path("/tmp")
    fake.run_id = "x"
    fake.args = argparse.Namespace(
        task="tworoom",
        epochs=10,
        batch_size=128,
        train_run_name="fblewm_tworoom_v2",
        backward_target="now",
    )
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "loss.backward.target=now" in cmd
    assert any("ConditionalLatentImaginer" in c for c in cmd)

    fake.args.backward_target = "encoder"
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "loss.backward.target=encoder" in cmd
    assert not any("ConditionalLatentImaginer" in c for c in cmd)

    fake.args.backward_target = "pred"
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "loss.backward.target=pred" in cmd
    assert not any("ConditionalLatentImaginer" in c for c in cmd)

    fake.args.backward_target = "pred_goal"
    fake.args.train_run_name = "fblewm_tworoom_v3"
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "loss.backward.target=pred_goal" in cmd
    assert any("ConditionalLatentImaginer" in c for c in cmd)
    assert "model.backward_anchor=pred" in cmd


def test_backward_target_fixed_bridge_overrides_imaginer():
    mod = _load_pipeline_module()
    parser = mod.build_parser()
    parsed = parser.parse_args(
        ["--task", "tworoom", "--backward-target", "fixed_bridge"]
    )
    assert parsed.backward_target == "fixed_bridge"
    assert mod.task_spec("tworoom").default_backward_target == "now"

    fake = argparse.Namespace()
    fake.root = Path("/tmp")
    fake.run_id = "x"
    fake.args = argparse.Namespace(
        task="tworoom",
        epochs=10,
        batch_size=128,
        train_run_name="fblewm_tworoom_v4",
        backward_target="fixed_bridge",
    )
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "loss.backward.target=fixed_bridge" in cmd
    assert any("ConditionalLatentImaginer" in c for c in cmd)
    assert "model.backward_anchor=pred" in cmd
    assert "loss.backward.p_noise=0.0" in cmd
    assert "loss.backward.goal_rank_weight=0.0" in cmd


def test_eval_cmd_forwards_diag_flags_and_eval_dir():
    mod = _load_pipeline_module()
    fake = argparse.Namespace()
    fake.root = Path("/home/yuanben/WorldModel/FBLeWM")
    fake.run_id = "x"
    fake.env = {
        "STABLEWM_HOME": "/tmp/swm",
        "LOCAL_DATASET_DIR": "/tmp/data",
    }
    fake.args = argparse.Namespace(
        task="pusht",
        policy="fblewm_pusht_v3/weights_epoch_10.pt",
        eval_modes="backward",
        eval_offsets="50,75,100",
        eval_dir="/tmp/diag/eval_cap1",
        eval_resume_dir=None,
        eval_horizon=None,
        eval_receding_horizon=None,
        starts_manifest="/tmp/starts.json",
        backward_depth_cap=2,
        record_cem_cost=True,
    )
    cmd = mod.Pipeline._cmd_eval(fake, None)
    assert "--backward-depth-cap=2" in cmd
    assert "--record-cem-cost" in cmd
    assert "--hydra-run-dir=/tmp/diag/eval_cap1" in cmd
    assert "--starts-manifest=/tmp/starts.json" in cmd
    assert "--modes=backward" in cmd
    assert "--offsets=50,75,100" in cmd


def test_forward_variant_cli_and_train_overrides():
    mod = _load_pipeline_module()
    parser = mod.build_parser()
    parsed = parser.parse_args(
        [
            "--forward-variant",
            "action_aligned",
            "--forward-action-weight",
            "0.5",
        ]
    )
    assert parsed.forward_variant == "action_aligned"
    assert parsed.forward_action_weight == 0.5

    fake = argparse.Namespace()
    fake.root = Path("/tmp")
    fake.run_id = "x"
    fake.args = argparse.Namespace(
        task="pusht",
        epochs=10,
        batch_size=128,
        train_run_name="fblewm_pusht_aaf_v1",
        backward_target="pred",
        forward_variant="latent",
        forward_action_weight=1.0,
    )
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "loss.forward.variant=latent" in cmd
    assert "loss.forward.action_weight=1.0" in cmd
    assert not any("ActionAlignedCausalLatentImaginer" in c for c in cmd)
    assert "loss.backward.weight=0.0" not in cmd

    fake.args.forward_variant = "action_aligned"
    fake.args.forward_action_weight = 0.5
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "loss.forward.variant=action_aligned" in cmd
    assert "loss.forward.action_weight=0.5" in cmd
    assert "loss.backward.weight=0.0" in cmd
    assert not any("ActionAlignedCausalLatentImaginer" in c for c in cmd)


def test_progress_parser_reads_forward_action_loss():
    mod = _load_pipeline_module()
    state = mod.ProgressState(current=0, total=1, brief="")
    mod.parse_progress("train/forward_action_loss=0.1234", state, "train")
    assert "forward_action_loss=0.1234" in state.brief


def test_sequential_action_cli_and_train_overrides():
    mod = _load_pipeline_module()
    parser = mod.build_parser()
    parsed = parser.parse_args(
        [
            "--forward-variant",
            "sequential_action",
            "--forward-action-weight",
            "0.25",
            "--forward-teacher-weight",
            "0.5",
        ]
    )
    assert parsed.forward_variant == "sequential_action"
    assert parsed.forward_action_weight == 0.25
    assert parsed.forward_teacher_weight == 0.5

    fake = argparse.Namespace()
    fake.root = Path("/tmp")
    fake.run_id = "x"
    fake.args = argparse.Namespace(
        task="pusht",
        epochs=10,
        batch_size=128,
        train_run_name="fblewm_pusht_saaf_v1",
        backward_target="pred",
        forward_variant="sequential_action",
        forward_action_weight=1.0,
        forward_teacher_weight=1.0,
    )
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "loss.forward.variant=sequential_action" in cmd
    assert "loss.forward.action_weight=1.0" in cmd
    assert "loss.forward.teacher_weight=1.0" in cmd
    assert "loss.backward.weight=0.0" in cmd
    assert not any("SequentialActionCausalLatentImaginer" in c for c in cmd)


def test_progress_parser_reads_sequential_forward_losses():
    mod = _load_pipeline_module()
    state = mod.ProgressState(current=0, total=1, brief="")
    mod.parse_progress(
        "val/forward_teacher_loss=0.11 val/forward_auto_step_loss=0.22 "
        "val/forward_roll_loss=0.33 val/forward_action_loss=0.44",
        state,
        "train",
    )
    assert "forward_teacher_loss=0.11" in state.brief
    assert "forward_auto_step_loss=0.22" in state.brief
    assert "forward_roll_loss=0.33" in state.brief
    assert "forward_action_loss=0.44" in state.brief


def test_branch_preserving_cli_and_train_overrides():
    mod = _load_pipeline_module()
    parser = mod.build_parser()
    parsed = parser.parse_args(
        [
            "--forward-variant",
            "branch_preserving",
            "--forward-branches",
            "4",
        ]
    )
    assert parsed.forward_variant == "branch_preserving"
    assert parsed.forward_branches == 4

    fake = argparse.Namespace()
    fake.root = Path("/tmp")
    fake.run_id = "x"
    fake.env = {"STABLEWM_HOME": "/tmp/swm", "LOCAL_DATASET_DIR": "/tmp/data"}
    fake.args = argparse.Namespace(
        task="cube",
        epochs=10,
        batch_size=128,
        train_run_name="fblewm_cube_bphas_v1",
        backward_target="encoder",
        forward_variant="branch_preserving",
        forward_action_weight=1.0,
        forward_teacher_weight=1.0,
        forward_branches=4,
        eval_modes=None,
        eval_offsets=None,
        policy=None,
        eval_dir=None,
        eval_resume_dir=None,
        eval_horizon=None,
        eval_receding_horizon=None,
        starts_manifest=None,
        backward_depth_cap=None,
        record_cem_cost=False,
    )
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "loss.forward.variant=branch_preserving" in cmd
    assert "loss.forward.branches=4" in cmd
    assert "loss.backward.weight=0.0" not in cmd
    assert not any("BranchPreservingCausalLatentImaginer" in c for c in cmd)
    assert mod.Pipeline._resolved_eval_modes(fake) == "official,forward"


def test_progress_parser_reads_branch_forward_losses():
    mod = _load_pipeline_module()
    state = mod.ProgressState(current=0, total=1, brief="")
    mod.parse_progress(
        "val/forward_step_loss=0.11 val/forward_roll_loss=0.22 val/forward_loss=0.33",
        state,
        "train",
    )
    assert "forward_step_loss=0.11" in state.brief
    assert "forward_roll_loss=0.22" in state.brief
    assert "forward_loss=0.33" in state.brief


def test_reacher_cli_choice_and_train_overrides():
    mod = _load_pipeline_module()
    parser = mod.build_parser()
    parsed = parser.parse_args(["--task", "reacher"])
    assert parsed.task == "reacher"
    assert "reacher" in parser._option_string_actions["--task"].choices

    fake = argparse.Namespace()
    fake.root = Path("/tmp")
    fake.run_id = "x"
    fake.args = argparse.Namespace(
        task="reacher",
        epochs=10,
        batch_size=128,
        train_run_name="fblewm_reacher_v1",
        backward_target="pred",
        forward_variant="latent",
        forward_action_weight=1.0,
        forward_teacher_weight=1.0,
    )
    cmd = mod.Pipeline._cmd_train(fake, None)
    assert "data=dmc" in cmd
    assert "output_model_name=fblewm_reacher_v1" in cmd
    assert "loss.backward.target=pred" in cmd
    assert "loss.backward.weight=0.0" not in cmd
