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
