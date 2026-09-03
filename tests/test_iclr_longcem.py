"""Contracts for 10-seed long-CEM runner and figure loaders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load(name: str, rel: str):
    path = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_multiseed_runner_includes_reacher():
    mod = _load("run_iclr_multiseed", "scripts/run_iclr_multiseed.py")
    assert "reacher" in mod.TASKS
    assert mod.TASKS["reacher"]["policy"] == "fblewm_reacher_v1/weights_epoch_10.pt"
    assert mod.TASKS["reacher"]["config"] == "reacher"


def test_longcem_runner_includes_reacher():
    mod = _load("run_iclr_longcem", "scripts/run_iclr_longcem.py")
    assert "reacher" in mod.TASKS
    assert mod.TASKS["reacher"]["policy"] == "fblewm_reacher_v1/weights_epoch_10.pt"
    assert mod.TASKS["reacher"]["config"] == "reacher"


def test_horizon_skips_offsets_with_short_budget():
    mod = _load("run_iclr_longcem", "scripts/run_iclr_longcem.py")
    assert mod.offsets_for_horizon(10) == (25, 50, 75, 100)
    assert mod.offsets_for_horizon(15) == (50, 75, 100)
    assert 25 not in mod.offsets_for_horizon(15)


def test_longcem_cmd_reuses_starts_and_sets_horizon():
    mod = _load("run_iclr_longcem", "scripts/run_iclr_longcem.py")
    cmd = mod.build_eval_cmd(
        python="python",
        matrix=Path("/tmp/eval_fblewm_matrix.py"),
        policy="fblewm/weights_epoch_10.pt",
        config="pusht",
        cache="/tmp/swm",
        seed=42,
        dest=Path("/tmp/out"),
        starts=Path("/tmp/starts.json"),
        horizon=10,
        offsets=(25, 50, 75, 100),
        num_eval=50,
    )
    assert "--modes=official" in cmd
    assert "--horizon=10" in cmd
    assert "--receding-horizon=10" in cmd
    assert "--starts-manifest=/tmp/starts.json" in cmd
    assert "--offsets=25,50,75,100" in cmd
    assert "--seed=42" in cmd


def test_plot_loaders_accept_h5_and_h10_layout(tmp_path):
    plot = _load("plot_iclr_multiseed", "scripts/plot_iclr_multiseed.py")
    seeds = plot.SEEDS
    multi = tmp_path / "multiseed"
    long = tmp_path / "longcem"
    for task in plot.TASKS:
        for seed in seeds:
            h5 = multi / task / f"seed_{seed}"
            h5.mkdir(parents=True)
            lines = ["==== FBLeWM MATRIX SUMMARY ====\n"]
            for mode in ("official", "forward"):
                for off in plot.OFFSETS:
                    lines.append(
                        f"mode={mode} offset={off} budget=50 success_rate={10.0 + seed} "
                        f"n_success=1/50 seconds=1.0\n"
                    )
            (h5 / "summary.txt").write_text("".join(lines))

            if task not in (*plot.LONGCEM_TASKS, *plot.OPTIONAL_LONGCEM_TASKS):
                continue
            if task in plot.OPTIONAL_LONGCEM_TASKS:
                continue
            h10 = long / "h10" / task / f"seed_{seed}"
            h10.mkdir(parents=True)
            lines = ["==== FBLeWM MATRIX SUMMARY ====\nhorizon: 10\n"]
            for off in plot.OFFSETS:
                lines.append(
                    f"mode=official offset={off} budget=50 success_rate={20.0 + seed} "
                    f"n_success=1/50 seconds=1.0\n"
                )
            (h10 / "summary.txt").write_text("".join(lines))

    data = plot.load_matrix(multi)
    cem = plot.load_longcem_official(long, 10)
    assert data["pusht"]["official"][25].shape == (10,)
    assert data["reacher"]["forward"][100].shape == (10,)
    assert cem["tworoom"][100].shape == (10,)
    assert "reacher" not in cem
    assert np.allclose(cem["pusht"][25], np.array([20.0 + s for s in seeds]))
    rec = plot.aggregate(data, longcem=cem)
    assert "official_h10" in rec["tasks"]["pusht"]
    assert "delta_vs_h10" in rec["tasks"]["pusht"]
    assert "official_h10" not in rec["tasks"]["reacher"]

    for seed in seeds:
        h10 = long / "h10" / "reacher" / f"seed_{seed}"
        h10.mkdir(parents=True)
        lines = ["==== FBLeWM MATRIX SUMMARY ====\nhorizon: 10\n"]
        for off in plot.OFFSETS:
            lines.append(
                f"mode=official offset={off} budget=50 success_rate={20.0 + seed} "
                f"n_success=1/50 seconds=1.0\n"
            )
        (h10 / "summary.txt").write_text("".join(lines))
    cem2 = plot.load_longcem_official(long, 10)
    assert cem2["reacher"][100].shape == (10,)
    rec2 = plot.aggregate(data, longcem=cem2)
    assert "official_h10" in rec2["tasks"]["reacher"]
    assert "delta_vs_h10" in rec2["tasks"]["reacher"]
