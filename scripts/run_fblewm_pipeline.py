#!/usr/bin/env python3
"""Unified rich + popen + tqdm pipeline for FBLeWM (PushT / Cube / TwoRoom).

Terminal: metadata table once, stage board, live X/Total refresh only.
Full subprocess output -> logs/runs/<run_id>/<nn>_<stage>.log

Task extension is driven by TASK_SPECS; default --task=pusht keeps legacy behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FBLEWM_ROOT = Path(
    os.environ.get("FBLEWM_ROOT", "/home/yuanben/WorldModel/FBLeWM")
).resolve()
DEFAULT_DATA_DIR = Path("/home/yuanben/WorldModel/LeWM/data")


@dataclass(frozen=True)
class TaskSpec:
    """Per-task wiring for data / train / eval (PushT-compatible defaults)."""

    task: str
    train_data: str
    eval_config: str
    default_run_name: str
    default_backward_target: str
    # Relative paths under LOCAL_DATASET_DIR or data/extracted that count as "ready".
    ready_names: Tuple[str, ...]
    # Relative symlink targets under STABLEWM_HOME/datasets/.
    dataset_links: Tuple[str, ...]
    # Upload filenames (primary: LOCAL_DATASET_DIR / LeWM/data; fallback: FBLeWM/data/incoming).
    zst_names: Tuple[str, ...]
    hf_urls: Tuple[str, ...]
    # Expected h5 basename after extract (used to locate file under extract root).
    h5_basenames: Tuple[str, ...]


TASK_SPECS: Dict[str, TaskSpec] = {
    "pusht": TaskSpec(
        task="pusht",
        train_data="pusht",
        eval_config="pusht",
        default_run_name="fblewm_bp",
        default_backward_target="pred",
        ready_names=("pusht_expert_train.h5", "pusht_expert_train.lance"),
        dataset_links=("pusht_expert_train.h5", "pusht_expert_train.lance"),
        zst_names=(),
        hf_urls=(),
        h5_basenames=("pusht_expert_train.h5",),
    ),
    "tworoom": TaskSpec(
        task="tworoom",
        train_data="tworoom",
        eval_config="tworoom",
        default_run_name="fblewm_tworoom_v2",
        default_backward_target="now",
        ready_names=("tworoom.h5",),
        dataset_links=("tworoom.h5",),
        zst_names=("tworoom.tar.zst",),
        hf_urls=(
            "https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst",
        ),
        h5_basenames=("tworoom.h5",),
    ),
    "cube": TaskSpec(
        task="cube",
        train_data="ogb",
        eval_config="cube",
        default_run_name="fblewm_cube_v2",
        default_backward_target="encoder",
        ready_names=(
            "ogbench/cube_single_expert.h5",
            "cube_single_expert.h5",
        ),
        dataset_links=("ogbench/cube_single_expert.h5",),
        zst_names=("cube_single_expert.tar.zst",),
        hf_urls=(
            "https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst",
        ),
        h5_basenames=("cube_single_expert.h5",),
    ),
    "reacher": TaskSpec(
        task="reacher",
        train_data="dmc",
        eval_config="reacher",
        default_run_name="fblewm_reacher_v1",
        default_backward_target="pred",
        ready_names=("reacher.h5", "dmc/reacher_random.h5"),
        dataset_links=("reacher.h5", "dmc/reacher_random.h5"),
        zst_names=("reacher.tar.zst",),
        hf_urls=(
            "https://huggingface.co/datasets/quentinll/lewm-reacher/resolve/main/reacher.tar.zst",
        ),
        h5_basenames=("reacher.h5",),
    ),
}


def task_spec(task: str) -> TaskSpec:
    if task not in TASK_SPECS:
        raise KeyError(f"unknown task {task!r}; expected one of {tuple(TASK_SPECS)}")
    return TASK_SPECS[task]


def apply_path_env(root: Path) -> dict:
    """Return env dict with artifacts locked under FBLEWM_ROOT; data reused from LeWM."""
    env = os.environ.copy()
    stable = root / ".stable-wm"
    hf = root / ".cache" / "huggingface"
    data = Path(env.get("LOCAL_DATASET_DIR", str(DEFAULT_DATA_DIR)))
    env["FBLEWM_ROOT"] = str(root)
    env["STABLEWM_HOME"] = str(stable)
    env["HF_HOME"] = str(hf)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf / "hub")
    env["LOCAL_DATASET_DIR"] = str(data)
    env["PYTHONUNBUFFERED"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    # Ensure FBLeWM repo is importable for hydra _target_: fblewm.FBLeWM
    env["PYTHONPATH"] = str(root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    for p in (
        root / "models",
        root / "logs" / "runs",
        root / "outputs" / "hydra",
        root / "outputs" / "eval",
        root / "outputs" / "diag",
        root / "outputs" / "checkpoints",
        root / "data" / "incoming",
        root / "data" / "extracted",
        stable / "checkpoints",
        stable / "datasets",
        hf / "hub",
    ):
        p.mkdir(parents=True, exist_ok=True)
    # Soft-link PushT dataset into STABLEWM_HOME/datasets for eval tooling.
    for name in ("pusht_expert_train.h5", "pusht_expert_train.lance"):
        src = data / name
        link = stable / "datasets" / name
        if src.exists() and not link.exists():
            try:
                link.symlink_to(src.resolve())
            except OSError:
                pass
    return env


def _symlink_force(link: Path, target: Path, logf) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            logf.write(f"link ok {link} -> {target}\n")
            return
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            logf.write(f"ERROR: refusing to replace non-file {link}\n")
            raise RuntimeError(f"cannot replace {link}")
    link.symlink_to(target.resolve())
    logf.write(f"symlink {link} -> {target}\n")


def _find_existing_ready(spec: TaskSpec, search_roots: Sequence[Path]) -> Optional[Path]:
    """Return absolute path to a ready dataset file if present."""
    for root in search_roots:
        for rel in spec.ready_names:
            p = root / rel
            if p.exists():
                return p.resolve()
    return None


def _find_zst(spec: TaskSpec, search_roots: Sequence[Path]) -> Optional[Path]:
    for root in search_roots:
        for name in spec.zst_names:
            p = root / name
            if p.exists() and p.stat().st_size > 0:
                return p.resolve()
    return None


def _find_h5_under(root: Path, basenames: Sequence[str]) -> Optional[Path]:
    for base in basenames:
        direct = root / base
        if direct.exists():
            return direct.resolve()
        hits = list(root.rglob(base))
        if hits:
            return hits[0].resolve()
    return None


# ---------------------------------------------------------------------------
# Progress parsing
# ---------------------------------------------------------------------------

RE_EPOCH = re.compile(r"Epoch\s*(?:\[?\s*)?(\d+)\s*/\s*(\d+)", re.I)
# Matches Lightning/PrintProgressBar: "step 50/1234" and "it: 50/1234"
RE_STEP = re.compile(r"(?:step|it|batch)\s*[|:]?\s*(\d+)\s*/\s*(\d+)", re.I)
RE_PCT = re.compile(r"(\d{1,3})%")
RE_LOSS = re.compile(
    r"(pred_loss|sigreg_loss|forward_teacher_loss|forward_auto_step_loss|forward_step_loss|forward_roll_loss|forward_loss|forward_action_loss|backward_loss|official_loss|train/loss|val/loss|loss)[=:\s]\s*([0-9.eE+-]+)",
    re.I,
)
RE_FAIL = re.compile(
    r"(Error|Traceback|CUDA out of memory|Driver/library version mismatch|FAILED)",
    re.I,
)
RE_EVAL_DONE = re.compile(
    r"eval\s+(\d+)\s*/\s*(\d+)\s+DONE\s*\|\s*mode=([\w]+)\s+offset=(\d+)\s+success_rate=([0-9.]+)%",
    re.I,
)
RE_EVAL_START = re.compile(
    r"eval\s+(\d+)\s*/\s*(\d+)\s+START\s*\|\s*mode=([\w]+)\s+offset=(\d+)",
    re.I,
)


@dataclass
class ProgressState:
    current: int = 0
    total: int = 1
    brief: str = ""
    unit: str = "it"  # "it" | "B" | "epoch" | "%"
    failed_hint: bool = False
    started_at: float = 0.0
    # Optional nested progress (e.g. train: epoch outer, step inner)
    sub_current: int = 0
    sub_total: int = 0
    sub_label: str = "step"
    error_msg: str = ""


def _extract_error_summary(log: Path, max_chars: int = 280) -> str:
    """Pull a short actionable error from a stage log."""
    if not log.exists():
        return "ERROR: stage failed (no log)"
    text = log.read_text(errors="replace")
    # Prefer last Exception-like line
    prefer = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("FutureWarning") or "deprecated" in s.lower():
            continue
        if (
            "Error" in s
            or s.startswith("Traceback")
            or "Exception" in s
            or s.startswith("[exit")
        ):
            prefer.append(s)
    if prefer:
        # last non-exit error line if possible
        err_lines = [x for x in prefer if not x.startswith("[exit")]
        msg = err_lines[-1] if err_lines else prefer[-1]
    else:
        tail = [ln.strip() for ln in text.splitlines() if ln.strip()]
        msg = tail[-1] if tail else "unknown error"
    if len(msg) > max_chars:
        msg = msg[: max_chars - 3] + "..."
    return f"ERROR: {msg}"


def _format_amount(n: int, unit: str) -> str:
    if unit == "B":
        if n >= 1024**3:
            return f"{n / 1024**3:.2f}GB"
        if n >= 1024**2:
            return f"{n / 1024**2:.1f}MB"
        if n >= 1024:
            return f"{n / 1024:.1f}KB"
        return f"{n}B"
    return str(n)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _eta_seconds(state: ProgressState) -> Optional[float]:
    """Estimate remaining time from outer progress (and sub-progress if useful)."""
    if state.started_at <= 0:
        return None
    elapsed = time.time() - state.started_at
    if elapsed < 0.5:
        return None
    cur, tot = int(state.current), max(int(state.total), 1)
    # Prefer fine-grained step rate during an epoch when available
    if state.sub_total > 0 and state.sub_current > 0 and state.unit == "epoch":
        # completed = finished epochs + fraction of current epoch
        # Lightning epochs are often 0-indexed in logs; treat current as completed count carefully
        done_units = max(cur - 1, 0) * state.sub_total + state.sub_current
        total_units = tot * state.sub_total
        if done_units <= 0:
            return None
        rate = done_units / elapsed
        remain_units = max(total_units - done_units, 0)
        return remain_units / rate if rate > 0 else None
    if cur <= 0:
        return None
    rate = cur / elapsed
    return max(tot - cur, 0) / rate if rate > 0 else None


def _text_bar(current: int, total: int, width: int = 36) -> str:
    total = max(int(total), 1)
    current = max(0, min(int(current), total))
    filled = int(width * current / total)
    return "█" * filled + "░" * (width - filled)


def parse_progress(line: str, state: ProgressState, stage_id: str) -> None:
    # Do not rewrite brief on every Traceback line (causes useless "see log").
    # Final ERROR summary is extracted from the log when the stage exits non-zero.
    if RE_FAIL.search(line) and "FutureWarning" not in line and "deprecated" not in line:
        state.failed_hint = True
    m = RE_EPOCH.search(line)
    if m:
        # Lightning may print 0-based or 1-based; keep raw and clamp display later
        state.unit = "epoch"
        state.current = int(m.group(1))
        state.total = max(int(m.group(2)), 1)
    m_eval = RE_EVAL_DONE.search(line) if stage_id == "eval" else None
    if m_eval:
        cur, tot = int(m_eval.group(1)), max(int(m_eval.group(2)), 1)
        state.current, state.total = cur, tot
        state.brief = (
            f"{m_eval.group(3)}@{m_eval.group(4)} sr={m_eval.group(5)}% ({cur}/{tot})"
        )
    m_run = RE_EVAL_START.search(line) if stage_id == "eval" else None
    if m_run and not m_eval:
        # Do not advance completed count; only show which unit is in flight.
        tot = max(int(m_run.group(2)), 1)
        state.total = tot
        # completed remains previous DONE value
        state.brief = (
            f"running {m_run.group(3)}@{m_run.group(4)} "
            f"({m_run.group(1)}/{tot})"
        )
    m2 = RE_STEP.search(line)
    if m2 and stage_id in ("train", "eval", "data") and not m_eval and not m_run:
        cur, tot = int(m2.group(1)), max(int(m2.group(2)), 1)
        if stage_id == "train" and state.unit == "epoch":
            state.sub_current, state.sub_total = cur, tot
            state.sub_label = "step"
        else:
            if state.total <= 1 or stage_id != "train":
                state.current, state.total = cur, tot
            state.brief = f"step {cur}/{tot}"
    else:
        m3 = RE_PCT.search(line)
        if m3 and stage_id == "data":
            state.unit = "%"
            state.current = int(m3.group(1))
            state.total = 100
    losses = RE_LOSS.findall(line)
    if losses:
        loss_s = " ".join(f"{k}={v}" for k, v in losses)[:160]
        if state.sub_total > 0:
            state.brief = f"{state.sub_label} {state.sub_current}/{state.sub_total}  {loss_s}"
        else:
            state.brief = loss_s
    elif line.strip() and not state.brief:
        state.brief = line.strip()[:100]


# ---------------------------------------------------------------------------
# Stage model
# ---------------------------------------------------------------------------

Status = str  # PENDING | RUNNING | DONE | FAILED | SKIPPED


@dataclass
class Stage:
    idx: int
    stage_id: str
    title: str
    status: Status = "PENDING"
    progress: ProgressState = field(default_factory=ProgressState)
    log_path: Optional[Path] = None
    started: float = 0.0
    ended: float = 0.0
    returncode: Optional[int] = None
    build_cmd: Optional[Callable[["Pipeline"], List[str]]] = None
    run_fn: Optional[Callable[["Pipeline", Path], int]] = None  # in-process


STAGES_SPEC = [
    (0, "setup", "Create FBLeWM dirs / path lock"),
    (1, "deps", "Verify/install Python deps"),
    (2, "config", "Resolve train/eval configs"),
    (3, "data_link", "Prepare/link task dataset into STABLEWM_HOME"),
    (4, "gpu_gate", "nvidia-smi + torch.cuda"),
    (5, "model_smoke", "Import FBLeWM stack probe"),
    (6, "train", "Train FBLeWM (official+F+B)"),
    (7, "eval", "Matrix eval (modes × offsets)"),
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = FBLEWM_ROOT
        self.env = apply_path_env(self.root)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.task}"
        self.log_dir = self.root / "logs" / "runs" / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stages: List[Stage] = [
            Stage(idx=i, stage_id=sid, title=title) for i, sid, title in STAGES_SPEC
        ]
        self._bind_stage_runners()
        self.meta = self._build_meta()
        self._events: List[dict] = []
        self._live = None
        self._console = None
        self._rich_ok = False
        self._pending_error: Optional[tuple] = None
        # Continuous UI clock: elapsed/ETA must tick even when subprocess is silent.
        self._ui_lock = threading.Lock()
        self._ui_active: Optional[Stage] = None
        self._ui_tick_stop: Optional[threading.Event] = None
        self._ui_tick_thread: Optional[threading.Thread] = None
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.table import Table
            from rich.panel import Panel
            from rich.layout import Layout
            from rich.text import Text
            from rich.progress import (
                Progress,
                BarColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            self._Console = Console
            self._Live = Live
            self._Table = Table
            self._Panel = Panel
            self._Layout = Layout
            self._Text = Text
            self._Progress = Progress
            self._BarColumn = BarColumn
            self._TextColumn = TextColumn
            self._TimeElapsedColumn = TimeElapsedColumn
            self._console = Console()
            self._rich_ok = True
        except ImportError:
            self._console = None

    def _build_meta(self) -> dict:
        gpu, driver = "unknown", "unknown"
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                parts = [p.strip() for p in out.split(",")]
                gpu = parts[0] if parts else gpu
                driver = parts[1] if len(parts) > 1 else driver
        except Exception:
            gpu = "UNAVAILABLE"
        return {
            "run_id": self.run_id,
            "task": self.args.task,
            "code_root": str(self.root),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "lewm"),
            "FBLEWM_ROOT": str(self.root),
            "STABLEWM_HOME": self.env["STABLEWM_HOME"],
            "data_dir": self.env["LOCAL_DATASET_DIR"],
            "models_dir": str(self.root / "models"),
            "outputs_dir": str(self.root / "outputs"),
            "HF_HOME": self.env["HF_HOME"],
            "gpu": gpu,
            "driver": driver,
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "train_run_name": getattr(self.args, "train_run_name", None),
            "forward_variant": getattr(self.args, "forward_variant", "latent"),
            "forward_action_weight": getattr(self.args, "forward_action_weight", 1.0),
            "forward_teacher_weight": getattr(self.args, "forward_teacher_weight", 1.0),
            "forward_branches": getattr(self.args, "forward_branches", 4),
            "forward_history_size": 2,
            "log_dir": str(self.log_dir),
        }

    def _bind_stage_runners(self) -> None:
        self.stages[0].run_fn = self._stage_setup
        self.stages[1].run_fn = self._stage_deps
        self.stages[2].run_fn = self._stage_config
        self.stages[3].run_fn = self._stage_data_link
        self.stages[4].run_fn = self._stage_gpu_gate
        self.stages[5].run_fn = self._stage_model_smoke
        self.stages[6].build_cmd = self._cmd_train
        self.stages[7].build_cmd = self._cmd_eval

    # ----- stage implementations -----

    def _stage_setup(self, stage: Stage, log: Path) -> int:
        stage.progress.total = 1
        stage.progress.current = 0
        stage.progress.brief = "path lock"
        self._refresh_ui(stage)
        with log.open("w") as f:
            f.write(f"FBLEWM_ROOT={self.root}\n")
            for k in (
                "STABLEWM_HOME",
                "LOCAL_DATASET_DIR",
                "HF_HOME",
                "PYTHONPATH",
            ):
                f.write(f"{k}={self.env.get(k)}\n")
            required = [
                self.root / "train.py",
                self.root / "fblewm.py",
                self.root / "module.py",
                self.root / "policy.py",
                self.root / "scripts" / "eval_fblewm_matrix.py",
            ]
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                f.write(f"missing required files: {missing}\n")
                stage.progress.brief = "missing source files"
                return 1
            f.write("setup ok\n")
        stage.progress.current = 1
        stage.progress.brief = "setup ok"
        return 0

    def _stage_deps(self, stage: Stage, log: Path) -> int:
        req = self.root / "scripts" / "requirements-fblewm.txt"
        pkgs = [
            "torch",
            "stable_worldmodel",
            "stable_pretraining",
            "rich",
            "tqdm",
            "huggingface_hub",
            "h5py",
            "hdf5plugin",
            "zstandard",
        ]
        stage.progress.total = len(pkgs) + 1
        missing = []
        with log.open("w") as f:
            for i, name in enumerate(pkgs, 1):
                stage.progress.current = i
                stage.progress.brief = f"check {name}"
                self._refresh_ui(stage)
                r = subprocess.run(
                    [sys.executable, "-c", f"import {name}"],
                    capture_output=True,
                    text=True,
                    env=self.env,
                )
                f.write(f"check {name}: rc={r.returncode}\n")
                if r.returncode != 0:
                    missing.append(name)
            if missing:
                f.write(f"missing: {missing}\ninstalling from {req}\n")
                stage.progress.brief = f"pip install ({len(missing)} missing)"
                self._refresh_ui(stage)
                cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(req),
                ]
                rc = self._run_cmd(cmd, stage, log, append=True)
                stage.progress.current = stage.progress.total
                return rc
            f.write("all deps present\n")
            stage.progress.current = stage.progress.total
            stage.progress.brief = "deps ok"
        return 0

    def _stage_config(self, stage: Stage, log: Path) -> int:
        stage.progress.total = 2
        stage.progress.current = 0
        stage.progress.brief = "resolve configs"
        self._refresh_ui(stage)
        spec = task_spec(self.args.task)
        train_yaml = self.root / "config" / "train" / "fblewm.yaml"
        eval_yaml = self.root / "config" / "eval" / f"{spec.eval_config}.yaml"
        data_yaml = self.root / "config" / "train" / "data" / f"{spec.train_data}.yaml"
        with log.open("w") as f:
            if not train_yaml.exists() or not eval_yaml.exists() or not data_yaml.exists():
                f.write(
                    f"missing configs: train={train_yaml.exists()} "
                    f"eval={eval_yaml.exists()} data={data_yaml.exists()}\n"
                    f"  {train_yaml}\n  {eval_yaml}\n  {data_yaml}\n"
                )
                return 1
            # Dump resolved train config via hydra compose (no training).
            train_script = (
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(self.root)!r})\n"
                "from hydra import compose, initialize_config_dir\n"
                "from omegaconf import OmegaConf\n"
                f"cfg_dir = {str(self.root / 'config' / 'train')!r}\n"
                f"overrides = {['data=' + spec.train_data] + self._forward_train_overrides()!r}\n"
                "with initialize_config_dir(version_base=None, config_dir=cfg_dir):\n"
                "    cfg = compose(config_name='fblewm', overrides=overrides)\n"
                f"out = Path({str(self.log_dir / 'resolved_train.yaml')!r})\n"
                "out.write_text(OmegaConf.to_yaml(cfg))\n"
                "print(out)\n"
            )
            cmd = [sys.executable, "-c", train_script]
            r = subprocess.run(cmd, capture_output=True, text=True, env=self.env, cwd=str(self.root))
            f.write(r.stdout)
            f.write(r.stderr)
            if r.returncode != 0:
                stage.progress.brief = "train config resolve failed"
                return 1
            stage.progress.current = 1
            self._refresh_ui(stage)
            eval_script = (
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(self.root)!r})\n"
                "from hydra import compose, initialize_config_dir\n"
                "from omegaconf import OmegaConf\n"
                f"cfg_dir = {str(self.root / 'config' / 'eval')!r}\n"
                "with initialize_config_dir(version_base=None, config_dir=cfg_dir):\n"
                f"    cfg = compose(config_name={spec.eval_config!r})\n"
                f"out = Path({str(self.log_dir / 'resolved_eval.yaml')!r})\n"
                "out.write_text(OmegaConf.to_yaml(cfg))\n"
                "print(out)\n"
            )
            cmd2 = [sys.executable, "-c", eval_script]
            r2 = subprocess.run(cmd2, capture_output=True, text=True, env=self.env, cwd=str(self.root))
            f.write(r2.stdout)
            f.write(r2.stderr)
            if r2.returncode != 0:
                stage.progress.brief = "eval config resolve failed"
                return 1
            (self.log_dir / "00_metadata.json").write_text(json.dumps(self.meta, indent=2))
            f.write("config ok\n")
        stage.progress.current = 2
        stage.progress.brief = "config ok"
        return 0

    def _extract_tar_zst(self, zst: Path, extract_dir: Path, logf) -> int:
        extract_dir.mkdir(parents=True, exist_ok=True)
        logf.write(f"extract {zst} -> {extract_dir}\n")
        # Prefer tar with zstd; fall back to zstd|tar.
        cmd = ["tar", "--use-compress-program=zstd", "-xf", str(zst), "-C", str(extract_dir)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        logf.write(r.stdout)
        logf.write(r.stderr)
        if r.returncode == 0:
            return 0
        logf.write("tar --use-compress-program=zstd failed; trying zstd -dc | tar\n")
        p1 = subprocess.Popen(
            ["zstd", "-dc", str(zst)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p2 = subprocess.run(
            ["tar", "-xf", "-", "-C", str(extract_dir)],
            stdin=p1.stdout,
            capture_output=True,
            text=True,
        )
        if p1.stdout:
            p1.stdout.close()
        err = p1.communicate()[1]
        logf.write(err.decode() if isinstance(err, (bytes, bytearray)) else (err or ""))
        logf.write(p2.stdout)
        logf.write(p2.stderr)
        return int(p2.returncode)

    def _stage_data_link(self, stage: Stage, log: Path) -> int:
        stage.progress.total = 1
        stage.progress.current = 0
        stage.progress.brief = "link dataset"
        self._refresh_ui(stage)
        spec = task_spec(self.args.task)
        # Shared data root (PushT / Cube / TwoRoom): LeWM/data via LOCAL_DATASET_DIR.
        local = Path(self.env["LOCAL_DATASET_DIR"])
        incoming_fallback = self.root / "data" / "incoming"
        # Extract beside shared data so LeWM and FBLeWM both see the same files.
        extracted = local / "extracted" / spec.task
        ds_dir = Path(self.env["STABLEWM_HOME"]) / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        local.mkdir(parents=True, exist_ok=True)
        incoming_fallback.mkdir(parents=True, exist_ok=True)

        with log.open("w") as f:
            f.write(f"task={spec.task}\n")
            f.write(f"LOCAL_DATASET_DIR={local}  # shared LeWM/FBLeWM data root\n")
            f.write(f"incoming_fallback={incoming_fallback}\n")

            # ---- PushT: legacy path (unchanged semantics) ----
            if spec.task == "pusht":
                ok = False
                for name in ("pusht_expert_train.h5", "pusht_expert_train.lance"):
                    src = local / name
                    link = ds_dir / name
                    f.write(f"check {src} exists={src.exists()}\n")
                    if src.exists():
                        ok = True
                        if not link.exists():
                            try:
                                link.symlink_to(src.resolve())
                                f.write(f"symlink {link} -> {src}\n")
                            except OSError as e:
                                f.write(f"symlink fail {link}: {e}\n")
                        else:
                            f.write(f"link exists {link}\n")
                if not ok:
                    f.write("ERROR: PushT dataset missing under LOCAL_DATASET_DIR\n")
                    stage.progress.brief = "dataset missing"
                    return 1
                f.write("data_link ok\n")
                stage.progress.current = 1
                stage.progress.brief = "data_link ok"
                return 0

            # ---- Cube / TwoRoom / Reacher: zst under LeWM/data (no server HF download) ----
            search_roots = (local, local / "extracted", extracted, incoming_fallback)
            ready = _find_existing_ready(spec, search_roots)
            if ready is None:
                # Prefer shared LeWM/data; FBLeWM/data/incoming is legacy fallback only.
                zst = _find_zst(spec, (local, incoming_fallback))
                if zst is None:
                    f.write(
                        "ERROR: dataset missing. Upload the .tar.zst to shared "
                        "LeWM/data then re-run.\n"
                    )
                    f.write(f"Primary upload dir:\n  {local}/\n")
                    f.write(f"Fallback:\n  {incoming_fallback}/\n")
                    for name, url in zip(spec.zst_names, spec.hf_urls):
                        f.write(f"  file: {name}\n  download (laptop): {url}\n")
                    stage.progress.brief = f"upload zst to {local.name}/"
                    return 1
                stage.progress.brief = f"extract {zst.name}"
                self._refresh_ui(stage)
                rc = self._extract_tar_zst(zst, extracted, f)
                if rc != 0:
                    stage.progress.brief = "extract failed"
                    return rc
                ready = _find_h5_under(extracted, spec.h5_basenames)
                if ready is None:
                    ready = _find_existing_ready(spec, (extracted, local, incoming_fallback))
                if ready is None:
                    f.write(
                        f"ERROR: extract finished but h5 not found "
                        f"(looked for {spec.h5_basenames}) under {extracted}\n"
                    )
                    stage.progress.brief = "h5 missing after extract"
                    return 1
                f.write(f"extracted h5: {ready}\n")
            else:
                f.write(f"reuse ready dataset: {ready}\n")

            # Symlink into STABLEWM_HOME/datasets with expected relative names.
            for rel in spec.dataset_links:
                link = ds_dir / rel
                try:
                    _symlink_force(link, ready, f)
                except Exception as e:
                    f.write(f"ERROR symlink {link}: {e}\n")
                    stage.progress.brief = "symlink failed"
                    return 1
                # Keep a stable name under shared LOCAL_DATASET_DIR when missing.
                local_link = local / rel
                if not local_link.exists():
                    try:
                        _symlink_force(local_link, ready, f)
                    except Exception as e:
                        f.write(f"WARN local symlink {local_link}: {e}\n")

            f.write("data_link ok\n")
        stage.progress.current = 1
        stage.progress.brief = "data_link ok"
        return 0

    def _stage_gpu_gate(self, stage: Stage, log: Path) -> int:
        stage.progress.total = 2
        with log.open("w") as f:
            stage.progress.current = 1
            stage.progress.brief = "nvidia-smi"
            self._refresh_ui(stage)
            r1 = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                env=self.env,
            )
            f.write(r1.stdout)
            f.write(r1.stderr)
            if r1.returncode != 0 or "mismatch" in (r1.stdout + r1.stderr).lower():
                stage.progress.brief = "nvidia-smi failed; see log"
                return 1
            stage.progress.current = 2
            stage.progress.brief = "torch.cuda"
            self._refresh_ui(stage)
            r2 = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import torch; assert torch.cuda.is_available(), 'cuda unavailable'; "
                    "print(torch.cuda.get_device_name(0))",
                ],
                capture_output=True,
                text=True,
                env=self.env,
            )
            f.write(r2.stdout)
            f.write(r2.stderr)
            if r2.returncode != 0:
                stage.progress.brief = "torch.cuda failed; see log"
                return 1
            stage.progress.brief = r2.stdout.strip()[:100]
        return 0

    def _stage_model_smoke(self, stage: Stage, log: Path) -> int:
        stage.progress.total = 3
        probe = r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
import torch
import stable_worldmodel as swm
import stable_pretraining as spt
from module import (
    CausalLatentImaginer,
    ConditionalLatentImaginer,
    ActionAlignedCausalLatentImaginer,
    SequentialActionCausalLatentImaginer,
    BranchPreservingCausalLatentImaginer,
    SIGReg,
)
from fblewm import FBLeWM
from policy import compute_imagine_steps
assert compute_imagine_steps(75, 0, 25, 5) == 10
head = ActionAlignedCausalLatentImaginer(dim=8, hidden_dim=16, depth=1, action_dim=10)
z = torch.randn(2, 3, 8)
a_hat, z_hat = head.forward_with_action(z)
assert z_hat.shape == z.shape and a_hat.shape == (2, 3, 10)
assert torch.allclose(head(z), z_hat)
seq = SequentialActionCausalLatentImaginer(dim=8, hidden_dim=16, depth=1, action_dim=10)
a_seq = seq.predict_action(z)
z_a = seq.transition(z, a_seq)
z_b = seq.transition(z, a_seq + 1)
assert a_seq.shape == (2, 3, 10) and z_a.shape == z.shape
assert not torch.allclose(z_a, z_b)
assert torch.allclose(seq(z), seq.forward_with_action(z)[1])
bp = BranchPreservingCausalLatentImaginer(dim=8, hidden_dim=16, depth=1, num_branches=3)
h = torch.randn(2, 2, 8)
y = bp.forward_branches(h)
assert y.shape == (2, 3, 8)
assert not torch.allclose(y[:, 0], y[:, 1])
h_asg = h.unsqueeze(1).expand(2, 3, 2, 8).contiguous()
y_asg = bp.forward_assigned(h_asg)
assert y_asg.shape == (2, 3, 8)
p0_m = h[:, 1].unsqueeze(1).expand(2, 3, 8)
h_roll = torch.stack([p0_m, y], dim=2)
y_roll = bp.forward_assigned(h_roll)
assert y_roll.shape == (2, 3, 8)
print('imports_ok')
print('cuda', torch.cuda.is_available())
"""
        stage.progress.current = 1
        stage.progress.brief = "import stack"
        self._refresh_ui(stage)
        rc = self._run_cmd([sys.executable, "-c", probe], stage, log)
        if rc != 0:
            return rc
        stage.progress.current = 2
        stage.progress.brief = "compile train.py"
        self._refresh_ui(stage)
        rc = self._run_cmd(
            [sys.executable, "-m", "py_compile", str(self.root / "train.py")],
            stage,
            log,
            append=True,
        )
        if rc != 0:
            return rc
        stage.progress.current = 3
        stage.progress.brief = "model stack ok"
        return 0

    def _resolved_eval_modes(self) -> str:
        modes = getattr(self.args, "eval_modes", None)
        variant = getattr(self.args, "forward_variant", None) or "latent"
        if modes:
            return modes
        if variant == "branch_preserving":
            return "official,forward"
        return "official,forward,backward"

    def _forward_train_overrides(self) -> List[str]:
        """Hydra overrides for the Forward variant. Eval still uses mode=forward."""
        variant = getattr(self.args, "forward_variant", None) or "latent"
        action_weight = getattr(self.args, "forward_action_weight", None)
        if action_weight is None:
            action_weight = 1.0
        teacher_weight = getattr(self.args, "forward_teacher_weight", None)
        if teacher_weight is None:
            teacher_weight = 1.0
        overrides = [
            f"loss.forward.variant={variant}",
            f"loss.forward.action_weight={action_weight}",
            f"loss.forward.teacher_weight={teacher_weight}",
        ]
        if variant in ("action_aligned", "sequential_action"):
            overrides.append("loss.backward.weight=0.0")
        if variant == "branch_preserving":
            branches = getattr(self.args, "forward_branches", None)
            if branches is None:
                branches = 4
            overrides.append(f"loss.forward.branches={int(branches)}")
        return overrides

    def _cmd_train(self, pipe: "Pipeline") -> List[str]:
        hydra_dir = str(self.root / "outputs" / "hydra" / self.run_id)
        spec = task_spec(self.args.task)
        run_name = getattr(self.args, "train_run_name", None) or spec.default_run_name
        b_target = getattr(self.args, "backward_target", None) or spec.default_backward_target
        cmd = [
            sys.executable,
            str(self.root / "train.py"),
            f"data={spec.train_data}",
            f"trainer.max_epochs={self.args.epochs}",
            f"loader.batch_size={self.args.batch_size}",
            f"output_model_name={run_name}",
            f"loss.backward.target={b_target}",
            "wandb.enabled=false",
            f"hydra.run.dir={hydra_dir}",
            "hydra.job.chdir=false",
        ]
        # Tests call Pipeline._cmd_train(fake, None); resolve via the class.
        cmd.extend(Pipeline._forward_train_overrides(self))
        if b_target in ("now", "pred_goal", "fixed_bridge"):
            cmd.append(
                "model.backward_imaginer._target_=module.ConditionalLatentImaginer"
            )
        if b_target == "now":
            cmd.append("model.backward_anchor=obs")
        elif b_target in ("pred_goal", "fixed_bridge"):
            cmd.append("model.backward_anchor=pred")
        if b_target == "fixed_bridge":
            cmd.append("loss.backward.p_noise=0.0")
            cmd.append("loss.backward.goal_rank_weight=0.0")
        return cmd

    def _cmd_eval(self, pipe: "Pipeline") -> List[str]:
        ckpt_root = Path(self.env["STABLEWM_HOME"]) / "checkpoints"
        spec = task_spec(self.args.task)
        policy = self.args.policy
        if not policy:
            # Only search this task's run dir (never pick another task's ckpt).
            preferred = ckpt_root / spec.default_run_name
            policy = f"{spec.default_run_name}/weights_epoch_10.pt"
            if preferred.is_dir():
                epoch_pts = sorted(
                    preferred.glob("weights_epoch_*.pt"),
                    key=lambda p: (
                        int(p.stem.split("_")[-1])
                        if p.stem.split("_")[-1].isdigit()
                        else -1,
                        p.stat().st_mtime,
                    ),
                )
                if epoch_pts:
                    policy = str(epoch_pts[-1].relative_to(ckpt_root))
        # Ensure eval HDF5 is visible (task-specific; PushT keeps legacy links).
        data = Path(self.env["LOCAL_DATASET_DIR"])
        ds_dir = Path(self.env["STABLEWM_HOME"]) / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        for rel in spec.dataset_links:
            src_candidates = [
                data / rel,
                self.root / "data" / "extracted" / spec.task / Path(rel).name,
            ]
            # Also accept ready_names locations.
            for rn in spec.ready_names:
                src_candidates.append(data / rn)
                src_candidates.append(self.root / "data" / "extracted" / spec.task / Path(rn).name)
            link = ds_dir / rel
            if link.exists():
                continue
            for src in src_candidates:
                if src.exists():
                    try:
                        link.parent.mkdir(parents=True, exist_ok=True)
                        link.symlink_to(src.resolve())
                    except OSError:
                        pass
                    break
        hydra_dir = str(self.root / "outputs" / "eval" / self.run_id)
        if getattr(self.args, "eval_dir", None):
            hydra_dir = str(self.args.eval_dir)
        elif getattr(self.args, "eval_resume_dir", None):
            hydra_dir = str(self.args.eval_resume_dir)
        modes = Pipeline._resolved_eval_modes(self)
        offsets = getattr(self.args, "eval_offsets", None) or "25,50,75,100"
        cmd = [
            sys.executable,
            str(self.root / "scripts" / "eval_fblewm_matrix.py"),
            f"--policy={policy}",
            f"--cache-dir={self.env['STABLEWM_HOME']}",
            f"--config-name={spec.eval_config}",
            f"--modes={modes}",
            f"--offsets={offsets}",
            f"--hydra-run-dir={hydra_dir}",
        ]
        if getattr(self.args, "eval_horizon", None) is not None:
            cmd.append(f"--horizon={int(self.args.eval_horizon)}")
        if getattr(self.args, "eval_receding_horizon", None) is not None:
            cmd.append(f"--receding-horizon={int(self.args.eval_receding_horizon)}")
        starts_manifest = getattr(self.args, "starts_manifest", None)
        if starts_manifest:
            cmd.append(f"--starts-manifest={starts_manifest}")
        elif getattr(self.args, "eval_resume_dir", None):
            starts = Path(self.args.eval_resume_dir) / "starts_manifest.json"
            if starts.exists():
                cmd.append(f"--starts-manifest={starts}")
        if getattr(self.args, "eval_resume_dir", None):
            cmd.append("--resume")
        if getattr(self.args, "backward_depth_cap", None) is not None:
            cmd.append(f"--backward-depth-cap={int(self.args.backward_depth_cap)}")
        if getattr(self.args, "record_cem_cost", False):
            cmd.append("--record-cem-cost")
        return cmd

    @staticmethod
    def _count_eval_units(modes_arg: str, offsets_arg: str) -> int:
        """Best-effort unit count for Rich progress total."""
        raw = [m.strip() for m in modes_arg.split(",") if m.strip()]
        n_modes = 0
        for m in raw:
            if m == "fusion":
                n_modes += 7
            elif m == "all":
                n_modes += 10
            else:
                n_modes += 1
        n_off = len([x for x in offsets_arg.split(",") if x.strip()])
        return max(n_modes * max(n_off, 1), 1)

    # ----- process runner -----

    def _run_cmd(
        self,
        cmd: List[str],
        stage: Stage,
        log: Path,
        append: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        mode = "a" if append else "w"
        run_env = env if env is not None else self.env
        with log.open(mode) as f:
            f.write(f"$ {shlex.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=run_env,
                cwd=str(self.root),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
                parse_progress(line, stage.progress, stage.stage_id)
                self._refresh_ui(stage)
            rc = proc.wait()
            stage.returncode = rc
            f.write(f"\n[exit {rc}]\n")
            return rc

    def _emit(self, stage: Stage, status: str) -> None:
        ev = {
            "ts": time.time(),
            "stage": stage.stage_id,
            "status": status,
            "current": stage.progress.current,
            "total": stage.progress.total,
            "brief": stage.progress.brief,
            "rc": stage.returncode,
        }
        self._events.append(ev)
        with (self.log_dir / "stages.jsonl").open("a") as f:
            f.write(json.dumps(ev) + "\n")

    # ----- UI -----

    def _render(self, active: Optional[Stage] = None):
        if not self._rich_ok:
            return None
        Table = self._Table
        Panel = self._Panel
        Text = self._Text
        from rich.console import Group
        from rich.progress_bar import ProgressBar

        board = Table(title="Stage Board", expand=True)
        board.add_column("#", width=3)
        board.add_column("stage", width=12)
        board.add_column("status", width=10)
        board.add_column("progress", width=18)
        board.add_column("brief")
        for s in self.stages:
            if s.progress.unit == "B":
                pct = (
                    f"{_format_amount(s.progress.current, 'B')}/"
                    f"{_format_amount(s.progress.total, 'B')}"
                )
            else:
                pct = f"{s.progress.current}/{s.progress.total}"
            status_style = {
                "FAILED": "bold red",
                "RUNNING": "bold cyan",
                "DONE": "green",
                "SKIPPED": "dim",
            }.get(s.status, "")
            brief = s.progress.error_msg or s.progress.brief or ""
            if s.status == "FAILED" and not brief.startswith("ERROR"):
                brief = f"ERROR: {brief}" if brief else "ERROR"
            board.add_row(
                str(s.idx),
                s.stage_id,
                Text(s.status, style=status_style) if status_style else s.status,
                pct,
                Text(brief[:70], style="red") if s.status == "FAILED" else brief[:70],
            )

        # Prefer showing FAILED stage detail even after loop moved on
        focus = active
        if focus is None:
            for s in self.stages:
                if s.status == "FAILED":
                    focus = s
                    break

        if focus and focus.status == "FAILED":
            err = focus.progress.error_msg or focus.progress.brief or "ERROR"
            detail = Group(
                Text(f"[{focus.idx}/8] {focus.stage_id}  FAILED", style="bold red"),
                Text(err, style="bold red"),
                Text(f"log -> {focus.log_path}", style="red"),
                Text("Fix the error above, then re-run with --from-stage ...", style="yellow"),
            )
            panel_title = "ERROR"
            panel_style = "red"
        elif focus and focus.status == "RUNNING":
            st = focus.progress
            tot = max(int(st.total), 1)
            cur = max(0, min(int(st.current), tot))
            if st.unit == "epoch" and st.sub_total > 0:
                done_units = max(cur - 1, 0) * st.sub_total + st.sub_current
                total_units = tot * st.sub_total
                frac = min(1.0, done_units / max(total_units, 1))
            else:
                frac = cur / tot
            bar = ProgressBar(total=100, completed=int(frac * 100), width=40)
            unit_name = {
                "B": "bytes",
                "epoch": "epoch",
                "%": "percent",
                "it": "it",
            }.get(st.unit, st.unit)
            if st.unit == "B":
                counts = (
                    f"{_format_amount(cur, 'B')} / {_format_amount(tot, 'B')} "
                    f"({100 * frac:5.1f}%)"
                )
            else:
                counts = f"{cur} / {tot} {unit_name} ({100 * frac:5.1f}%)"
            eta = _format_duration(_eta_seconds(st))
            elapsed = _format_duration(
                time.time() - st.started_at if st.started_at > 0 else None
            )
            sub = ""
            if st.unit == "epoch" and st.sub_total > 0:
                sub = (
                    f"  |  {st.sub_label} {st.sub_current}/{st.sub_total}"
                )
            detail = Group(
                Text(f"[{focus.idx}/8] {focus.stage_id}  RUNNING", style="bold cyan"),
                bar,
                Text(f"{counts}{sub}  elapsed {elapsed}  ETA {eta}"),
                Text(f"{st.brief or ''}", style="white"),
                Text(f"log -> {focus.log_path}", style="dim"),
            )
            panel_title = "Active progress"
            panel_style = "cyan"
        else:
            detail = Text("idle", style="dim")
            panel_title = "Active progress"
            panel_style = "dim"
        return Group(board, Panel(detail, title=panel_title, border_style=panel_style))

    def _print_metadata_once(self) -> None:
        if self._rich_ok:
            t = self._Table(title="LeWM Pipeline Metadata", show_header=True)
            t.add_column("key")
            t.add_column("value")
            for k, v in self.meta.items():
                t.add_row(k, str(v))
            self._console.print(t)
        else:
            print("=== Metadata ===")
            for k, v in self.meta.items():
                print(f"  {k}: {v}")
            print("=== Stages ===")
            for s in self.stages:
                print(f"  [{s.idx}] {s.stage_id:12s} {s.status}")

    def _start_ui_ticker(self, stage: Stage) -> None:
        """Keep Live elapsed/ETA updating ~4 Hz while a stage is RUNNING."""
        self._stop_ui_ticker()
        self._ui_active = stage
        stop = threading.Event()
        self._ui_tick_stop = stop

        def _tick() -> None:
            while not stop.wait(0.25):
                active = self._ui_active
                if active is None or active.status != "RUNNING":
                    continue
                self._refresh_ui(active)

        t = threading.Thread(target=_tick, name="lewm-ui-tick", daemon=True)
        self._ui_tick_thread = t
        t.start()

    def _stop_ui_ticker(self) -> None:
        stop = self._ui_tick_stop
        thr = self._ui_tick_thread
        self._ui_tick_stop = None
        self._ui_tick_thread = None
        self._ui_active = None
        if stop is not None:
            stop.set()
        if thr is not None and thr.is_alive():
            thr.join(timeout=1.0)

    def _refresh_ui(self, active: Optional[Stage] = None) -> None:
        with self._ui_lock:
            if self._live is not None:
                self._live.update(self._render(active))
            elif active:
                # Fallback single-line tqdm-like bar (carriage return, no spam)
                st = active.progress
                tot = max(int(st.total), 1)
                cur = max(0, min(int(st.current), tot))
                bar = _text_bar(cur, tot, width=28)
                if st.unit == "B":
                    counts = (
                        f"{_format_amount(cur, 'B')}/"
                        f"{_format_amount(tot, 'B')}"
                    )
                else:
                    counts = f"{cur}/{tot}"
                eta = _format_duration(_eta_seconds(st))
                elapsed = _format_duration(
                    time.time() - st.started_at if st.started_at > 0 else None
                )
                brief = (st.brief or "")[:28]
                print(
                    f"\r{active.stage_id:10s} |{bar}| {counts} "
                    f"elapsed {elapsed} ETA {eta} {brief}   ",
                    end="",
                    flush=True,
                )

    def _should_skip(self, stage: Stage) -> bool:
        if self.args.only_stage and stage.stage_id != self.args.only_stage:
            return True
        if stage.stage_id == "deps" and self.args.skip_deps:
            return True
        if stage.stage_id == "eval" and self.args.skip_eval:
            return True
        if self.args.from_stage:
            order = [s.stage_id for s in self.stages]
            if order.index(stage.stage_id) < order.index(self.args.from_stage):
                return True
        return False

    def run(self) -> int:
        (self.log_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        self._print_metadata_once()

        live_cm = None
        if self._rich_ok:
            live_cm = self._Live(
                self._render(None),
                console=self._console,
                refresh_per_second=12,
            )
            self._live = live_cm.__enter__()

        final_rc = 0
        try:
            for stage in self.stages:
                log = self.log_dir / f"{stage.idx:02d}_{stage.stage_id}.log"
                stage.log_path = log
                if self._should_skip(stage):
                    stage.status = "SKIPPED"
                    self._emit(stage, "SKIPPED")
                    self._refresh_ui(None)
                    continue

                stage.status = "RUNNING"
                stage.started = time.time()
                stage.progress = ProgressState(
                    current=0,
                    total=1,
                    brief="starting",
                    started_at=stage.started,
                )
                self._emit(stage, "RUNNING")
                self._start_ui_ticker(stage)
                self._refresh_ui(stage)

                try:
                    if stage.run_fn is not None:
                        rc = stage.run_fn(stage, log)
                    elif stage.build_cmd is not None:
                        cmd = stage.build_cmd(self)
                        if stage.stage_id == "train":
                            stage.progress.unit = "epoch"
                            stage.progress.total = max(self.args.epochs, 1)
                            stage.progress.current = 0
                            stage.progress.started_at = stage.started
                            stage.progress.brief = f"training {self.args.epochs} epochs"
                        elif stage.stage_id == "eval":
                            stage.progress.unit = "it"
                            modes = Pipeline._resolved_eval_modes(self)
                            offsets = getattr(self.args, "eval_offsets", None) or (
                                "25,50,75,100"
                            )
                            n_units = self._count_eval_units(modes, offsets)
                            # Progress advances only on eval i/N DONE lines.
                            stage.progress.total = n_units
                            stage.progress.current = 0
                            stage.progress.started_at = stage.started
                            stage.progress.brief = f"modes={modes} offsets={offsets}"
                        self._refresh_ui(stage)
                        rc = self._run_cmd(cmd, stage, log)
                    else:
                        rc = 1
                        log.write_text("no runner bound\n")
                except Exception as e:
                    log.write_text(f"exception: {e}\n")
                    rc = 1
                finally:
                    self._stop_ui_ticker()

                stage.ended = time.time()
                stage.returncode = rc
                if rc == 0:
                    stage.status = "DONE"
                    stage.progress.current = stage.progress.total
                    self._emit(stage, "DONE")
                else:
                    stage.status = "FAILED"
                    summary = _extract_error_summary(log)
                    stage.progress.error_msg = summary
                    stage.progress.brief = summary
                    self._emit(stage, "FAILED")
                    final_rc = rc
                    self._refresh_ui(stage)
                    # Also print a persistent red error after Live exits
                    self._pending_error = (stage.stage_id, summary, str(log))
                    break
                self._refresh_ui(None)
        finally:
            self._stop_ui_ticker()
            if live_cm is not None:
                live_cm.__exit__(None, None, None)
                self._live = None

        self._print_final()
        if self._pending_error and self._rich_ok:
            sid, summary, logp = self._pending_error
            self._console.print(
                self._Panel(
                    self._Text(
                        f"stage={sid}\n{summary}\nlog={logp}\n"
                        f"hint: on this HKU host use official HF "
                        f"(unset HF_ENDPOINT); avoid hf-mirror for this dataset",
                        style="bold red",
                    ),
                    title="ERROR",
                    border_style="red",
                )
            )
        elif self._pending_error:
            sid, summary, logp = self._pending_error
            print(f"\nERROR [{sid}]\n{summary}\nlog={logp}\n", file=sys.stderr)
        return final_rc

    def _print_final(self) -> None:
        rows = []
        for s in self.stages:
            dur = (
                f"{(s.ended - s.started):.1f}s"
                if s.ended and s.started
                else "-"
            )
            rows.append(
                (
                    s.stage_id,
                    s.status,
                    dur,
                    str(s.log_path) if s.log_path else "-",
                )
            )
        if self._rich_ok:
            t = self._Table(title="Final Summary")
            t.add_column("stage")
            t.add_column("status")
            t.add_column("duration")
            t.add_column("log")
            for r in rows:
                t.add_row(*r)
            self._console.print(t)
            self._console.print(f"run_id={self.run_id}")
            self._console.print(f"log_dir={self.log_dir}")
            ckpt = Path(self.env["STABLEWM_HOME"]) / "checkpoints"
            self._console.print(f"checkpoints_dir={ckpt}")
        else:
            print("\n=== Final Summary ===")
            for r in rows:
                print("  ", r)
            print("log_dir=", self.log_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FBLeWM rich-popen-tqdm pipeline")
    p.add_argument(
        "--task",
        default="pusht",
        choices=sorted(TASK_SPECS.keys()),
        help="Environment task (default: pusht; cube/tworoom/reacher reuse same progress UI)",
    )
    p.add_argument("--epochs", type=int, default=10, help="Default PushT epochs is 10")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-deps", action="store_true")
    p.add_argument("--from-stage", default=None, help="Start from stage id")
    p.add_argument("--only-stage", default=None, help="Run a single stage id")
    p.add_argument(
        "--policy",
        default=None,
        help="Eval policy path relative to STABLEWM_HOME/checkpoints",
    )
    p.add_argument(
        "--eval-resume-dir",
        default=None,
        help="Resume matrix eval from an existing outputs/eval/<run_id> directory",
    )
    p.add_argument(
        "--eval-modes",
        default=None,
        help=(
            "Eval modes for matrix script (e.g. fusion, fusion_avg05,meet, or all). "
            "Default official,forward,backward; branch_preserving defaults to "
            "official,forward."
        ),
    )
    p.add_argument(
        "--eval-offsets",
        default="25,50,75,100",
        help="Comma-separated goal offsets",
    )
    p.add_argument(
        "--starts-manifest",
        default=None,
        help="Reuse starts_manifest.json for fair comparison across runs",
    )
    p.add_argument(
        "--train-run-name",
        default=None,
        help=(
            "Checkpoint dir under STABLEWM_HOME/checkpoints "
            "(default: fblewm_bp for pusht, fblewm_tworoom_v2 / fblewm_cube_v2 for others; "
            "never overwrite protected fblewm / fblewm_bp / fblewm_tworoom / fblewm_cube)"
        ),
    )
    p.add_argument(
        "--forward-variant",
        default="latent",
        choices=["latent", "action_aligned", "sequential_action", "branch_preserving"],
        help=(
            "Training Forward head: latent=F(p)->z (legacy); "
            "action_aligned=F(p)->(A_hat,z_hat); "
            "sequential_action=A=G(p), z'=H(p,A); "
            "branch_preserving=F_m([z_{t-1},z_t])->z_{t+1}. "
            "Eval still uses mode=forward."
        ),
    )
    p.add_argument(
        "--forward-action-weight",
        type=float,
        default=1.0,
        help="Weight on action MSE inside forward_loss for action-aligned variants",
    )
    p.add_argument(
        "--forward-teacher-weight",
        type=float,
        default=1.0,
        help="Weight on teacher-forced H(p, A_tgt) when --forward-variant=sequential_action",
    )
    p.add_argument(
        "--forward-branches",
        type=int,
        default=4,
        help="Number of Forward heads when --forward-variant=branch_preserving",
    )
    p.add_argument(
        "--backward-target",
        default=None,
        choices=["pred", "encoder", "now", "pred_goal", "fixed_bridge"],
        help=(
            "B objective: pred=unary B(z)->p, encoder=unary B(z)->z, "
            "now=B(z_now, z_goal)->z, pred_goal=B(P, z_later)->z, "
            "fixed_bridge=B(P1, z_later) with frozen P1 "
            "(default: pred for pusht, now for tworoom, encoder for cube)"
        ),
    )
    p.add_argument(
        "--eval-horizon",
        type=int,
        default=None,
        help="CEM latent horizon override (e.g. 2 → plan_len=10; enables F/B at offset=25)",
    )
    p.add_argument(
        "--eval-receding-horizon",
        type=int,
        default=None,
        help="Receding horizon override (default: same as --eval-horizon)",
    )
    p.add_argument(
        "--eval-dir",
        default=None,
        help=(
            "Write matrix eval here instead of outputs/eval/<run_id>. "
            "Use outputs/diag/... for diagnostic ablations."
        ),
    )
    p.add_argument(
        "--backward-depth-cap",
        type=int,
        default=None,
        help="Eval-only pred_goal recursion cap (min(k, cap)); default unset.",
    )
    p.add_argument(
        "--record-cem-cost",
        action="store_true",
        help="Record real CEM candidate-cost traces during eval.",
    )
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not FBLEWM_ROOT.is_dir():
        print(f"FBLEWM_ROOT missing: {FBLEWM_ROOT}", file=sys.stderr)
        return 2
    if not (FBLEWM_ROOT / "train.py").exists():
        print(f"train.py not found under {FBLEWM_ROOT}", file=sys.stderr)
        return 2
    spec = task_spec(args.task)
    if getattr(args, "train_run_name", None) is None:
        args.train_run_name = spec.default_run_name
    if getattr(args, "backward_target", None) is None:
        args.backward_target = spec.default_backward_target
    pipe = Pipeline(args)
    return pipe.run()


if __name__ == "__main__":
    sys.exit(main())
