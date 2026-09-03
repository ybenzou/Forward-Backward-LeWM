#!/usr/bin/env python3
"""One-command TwoRoom TRM: train head, then evaluate on frozen manifests."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_trm_baseline, train_trm

DEFAULT_SEEDS = tuple(range(42, 52))
DEFAULT_OUT = ROOT / "outputs" / "baselines" / "trm_compare" / "v1"
DEFAULT_MANIFEST_ROOT = ROOT / "outputs" / "diag" / "iclr_bar" / "multiseed"
DEFAULT_POLICY = "fblewm_tworoom/weights_epoch_10.pt"
DEFAULT_DATA_NAMES = ("tworoom.h5", "datasets/tworoom.h5")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TwoRoom TRM train + eval")
    parser.add_argument("--stage", choices=("all", "train", "eval"), default="all")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Eval groups. Default: 42-51. Use 42 for a pilot.",
    )
    parser.add_argument("--offsets", default="25,50,75,100")
    return parser.parse_args(argv)


def _cache_dir() -> Path:
    return Path(os.environ.get("STABLEWM_HOME", ROOT / ".stable-wm")).resolve()


def _resolve_data(cache_dir: Path) -> Path:
    searched = [cache_dir / name for name in DEFAULT_DATA_NAMES]
    local = os.environ.get("LOCAL_DATASET_DIR")
    if local:
        searched.append(Path(local) / "tworoom.h5")
    searched.append(Path("/home/yuanben/WorldModel/LeWM/data/tworoom.h5"))
    for path in searched:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "TwoRoom dataset not found; searched:\n"
        + "\n".join(f"  - {path}" for path in searched)
    )


def _exclude_manifests() -> list[str]:
    paths = [
        DEFAULT_MANIFEST_ROOT / "tworoom" / f"seed_{seed}" / "starts_manifest.json"
        for seed in DEFAULT_SEEDS
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing frozen TwoRoom manifests:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    return [str(path) for path in paths]


def _train() -> int:
    cache_dir = _cache_dir()
    argv = [
        "--task",
        "tworoom",
        "--checkpoint",
        DEFAULT_POLICY,
        "--data",
        str(_resolve_data(cache_dir)),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(DEFAULT_OUT),
    ]
    for path in _exclude_manifests():
        argv.extend(["--exclude-manifest", path])
    print("[trm tworoom] train", flush=True)
    return int(train_trm.main(argv))


def _eval(seeds: str, offsets: str) -> int:
    head = DEFAULT_OUT / "heads" / "tworoom" / "true.pt"
    argv = [
        "--task",
        "tworoom",
        "--trm-head",
        str(head),
        "--seeds",
        seeds,
        "--offsets",
        offsets,
        "--cache-dir",
        str(_cache_dir()),
        "--manifest-root",
        str(DEFAULT_MANIFEST_ROOT),
        "--out-root",
        str(DEFAULT_OUT / "eval"),
    ]
    print("[trm tworoom] eval", flush=True)
    return int(run_trm_baseline.main(argv))


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if args.stage in ("all", "train"):
        code = _train()
        if code != 0:
            return code
    if args.stage in ("all", "eval"):
        return _eval(args.seeds, args.offsets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
