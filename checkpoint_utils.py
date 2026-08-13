"""FBLeWM checkpoint loading helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate


ROOT = Path(__file__).resolve().parent


def ensure_repo_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def load_fblewm_checkpoint(name: str, cache_dir: str | None = None):
    """Load FBLeWM via stable_worldmodel, validating imaginer weights exist.

    Raises RuntimeError if the checkpoint looks like official LeWM (missing F/B).
    """
    ensure_repo_on_path()
    from stable_worldmodel.wm.utils import get_cache_dir

    cache = Path(get_cache_dir(cache_dir, sub_folder="checkpoints"))
    searched = [
        cache / name,
        Path(name),
        # Fallback: always also try FBLeWM's own checkpoint root.
        ROOT / ".stable-wm" / "checkpoints" / name,
    ]
    pt_path = next((p for p in searched if p.exists()), None)
    if pt_path is None:
        raise FileNotFoundError(
            f"checkpoint not found: {name}\n"
            f"  searched:\n"
            + "\n".join(f"    - {p}" for p in searched)
            + "\n  Hint: source scripts/env.sh so STABLEWM_HOME points to "
            f"{ROOT / '.stable-wm'} (not LeWM)."
        )

    cfg_path = pt_path.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found next to {pt_path}")

    with open(cfg_path) as f:
        config = json.load(f)

    state = torch.load(pt_path, map_location="cpu")
    has_f = any(k.startswith("forward_imaginer.") for k in state)
    has_b = any(k.startswith("backward_imaginer.") for k in state)
    if not (has_f and has_b):
        raise RuntimeError(
            "This is an official LeWM checkpoint, not an FBLeWM checkpoint "
            "(missing forward_imaginer / backward_imaginer weights). "
            f"path={pt_path}"
        )

    target = config.get("_target_", "")
    if "fblewm" not in target.lower() and "FBLeWM" not in target:
        # Still allow if weights contain F/B, but warn via exception if clearly JEPA-only.
        if target.endswith("JEPA") or target.endswith("jepa.JEPA"):
            raise RuntimeError(
                "This is an official LeWM checkpoint, not an FBLeWM checkpoint "
                f"(_target_={target}). path={pt_path}"
            )

    model = instantiate(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Require imaginer keys present in state; ignore unexpected.
    missing_imag = [m for m in missing if "imaginer" in m]
    if missing_imag:
        raise RuntimeError(
            "FBLeWM checkpoint is missing imaginer parameters: "
            f"{missing_imag[:8]}..."
        )
    return model
