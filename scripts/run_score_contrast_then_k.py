#!/usr/bin/env python3
"""Sequential interpretability jobs, resume-safe.

1) 400 held-out windows: F^5 / F^10 / F^15 score histograms
2) 10-group planning: fixed k=5,10,15 at o=75/100
   (k=0 = LeWM and dynamic = HAS are reused from the main matrix)
3) Redraw the 10-group k figure
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SCRIPTS = ROOT / "scripts"


def main() -> int:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")

    steps = [
        [PY, str(SCRIPTS / "analyze_has_score_contrast.py")],
        [PY, str(SCRIPTS / "run_k_ablation_multiseed.py")],
    ]
    for cmd in steps:
        print("\n==== RUN ====\n" + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
