#!/usr/bin/env python3
"""Collect seed-42 fixed-k Forward evals into a LaTeX table fragment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "diag" / "k_ablation"
TASKS = ("pusht", "tworoom", "reacher")
TITLES = {"pusht": "PushT", "tworoom": "TwoRoom", "reacher": "Reacher"}
CONDS = (
    ("k0", r"$k{=}0$"),
    ("k5", r"$k{=}5$"),
    ("dynamic", "dynamic"),
    ("k15", r"$k{=}15$"),
)
OFFSETS = (75, 100)


def _rate(path: Path, offset: int) -> str:
    if not path.exists():
        return "---"
    rec = json.loads(path.read_text())
    for row in rec.get("results", []):
        if row.get("mode") == "forward" and int(row.get("offset")) == offset:
            return f"{float(row['success_rate']):.1f}"
    summary = path.with_name("summary.txt")
    if summary.exists():
        for line in summary.read_text().splitlines():
            if not line.startswith("mode="):
                continue
            parts = dict(x.split("=", 1) for x in line.split() if "=" in x)
            if parts.get("mode") == "forward" and int(parts["offset"]) == offset:
                return f"{float(parts['success_rate']):.1f}"
    return "---"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper" / "tables" / "tab_k_depth.tex",
    )
    args = p.parse_args()

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Task & $o$ & $k{=}0$ & $k{=}5$ & dynamic & $k{=}15$ \\",
        r"\midrule",
    ]
    for task in TASKS:
        for offset in OFFSETS:
            cells = [_rate(args.root / task / cond / "results.json", offset) for cond, _ in CONDS]
            o_tex = r"$75$" if offset == 75 else r"$100$"
            lines.append(
                f"{TITLES[task]} & {o_tex} & " + " & ".join(cells) + r" \\"
            )
        if task != TASKS[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
