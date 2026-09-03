#!/usr/bin/env python3
"""Summarize Backward diagnostic-gate results into SUMMARY.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "outputs" / "diag" / "backward_gate_v3"

OFFICIAL = {
    "pusht": {
        "eval": ROOT / "outputs" / "eval" / "20260817_115825_pusht" / "summary.txt",
        "lewm": {50: 42.0, 75: 14.0, 100: 16.0},
        "forward": {50: 66.0, 75: 40.0, 100: 20.0},
        "backward": {50: 38.0, 75: 14.0, 100: 2.0},
    },
    "tworoom": {
        "eval": ROOT / "outputs" / "eval" / "20260817_035625_tworoom" / "summary.txt",
        "lewm": {50: 44.0, 75: 32.0, 100: 16.0},
        "forward": {50: 88.0, 75: 88.0, 100: 64.0},
        "backward": {50: 92.0, 75: 90.0, 100: 60.0},
    },
}

VARIANTS = ("eval_cap1", "eval_cap2", "eval_cap5", "eval_meet")


def _load_rates(summary_json: Path) -> dict[tuple[str, int], dict]:
    if not summary_json.exists():
        return {}
    data = json.loads(summary_json.read_text())
    out = {}
    for row in data.get("results", []):
        out[(row["mode"], int(row["offset"]))] = row
    return out


def _load_cem(eval_dir: Path) -> dict[str, dict]:
    path = eval_dir / "cem_cost.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[f"{rec['mode']}_{rec['offset']}"] = rec.get("summary", {})
    return out


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.1f}"


def main() -> int:
    lines = [
        "# Backward diagnostic gate",
        "",
        "Diagnostic only. Official v3 eval dirs and paper figures were not overwritten.",
        "",
    ]
    rates = {}
    cem = {}
    latent = {}
    for task in ("pusht", "tworoom"):
        lines.append(f"## {task}")
        lines.append("")
        ref = OFFICIAL[task]
        lines.append("| variant | 50 | 75 | 100 | CEM std | collapsed |")
        lines.append("|---|---:|---:|---:|---:|---|")
        lines.append(
            f"| LeWM (ref) | {ref['lewm'][50]:.1f} | {ref['lewm'][75]:.1f} | {ref['lewm'][100]:.1f} | - | - |"
        )
        lines.append(
            f"| Forward (ref) | {ref['forward'][50]:.1f} | {ref['forward'][75]:.1f} | {ref['forward'][100]:.1f} | - | - |"
        )
        lines.append(
            f"| Backward uncapped (ref) | {ref['backward'][50]:.1f} | {ref['backward'][75]:.1f} | {ref['backward'][100]:.1f} | - | - |"
        )
        task_rates = {}
        for name in VARIANTS:
            eval_dir = DIAG / task / name
            rowmap = _load_rates(eval_dir / "results.json")
            cem[f"{task}/{name}"] = _load_cem(eval_dir)
            mode = "meet" if name == "eval_meet" else "backward"
            vals = []
            for off in (50, 75, 100):
                rec = rowmap.get((mode, off))
                vals.append(None if rec is None else float(rec["success_rate"]))
                if rec is not None:
                    task_rates[(name, off)] = float(rec["success_rate"])
            cem_s = list(cem[f"{task}/{name}"].values())
            std = None
            collapsed = None
            if cem_s:
                stds = [c.get("cost_std_mean") for c in cem_s if c.get("cost_std_mean") == c.get("cost_std_mean")]
                std = sum(stds) / len(stds) if stds else None
                collapsed = any(bool(c.get("cost_std_collapsed")) for c in cem_s)
            lines.append(
                f"| {name} | {_fmt(vals[0])} | {_fmt(vals[1])} | {_fmt(vals[2])} | "
                f"{'-' if std is None else f'{std:.4g}'} | {collapsed} |"
            )
        rates[task] = task_rates
        lat_path = DIAG / task / "latent" / "latent_trace.json"
        if lat_path.exists():
            latent[task] = json.loads(lat_path.read_text())
            lines.append("")
            lines.append("Latent probe (true vs shuffled goal):")
            for off, rec in latent[task].get("offsets", {}).items():
                lines.append(
                    f"- offset {off}: k={rec['k']} identity_gap={rec['identity_gap']['mean']:.4f} "
                    f"sep={rec['goal_separation_final']['mean']:.4f} "
                    f"true<shuf={rec['true_closer_than_shuffled']:.2f} "
                    f"norm[0→k]={rec['norm']['mean'][0]:.3f}→{rec['norm']['mean'][-1]:.3f}"
                )
        lines.append("")

    # Gate
    def _ge_lewm(task: str, variant: str, offs=(50, 75)) -> bool:
        ref = OFFICIAL[task]
        got = rates.get(task, {})
        return all(
            (variant, off) in got and got[(variant, off)] + 1e-9 >= ref["lewm"][off]
            for off in offs
        )

    def _saves_pusht(variant: str) -> bool:
        got = rates.get("pusht", {})
        ref = OFFICIAL["pusht"]
        # "明显救回": 50/75 both above uncapped B and at least LeWM
        return all(
            (variant, off) in got
            and got[(variant, off)] >= ref["lewm"][off]
            and got[(variant, off)] > ref["backward"][off]
            for off in (50, 75)
        )

    def _tworoom_above_lewm(variant: str) -> bool:
        return _ge_lewm("tworoom", variant)

    def _cem_ok(task: str, variant: str) -> bool:
        recs = cem.get(f"{task}/{variant}", {})
        if not recs:
            return False
        return not any(bool(c.get("cost_std_collapsed")) for c in recs.values())

    missing = [
        f"{task}/{name}"
        for task in ("pusht", "tworoom")
        for name in VARIANTS
        if not (DIAG / task / name / "results.json").exists()
    ]
    lines.append("## Gate decision")
    lines.append("")
    if missing:
        lines.append("Incomplete runs: " + ", ".join(missing))
        lines.append("No-Go until all same-start ablations finish.")
        decision = "NO-GO (incomplete)"
        formula = "n/a"
    else:
        cap12_ok = (
            (_saves_pusht("eval_cap1") or _saves_pusht("eval_cap2"))
            and (_tworoom_above_lewm("eval_cap1") or _tworoom_above_lewm("eval_cap2"))
            and _cem_ok("pusht", "eval_cap1")
            and _cem_ok("tworoom", "eval_cap1")
        )
        meet_ok = (
            _ge_lewm("pusht", "eval_meet")
            and _ge_lewm("tworoom", "eval_meet")
            and _cem_ok("pusht", "eval_meet")
            and _cem_ok("tworoom", "eval_meet")
        )
        if cap12_ok:
            decision = "GO"
            formula = "depth-conditioned B(P, z_g, k) in one call; train long-range, no recursion"
        elif meet_ok:
            decision = "GO"
            formula = "trained meet-in-the-middle: align F^{k_f}(P) and B^{k_b}(z_g) at the same time"
        else:
            decision = "NO-GO for cap/meet as a unified interface"
            formula = (
                "Temporal Bridge B(z_now, z_goal, remaining) -> z_{t+25} "
                "with true waypoint supervision; do not consume candidate P"
            )
        # Final must: both tasks Backward 50/75 >= LeWM and cost var not collapsed
        if decision == "GO":
            chosen = "eval_cap1" if _saves_pusht("eval_cap1") else (
                "eval_cap2" if cap12_ok else "eval_meet"
            )
            if not (
                _ge_lewm("pusht", chosen)
                and _ge_lewm("tworoom", chosen)
                and _cem_ok("pusht", chosen)
                and _cem_ok("tworoom", chosen)
            ):
                decision = "NO-GO (fails joint 50/75 >= LeWM or cost-collapse test)"
        lines.append(f"- Decision: **{decision}**")
        lines.append(f"- Next B formula: `{formula}`")
        lines.append(
            "- Paper claims and `outputs/figures/` were not updated in this diagnostic round."
        )

    text = "\n".join(lines) + "\n"
    out = DIAG / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
