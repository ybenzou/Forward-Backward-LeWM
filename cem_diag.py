"""CEM cost-trace callback for Backward diagnostics.

Records real candidate-cost statistics per CEM iteration. Default eval
does not attach this callback; success-rate files stay unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from stable_worldmodel.solver.callbacks.common import Callback


def _per_env_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().float().cpu().reshape(-1).tolist()]


class CemCostTrace(Callback):
    """Accumulate per-replan CEM cost stats without touching planning."""

    name = "cem_cost_trace"

    def __init__(self) -> None:
        super().__init__(reduction="none")
        self.records: list[dict[str, Any]] = []
        self._replan_meta: dict[str, Any] = {}
        self._t0 = 0.0

    def begin_replan(self, **meta: Any) -> None:
        self._replan_meta = dict(meta)
        self._t0 = time.time()

    def compute(self, **state: Any) -> dict[str, list[float]]:
        costs = state["costs"].detach().float()
        topk_vals = state["topk_vals"].detach().float()
        topk_candidates = state["topk_candidates"].detach().float()
        cost_min = costs.min(dim=1).values
        cost_max = costs.max(dim=1).values
        elite_spread = topk_candidates.std(dim=1).flatten(1).mean(dim=-1)
        return {
            "cost_mean": _per_env_list(costs.mean(dim=1)),
            "cost_std": _per_env_list(costs.std(dim=1, unbiased=False)),
            "cost_min": _per_env_list(cost_min),
            "cost_max": _per_env_list(cost_max),
            "cost_range": _per_env_list(cost_max - cost_min),
            "elite_cost_mean": _per_env_list(topk_vals.mean(dim=1)),
            "elite_cost_min": _per_env_list(topk_vals.min(dim=1).values),
            "elite_spread": _per_env_list(elite_spread),
        }

    def end_replan(self) -> None:
        rec = {
            **self._replan_meta,
            "solve_seconds": float(time.time() - self._t0),
            "iterations": list(self.history),
        }
        self.records.append(rec)

    def summarize(self) -> dict[str, Any]:
        """Collapse records into coarse collapse / spread diagnostics."""
        stds: list[float] = []
        ranges: list[float] = []
        means: list[float] = []
        mins: list[float] = []
        elite_spreads: list[float] = []
        for rec in self.records:
            for batch in rec.get("iterations", []):
                for step in batch:
                    if not isinstance(step, dict):
                        continue
                    stds.extend(step.get("cost_std", []))
                    ranges.extend(step.get("cost_range", []))
                    means.extend(step.get("cost_mean", []))
                    mins.extend(step.get("cost_min", []))
                    elite_spreads.extend(step.get("elite_spread", []))
        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else float("nan")

        return {
            "n_replans": len(self.records),
            "n_cost_rows": len(stds),
            "cost_std_mean": _mean(stds),
            "cost_range_mean": _mean(ranges),
            "cost_mean_mean": _mean(means),
            "cost_min_mean": _mean(mins),
            "elite_spread_mean": _mean(elite_spreads),
            "cost_std_collapsed": bool(stds) and _mean(stds) < 1e-6,
        }
