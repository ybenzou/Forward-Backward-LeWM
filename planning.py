"""Planning-mode names and helpers for FBLeWM fusion strategies."""

from __future__ import annotations

# Base modes from the original plan.
BASE_MODES = ("official", "forward", "backward")

# Opt-in post-hoc terminal metrics. Kept out of BASE_MODES so existing
# train/eval commands retain their exact default mode set.
TRM_MODES = ("trm_replace", "trm_hybrid")

# Cost-level fusion / stage-switch / meet-in-the-middle (eval-only strategies).
FUSION_MODES = (
    "fusion_avg05",  # C = 0.5 C_F + 0.5 C_B
    "fusion_avg07",  # C = 0.7 C_F + 0.3 C_B  (favor Forward)
    "fusion_ofb",  # C = (C_official + C_F + C_B) / 3
    "fusion_max",  # C = max(C_F, C_B)
    "fusion_min",  # C = min(C_F, C_B)
    "switch_remain",  # remain>50 → F, else B
    "switch_offset",  # offset>=100 → F, else fusion_avg05
    "meet",  # MSE(F^{k_f}(P), B^{k_b}(z_goal)), k_f+k_b=k
)

ALL_PLANNING_MODES = BASE_MODES + FUSION_MODES + TRM_MODES

# Modes that need both F^k and B^k inside get_cost (true final goal, not B-subgoal inject).
MODES_NEED_DUAL_COST = frozenset(
    {
        "fusion_avg05",
        "fusion_avg07",
        "fusion_ofb",
        "fusion_max",
        "fusion_min",
        "switch_remain",
        "switch_offset",
        "meet",
    }
)

# Modes where policy injects B^k(z_goal) as goal_emb (pure backward).
MODES_INJECT_BACKWARD_SUBGOAL = frozenset({"backward"})

# Modes that pass per-env branch ids into CEM info.
MODES_NEED_PLAN_BRANCH = frozenset({"switch_remain"})


def resolve_fusion_alpha(mode: str) -> float | None:
    if mode == "fusion_avg05":
        return 0.5
    if mode == "fusion_avg07":
        return 0.7
    if mode == "switch_offset":
        # Used when offset < cutoff.
        return 0.5
    return None


def coarsen_backward_steps(k: int, max_steps: int = 3, block: int = 5) -> int:
    """Map fine imagination depth k (action-blocks) to a few B(z0, g) pulls.

    k=0 stays 0. Otherwise ``ceil(k / block)`` capped at ``max_steps``.
    With the default schedule this is 0/1/2/3 for offsets 25/50/75/100.
    """
    k = int(k)
    if k < 0:
        raise ValueError(f"backward steps must be >= 0, got {k}")
    if k == 0:
        return 0
    max_steps = int(max_steps)
    block = int(block)
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    return min(max_steps, max(1, (k + block - 1) // block))


def split_meet_steps(k: int) -> tuple[int, int]:
    """Split total imagination depth into (k_forward, k_backward)."""
    k = int(k)
    if k < 0:
        raise ValueError(f"meet steps must be >= 0, got {k}")
    k_f = k // 2
    k_b = k - k_f
    return k_f, k_b
