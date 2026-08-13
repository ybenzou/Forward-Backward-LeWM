"""FBLeWM planning policy with dynamic imagination depth."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.solver.solver import Solver

from planning import (
    ALL_PLANNING_MODES,
    MODES_INJECT_BACKWARD_SUBGOAL,
    MODES_NEED_PLAN_BRANCH,
)


def compute_imagine_steps(
    goal_offset: int,
    elapsed: int,
    plan_len: int,
    action_block: int,
) -> int:
    """Return imagination recursion depth k for one replan.

    k = max((goal_offset - elapsed - plan_len) / action_block, 0)

    Raises ValueError if offsets are not divisible by action_block / plan_len rules.
    """
    goal_offset = int(goal_offset)
    elapsed = int(elapsed)
    plan_len = int(plan_len)
    action_block = int(action_block)

    if action_block <= 0:
        raise ValueError(f"action_block must be > 0, got {action_block}")
    if plan_len <= 0:
        raise ValueError(f"plan_len must be > 0, got {plan_len}")
    if goal_offset % action_block != 0:
        raise ValueError(
            f"goal_offset={goal_offset} must be divisible by action_block={action_block}"
        )
    if plan_len % action_block != 0:
        raise ValueError(
            f"plan_len={plan_len} must be divisible by action_block={action_block}"
        )

    remain = goal_offset - elapsed - plan_len
    if remain <= 0:
        return 0
    if remain % action_block != 0:
        raise ValueError(
            f"non-divisible imagination remainder: "
            f"(goal_offset={goal_offset} - elapsed={elapsed} - plan_len={plan_len}) "
            f"= {remain} not divisible by action_block={action_block}"
        )
    return remain // action_block


def expected_replan_depths(goal_offset: int, plan_len: int = 25, action_block: int = 5):
    """Helper for tests: depths at elapsed = 0, plan_len, 2*plan_len, ... until k=0."""
    depths = []
    elapsed = 0
    while True:
        k = compute_imagine_steps(goal_offset, elapsed, plan_len, action_block)
        depths.append(k)
        if k == 0:
            break
        elapsed += plan_len
        if elapsed > goal_offset + plan_len:
            break
    return depths


def compute_switch_remain_branch(
    goal_offset: int,
    elapsed: int,
    threshold: int = 50,
) -> int:
    """0 = forward, 1 = backward. remaining env steps to goal = offset - elapsed."""
    remaining = int(goal_offset) - int(elapsed)
    return 0 if remaining > int(threshold) else 1


class FBWorldModelPolicy(WorldModelPolicy):
    """WorldModelPolicy with per-env elapsed tracking and F/B imagination hooks."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig,
        *,
        goal_offset: int,
        planning_mode: str = "official",
        process=None,
        transform=None,
        switch_remain_threshold: int = 50,
        switch_offset_cutoff: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            solver=solver, config=config, process=process, transform=transform, **kwargs
        )
        if planning_mode not in ALL_PLANNING_MODES:
            raise ValueError(
                f"unknown planning_mode {planning_mode!r}; expected one of {ALL_PLANNING_MODES}"
            )
        self.goal_offset = int(goal_offset)
        self.planning_mode = planning_mode
        self.switch_remain_threshold = int(switch_remain_threshold)
        self.switch_offset_cutoff = int(switch_offset_cutoff)
        self._elapsed_steps: np.ndarray | None = None
        self.last_imagine_steps: list[int] = []
        self.last_plan_branches: list[int] = []
        self.model = getattr(solver, "model", None)
        self._configure_model_planning()

    def _configure_model_planning(self) -> None:
        model = self.model
        if model is None:
            return
        if hasattr(model, "set_planning_mode"):
            model.set_planning_mode(self.planning_mode)
        if hasattr(model, "set_goal_offset"):
            model.set_goal_offset(self.goal_offset)
        if hasattr(model, "set_switch_remain_threshold"):
            model.set_switch_remain_threshold(self.switch_remain_threshold)
        if hasattr(model, "set_switch_offset_cutoff"):
            model.set_switch_offset_cutoff(self.switch_offset_cutoff)

    @property
    def plan_len_env(self) -> int:
        return int(self.cfg.horizon * self.cfg.action_block)

    def set_env(self, env: Any) -> None:
        self.env = env
        n_envs = getattr(env, "num_envs", 1)
        self.solver.configure(
            action_space=env.action_space, n_envs=n_envs, config=self.cfg
        )
        self._action_buffer = [
            deque(maxlen=self.flatten_receding_horizon) for _ in range(n_envs)
        ]
        self._elapsed_steps = np.zeros(n_envs, dtype=np.int64)
        self._next_init = None
        assert isinstance(self.solver, Solver), "Solver must implement the Solver protocol"

    def _encode_goal_pixels(self, sliced: dict) -> torch.Tensor:
        """Encode goal image (and optional goal state) -> (B, T_g, D).

        World eval stores goal columns as ``goal_<name>`` (including
        ``goal_action``). Remapping ``goal_action`` → ``action`` must NOT
        feed the action encoder: env actions are raw dim-2, while
        ``Embedder`` expects frameskip-blocked dim-10.
        """
        model = self.solver.model
        device = next(model.parameters()).device
        pixels = sliced["goal"].to(device)
        if pixels.ndim == 6:
            # CEM-expanded (B, S, T, C, H, W) — take first sample.
            pixels = pixels[:, 0]
        goal: dict[str, Any] = {"pixels": pixels}
        # Optional proprio/state goal keys only (never action).
        skip_suffixes = {
            "action",
            "emb",
            "act_emb",
            "goal_emb",
            "imagine_steps",
            "predicted_emb",
            "plan_branch",
        }
        for k, v in list(sliced.items()):
            if not (k.startswith("goal_") and torch.is_tensor(v)):
                continue
            short = k[len("goal_") :]
            if short in skip_suffixes or short.startswith("goal"):
                continue
            goal[short] = v.to(device)
        # Belt-and-suspenders: encode() would otherwise run action_encoder.
        goal.pop("action", None)
        encoded = model.encode(goal)
        return encoded["emb"]  # (B, T_g, D)

    def _encode_now_pixels(self, sliced: dict) -> torch.Tensor:
        """Encode current observation pixels -> (B, D) last-frame latent."""
        model = self.solver.model
        device = next(model.parameters()).device
        pixels = sliced["pixels"].to(device)
        if pixels.ndim == 6:
            pixels = pixels[:, 0]
        now = {"pixels": pixels}
        now.pop("action", None)
        encoded = model.encode(now)
        return encoded["emb"][:, -1, :]

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert self._elapsed_steps is not None, "call set_env first"
        assert self._action_buffer is not None

        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs

        needs_flush = info_dict.pop("_needs_flush", None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()
                    if self._next_init is not None:
                        self._next_init[i] = 0
                    self._elapsed_steps[i] = 0

        terminated = info_dict.get("terminated")
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(n_envs, dtype=bool)
        )

        replan_idx = [
            i
            for i in range(n_envs)
            if len(self._action_buffer[i]) == 0 and not dead[i]
        ]

        if replan_idx:
            idx_tensor = torch.as_tensor(replan_idx, dtype=torch.long)
            sliced = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    sliced[k] = v[idx_tensor]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[replan_idx]
                elif isinstance(v, list):
                    sliced[k] = [v[i] for i in replan_idx]
                else:
                    sliced[k] = v

            # Per-env imagination depths at replan time.
            ks = [
                compute_imagine_steps(
                    self.goal_offset,
                    int(self._elapsed_steps[i]),
                    self.plan_len_env,
                    int(self.cfg.action_block),
                )
                for i in replan_idx
            ]
            self.last_imagine_steps = list(ks)
            k_tensor = torch.tensor(ks, dtype=torch.int64).view(-1, 1)
            sliced["imagine_steps"] = k_tensor

            model = self.solver.model
            self.model = model
            self._configure_model_planning()

            if self.planning_mode in MODES_NEED_PLAN_BRANCH:
                branches = [
                    compute_switch_remain_branch(
                        self.goal_offset,
                        int(self._elapsed_steps[i]),
                        self.switch_remain_threshold,
                    )
                    for i in replan_idx
                ]
                self.last_plan_branches = list(branches)
                sliced["plan_branch"] = torch.tensor(branches, dtype=torch.int64).view(
                    -1, 1
                )

            if self.planning_mode in MODES_INJECT_BACKWARD_SUBGOAL:
                # Precompute B(z_now, ... B(z_now, z_goal)) once per env (not per CEM sample).
                with torch.no_grad():
                    goal_emb = self._encode_goal_pixels(sliced)  # (B, Tg, D)
                    z_goal = goal_emb[:, -1, :]  # (B, D)
                    z_now = self._encode_now_pixels(sliced)  # (B, D)
                    z_sub = model.imagine_backward(
                        z_goal, k_tensor.view(-1), z_now=z_now
                    )
                    sliced["goal_emb"] = z_sub.unsqueeze(1)  # (B, 1, D)

            sliced_init = (
                self._next_init[idx_tensor] if self._next_init is not None else None
            )

            outputs = self.solver(sliced, init_action=sliced_init)

            actions = outputs["actions"]
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]

            if self.cfg.warm_start and rest.shape[1] > 0:
                if self._next_init is None:
                    self._next_init = torch.zeros(
                        n_envs, rest.shape[1], rest.shape[2], dtype=rest.dtype
                    )
                self._next_init[idx_tensor] = rest
            elif not self.cfg.warm_start:
                self._next_init = None

            plan = plan.reshape(
                len(replan_idx), self.flatten_receding_horizon, -1
            )

            for row, env_i in enumerate(replan_idx):
                self._action_buffer[env_i].extend(plan[row])

        action_dim = self.env.single_action_space.shape[-1]
        action = torch.full((n_envs, action_dim), float("nan"))
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
                self._elapsed_steps[i] += 1

        action = action.reshape(*self.env.action_space.shape)
        action = action.float().numpy()

        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        return action
