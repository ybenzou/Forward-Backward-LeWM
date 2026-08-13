"""FBLeWM: official LeWM + detached Forward/Backward Causal Latent Imaginers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from planning import (
    ALL_PLANNING_MODES,
    coarsen_backward_steps,
    resolve_fusion_alpha,
    split_meet_steps,
)

PLANNING_MODES = ALL_PLANNING_MODES


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def _as_int_steps(steps) -> int:
    if torch.is_tensor(steps):
        if steps.numel() != 1:
            raise ValueError(f"scalar steps expected, got shape {tuple(steps.shape)}")
        steps = int(steps.item())
    else:
        steps = int(steps)
    if steps < 0:
        raise ValueError(f"imagine steps must be >= 0, got {steps}")
    return steps


class FBLeWM(nn.Module):
    """Official LeWM world model plus independent Forward/Backward imaginers."""

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        forward_imaginer,
        backward_imaginer,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.forward_imaginer = forward_imaginer
        self.backward_imaginer = backward_imaginer
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self._planning_mode = "official"
        self._fusion_alpha = 0.5
        self._goal_offset = 25
        self._switch_remain_threshold = 50
        self._switch_offset_cutoff = 100

    def set_planning_mode(self, mode: str) -> None:
        if mode not in PLANNING_MODES:
            raise ValueError(
                f"unknown planning mode {mode!r}; expected one of {PLANNING_MODES}"
            )
        self._planning_mode = mode
        alpha = resolve_fusion_alpha(mode)
        if alpha is not None:
            self._fusion_alpha = float(alpha)

    def set_goal_offset(self, goal_offset: int) -> None:
        self._goal_offset = int(goal_offset)

    def set_switch_remain_threshold(self, threshold: int) -> None:
        self._switch_remain_threshold = int(threshold)

    def set_switch_offset_cutoff(self, cutoff: int) -> None:
        self._switch_offset_cutoff = int(cutoff)

    @property
    def planning_mode(self) -> str:
        return self._planning_mode

    def encode(self, info):
        """Encode observations and actions into embeddings."""
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding. emb/act_emb: (B, T, D)."""
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def _recurse_imaginer(self, imaginer: nn.Module, z: torch.Tensor, steps) -> torch.Tensor:
        """Recursively apply imaginer for a scalar number of steps."""
        n = _as_int_steps(steps)
        if n == 0:
            return z
        out = z
        for _ in range(n):
            out = imaginer(out)
        return out

    def _recurse_imaginer_masked(
        self, imaginer: nn.Module, z: torch.Tensor, steps: torch.Tensor
    ) -> torch.Tensor:
        """Apply imaginer with per-sample step counts.

        z: (..., D)
        steps: broadcastable to z[..., 0] as int64, e.g. (B,), (B,1), (B,S), (B,S,1)
        """
        if not torch.is_tensor(steps):
            return self._recurse_imaginer(imaginer, z, steps)

        steps = steps.to(device=z.device, dtype=torch.long)
        while steps.ndim < z.ndim - 1:
            steps = steps.unsqueeze(-1)
        if steps.ndim == z.ndim:
            if steps.size(-1) != 1:
                raise ValueError(
                    f"steps last dim must be 1 when matching z ndim, got {tuple(steps.shape)}"
                )
            steps = steps.squeeze(-1)
        if steps.shape != z.shape[:-1]:
            steps = steps.expand(z.shape[:-1])

        max_k = int(steps.max().item()) if steps.numel() else 0
        if max_k < 0:
            raise ValueError(f"imagine steps must be >= 0, got min={int(steps.min().item())}")
        if max_k == 0:
            return z

        out = z
        for step_index in range(max_k):
            mask = steps > step_index
            if not bool(mask.any()):
                break
            nxt = imaginer(out)
            out = torch.where(mask.unsqueeze(-1), nxt, out)
        return out

    def imagine_forward(self, z: torch.Tensor, steps=1) -> torch.Tensor:
        if torch.is_tensor(steps) and steps.numel() != 1:
            return self._recurse_imaginer_masked(self.forward_imaginer, z, steps)
        return self._recurse_imaginer(self.forward_imaginer, z, steps)

    def _is_conditional_backward(self) -> bool:
        return bool(getattr(self.backward_imaginer, "is_conditional", False))

    def _coarsen_backward_steps(self, steps, *, max_steps: int = 3, block: int = 5):
        """Cap/coarsen k for conditional B; leave legacy unary B on the fine schedule."""
        if not self._is_conditional_backward():
            return steps
        if not torch.is_tensor(steps):
            return coarsen_backward_steps(steps, max_steps=max_steps, block=block)
        flat = steps.reshape(-1)
        out = torch.empty_like(flat)
        for i in range(flat.numel()):
            out[i] = coarsen_backward_steps(
                int(flat[i].item()), max_steps=max_steps, block=block
            )
        return out.reshape(steps.shape)

    def _apply_backward(self, z_now: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return self.backward_imaginer(z_now, z_goal)

    def _recurse_conditional_backward(
        self, z_now: torch.Tensor, z_goal: torch.Tensor, steps
    ) -> torch.Tensor:
        n = _as_int_steps(steps)
        if n == 0:
            return z_goal
        g = z_goal
        for _ in range(n):
            g = self._apply_backward(z_now, g)
        return g

    def _recurse_conditional_backward_masked(
        self, z_now: torch.Tensor, z_goal: torch.Tensor, steps: torch.Tensor
    ) -> torch.Tensor:
        steps = steps.to(device=z_goal.device, dtype=torch.long)
        while steps.ndim < z_goal.ndim - 1:
            steps = steps.unsqueeze(-1)
        if steps.ndim == z_goal.ndim:
            if steps.size(-1) != 1:
                raise ValueError(
                    f"steps last dim must be 1 when matching z ndim, got {tuple(steps.shape)}"
                )
            steps = steps.squeeze(-1)
        if steps.shape != z_goal.shape[:-1]:
            steps = steps.expand(z_goal.shape[:-1])

        max_k = int(steps.max().item()) if steps.numel() else 0
        if max_k < 0:
            raise ValueError(
                f"imagine steps must be >= 0, got min={int(steps.min().item())}"
            )
        if max_k == 0:
            return z_goal

        g = z_goal
        for step_index in range(max_k):
            mask = steps > step_index
            if not bool(mask.any()):
                break
            nxt = self._apply_backward(z_now, g)
            g = torch.where(mask.unsqueeze(-1), nxt, g)
        return g

    def imagine_backward(
        self, z_goal: torch.Tensor, steps=1, z_now: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Roll goal backward. Conditional B: g <- B(z_now, g) with z_now frozen."""
        if not self._is_conditional_backward():
            if torch.is_tensor(steps) and steps.numel() != 1:
                return self._recurse_imaginer_masked(
                    self.backward_imaginer, z_goal, steps
                )
            return self._recurse_imaginer(self.backward_imaginer, z_goal, steps)

        steps = self._coarsen_backward_steps(steps)
        if not torch.is_tensor(steps):
            if _as_int_steps(steps) == 0:
                return z_goal
        elif int(steps.max().item()) == 0:
            return z_goal

        if z_now is None:
            raise ValueError(
                "conditional backward imaginer requires z_now "
                "(current observation latent); got z_now=None"
            )
        if torch.is_tensor(steps) and steps.numel() != 1:
            return self._recurse_conditional_backward_masked(z_now, z_goal, steps)
        return self._recurse_conditional_backward(z_now, z_goal, steps)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the official predictor given action candidates.

        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
        """
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        if "emb" not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            # Avoid encoding goal tensors as current state.
            _init = {
                k: v
                for k, v in _init.items()
                if k
                not in (
                    "goal",
                    "goal_emb",
                    "imagine_steps",
                    "predicted_emb",
                    "plan_branch",
                )
                and not k.startswith("goal_")
            }
            _init = self.encode(_init)
            info["emb"] = _init["emb"].detach().unsqueeze(1).expand(B, S, -1, -1)

        emb = rearrange(info["emb"], "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: dict):
        """Terminal MSE between predicted embeddings and goal embeddings -> (B, S)."""
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]

        # Normalize goal to match predicted trailing dims.
        if goal_emb.ndim == pred_emb.ndim - 1:
            goal_emb = goal_emb.unsqueeze(-2)
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))
        return cost

    def _encode_goal(self, info_dict: dict) -> torch.Tensor:
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in list(goal.keys()):
            if k.startswith("goal_") and k != "goal_emb":
                goal[k[len("goal_") :]] = goal.pop(k)
        for drop in (
            "action",
            "goal_emb",
            "imagine_steps",
            "predicted_emb",
            "emb",
            "plan_branch",
        ):
            goal.pop(drop, None)
        goal = self.encode(goal)
        return goal["emb"]

    def _latent_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE over latent dim. pred/target: (B, S, D) -> cost (B, S)."""
        if target.ndim == pred.ndim - 1:
            target = target.unsqueeze(1).expand_as(pred)
        if target.shape != pred.shape:
            target = target.expand_as(pred)
        return (pred - target.detach()).pow(2).sum(dim=-1)

    def _as_bs_latent(self, goal_emb: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
        """Normalize goal embedding to (B, S, D) matching predictor endpoint."""
        z = goal_emb
        if z.ndim == endpoint.ndim + 1:
            z = z[..., -1, :]
        if z.ndim == endpoint.ndim - 1:
            z = z.unsqueeze(1).expand_as(endpoint)
        if z.shape != endpoint.shape:
            z = z.expand_as(endpoint)
        return z

    def _current_now_latent(
        self, info_dict: dict, endpoint: torch.Tensor
    ) -> torch.Tensor:
        """Last encoded observation (not predictor rollout) as frozen z_now."""
        emb = info_dict.get("emb")
        if torch.is_tensor(emb):
            z_now = emb[..., -1, :]
            if z_now.ndim == endpoint.ndim - 1:
                z_now = z_now.unsqueeze(1).expand_as(endpoint)
            elif z_now.shape != endpoint.shape:
                z_now = z_now.expand_as(endpoint)
            return z_now
        return endpoint.detach()

    def _forward_cost(
        self, endpoint: torch.Tensor, z_goal: torch.Tensor, steps
    ) -> torch.Tensor:
        if steps is None:
            imagined = endpoint
        else:
            imagined = self.imagine_forward(endpoint, steps)
        return self._latent_mse(imagined, z_goal)

    def _backward_cost(
        self,
        endpoint: torch.Tensor,
        z_goal: torch.Tensor,
        steps,
        z_now: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if steps is None:
            z_sub = z_goal
        else:
            z_sub = self.imagine_backward(z_goal, steps, z_now=z_now)
        return self._latent_mse(endpoint, z_sub)

    def _meet_cost(
        self,
        endpoint: torch.Tensor,
        z_goal: torch.Tensor,
        steps,
        z_now: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Meet-in-the-middle: MSE(F^{k_f}(P), B^{k_b}(z_goal)), k_f+k_b=k."""
        if steps is None:
            return self._latent_mse(endpoint, z_goal)

        if not torch.is_tensor(steps):
            k_f, k_b = split_meet_steps(steps)
            z_f = self.imagine_forward(endpoint, k_f)
            z_b = self.imagine_backward(z_goal, k_b, z_now=z_now)
            return self._latent_mse(z_f, z_b)

        steps = steps.to(device=endpoint.device, dtype=torch.long)
        while steps.ndim < endpoint.ndim - 1:
            steps = steps.unsqueeze(-1)
        if steps.ndim == endpoint.ndim:
            steps = steps.squeeze(-1)
        if steps.shape != endpoint.shape[:-1]:
            steps = steps.expand(endpoint.shape[:-1])

        # Per-sample split; vectorize via masked recursion with k_f / k_b tensors.
        k_f = torch.div(steps, 2, rounding_mode="floor")
        k_b = steps - k_f
        z_f = self.imagine_forward(endpoint, k_f)
        z_b = self.imagine_backward(z_goal, k_b, z_now=z_now)
        return self._latent_mse(z_f, z_b)

    def _combine_fusion(
        self, mode: str, c_f: torch.Tensor, c_b: torch.Tensor
    ) -> torch.Tensor:
        if mode == "fusion_avg05" or mode == "fusion_avg07":
            a = self._fusion_alpha
            return a * c_f + (1.0 - a) * c_b
        if mode == "fusion_max":
            return torch.maximum(c_f, c_b)
        if mode == "fusion_min":
            return torch.minimum(c_f, c_b)
        if mode == "switch_offset":
            if self._goal_offset >= self._switch_offset_cutoff:
                return c_f
            a = self._fusion_alpha
            return a * c_f + (1.0 - a) * c_b
        raise ValueError(f"not a fusion combine mode: {mode}")

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Cost of action candidates. Returns (B, S)."""
        assert "goal" in info_dict or "goal_emb" in info_dict, "goal/goal_emb missing"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        mode = self._planning_mode
        imagine_steps = info_dict.get("imagine_steps", None)

        if "goal_emb" not in info_dict:
            info_dict["goal_emb"] = self._encode_goal(info_dict)

        info_dict = self.rollout(info_dict, action_candidates)
        endpoint = info_dict["predicted_emb"][..., -1, :]  # (B, S, D)
        z_goal = self._as_bs_latent(info_dict["goal_emb"], endpoint)

        if mode == "forward":
            return self._forward_cost(endpoint, z_goal, imagine_steps)

        if mode == "official":
            return self.criterion(info_dict)

        # Remaining modes may use B; z_now is the frozen current observation.
        z_now = self._current_now_latent(info_dict, endpoint)

        if mode == "meet":
            return self._meet_cost(endpoint, z_goal, imagine_steps, z_now=z_now)

        if mode == "fusion_ofb":
            # Present (official) + future (forward) + rolled-back subgoal (backward).
            c_o = self._latent_mse(endpoint, z_goal)
            c_f = self._forward_cost(endpoint, z_goal, imagine_steps)
            c_b = self._backward_cost(endpoint, z_goal, imagine_steps, z_now=z_now)
            return (c_o + c_f + c_b) / 3.0

        if mode in (
            "fusion_avg05",
            "fusion_avg07",
            "fusion_max",
            "fusion_min",
            "switch_offset",
        ):
            c_f = self._forward_cost(endpoint, z_goal, imagine_steps)
            c_b = self._backward_cost(endpoint, z_goal, imagine_steps, z_now=z_now)
            return self._combine_fusion(mode, c_f, c_b)

        if mode == "switch_remain":
            c_f = self._forward_cost(endpoint, z_goal, imagine_steps)
            c_b = self._backward_cost(endpoint, z_goal, imagine_steps, z_now=z_now)
            branch = info_dict.get("plan_branch", None)
            if branch is None:
                # Fallback: treat as pure forward if branch missing.
                return c_f
            branch = branch.to(device=c_f.device)
            while branch.ndim < c_f.ndim:
                branch = branch.unsqueeze(-1)
            if branch.ndim > c_f.ndim:
                branch = branch.squeeze(-1)
            if branch.shape != c_f.shape:
                branch = branch.expand_as(c_f)
            # 0 → forward, 1 → backward
            return torch.where(branch > 0, c_b, c_f)

        # backward: terminal predicted vs injected B(z_now, z_goal) subgoal
        return self.criterion(info_dict)
