"""FBLeWM: official LeWM + detached Forward/Backward Causal Latent Imaginers.

Forward imaginer may be unary (latent-only), parallel action-aligned
(``F(p) -> (A_hat, z_hat)``), sequential (``A=G(p); z'=H(p,A)``), or
branch-preserving (``F_m([z_{t-1}, z_t]) -> z_{t+1}``). Unary planning
still calls ``forward(z)``. Branch-preserving scoring uses
``imagine_forward_branches`` and best-of-M terminal cost. Predicted
actions are never executed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from planning import (
    ALL_PLANNING_MODES,
    FUSION_MODES,
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
        backward_anchor=None,
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
        self._backward_depth_cap = None
        self._forward_depth_override = None
        self.backward_anchor = self._normalize_backward_anchor(backward_anchor)
        # Runtime-only TRM attachment. Bypass nn.Module registration so loading,
        # saving, and state_dict keys of all existing checkpoints stay unchanged.
        object.__setattr__(self, "_trm_head", None)
        self._trm_weight = 1.0
        self._trm_eps = 1e-8
        self._trm_metadata = None

    def _effective_backward_anchor(self):
        if self.backward_anchor is not None:
            return self.backward_anchor
        if self._is_conditional_backward():
            return "obs"
        return None

    @staticmethod
    def _normalize_backward_anchor(anchor):
        if anchor is None or anchor == "" or anchor == "none":
            return None
        anchor = str(anchor)
        if anchor in ("obs", "now"):
            return "obs"
        if anchor in ("pred", "pred_goal"):
            return "pred"
        raise ValueError(
            f"backward_anchor must be none/obs/pred, got {anchor!r}"
        )

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

    def set_trm_head(
        self,
        head: nn.Module,
        *,
        weight: float = 1.0,
        eps: float = 1e-8,
        metadata: dict | None = None,
    ) -> None:
        """Attach an eval-only TRM head without changing the model state dict."""
        if not isinstance(head, nn.Module):
            raise TypeError(f"TRM head must be nn.Module, got {type(head)!r}")
        weight = float(weight)
        eps = float(eps)
        if not torch.isfinite(torch.tensor(weight)):
            raise ValueError(f"TRM weight must be finite, got {weight}")
        if eps <= 0:
            raise ValueError(f"TRM eps must be > 0, got {eps}")

        latent_dim = getattr(head, "latent_dim", getattr(head, "dim", None))
        expected_dim = getattr(self.forward_imaginer, "dim", None)
        if latent_dim is not None and expected_dim is not None:
            if int(latent_dim) != int(expected_dim):
                raise ValueError(
                    f"TRM latent dim {int(latent_dim)} != model dim {int(expected_dim)}"
                )

        device = next(self.parameters()).device
        head = head.to(device=device).eval()
        head.requires_grad_(False)
        object.__setattr__(self, "_trm_head", head)
        self._trm_weight = weight
        self._trm_eps = eps
        self._trm_metadata = dict(metadata or {})

    def clear_trm_head(self) -> None:
        """Remove the runtime-only TRM head."""
        object.__setattr__(self, "_trm_head", None)
        self._trm_metadata = None

    def set_switch_remain_threshold(self, threshold: int) -> None:
        self._switch_remain_threshold = int(threshold)

    def set_switch_offset_cutoff(self, cutoff: int) -> None:
        self._switch_offset_cutoff = int(cutoff)

    def set_forward_depth_override(self, steps) -> None:
        """Eval-only: replace Forward ``imagine_steps`` with a fixed ``k``.

        ``None`` keeps the dynamic schedule. Training does not call this.
        Official / Backward / fusion costs are unchanged.
        """
        if steps is None or steps == "" or steps == "none":
            self._forward_depth_override = None
            return
        steps = int(steps)
        if steps < 0:
            raise ValueError(f"forward_depth_override must be >= 0, got {steps}")
        self._forward_depth_override = steps

    @property
    def forward_depth_override(self):
        return self._forward_depth_override

    def _apply_forward_depth_override(self, steps):
        override = self._forward_depth_override
        if override is None:
            return steps
        override = int(override)
        if steps is None or not torch.is_tensor(steps):
            return override
        return torch.full_like(steps, override)

    def set_backward_depth_cap(self, cap) -> None:
        """Eval-only: cap pred_goal recursion to min(k, cap). None disables.

        k=0 is never changed. Training does not call this.
        """
        if cap is None or cap == "" or cap == "none":
            self._backward_depth_cap = None
            return
        cap = int(cap)
        if cap < 0:
            raise ValueError(f"backward_depth_cap must be >= 0, got {cap}")
        self._backward_depth_cap = cap

    @property
    def backward_depth_cap(self):
        return self._backward_depth_cap

    def _apply_pred_goal_depth_cap(self, steps):
        """Clamp pred_goal recursion depth. Leaves k=0 and non-pred anchors untouched."""
        cap = self._backward_depth_cap
        if cap is None or self._effective_backward_anchor() != "pred":
            return steps
        cap = int(cap)
        if not torch.is_tensor(steps):
            k = _as_int_steps(steps)
            if k == 0:
                return 0
            return min(k, cap)
        return steps.clamp(max=cap)

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

    def _is_branch_preserving_forward(self) -> bool:
        return bool(getattr(self.forward_imaginer, "is_branch_preserving", False))

    def _normalize_branch_steps(self, steps, leading_shape, device) -> torch.Tensor:
        """Broadcast scalar/tensor steps to ``leading_shape`` as int64 >= 0."""
        if not torch.is_tensor(steps):
            n = _as_int_steps(steps)
            return torch.full(leading_shape, n, device=device, dtype=torch.long)
        steps = steps.to(device=device, dtype=torch.long)
        target_ndim = len(leading_shape)
        while steps.ndim < target_ndim:
            steps = steps.unsqueeze(-1)
        if steps.ndim == target_ndim + 1:
            if steps.size(-1) != 1:
                raise ValueError(
                    "steps last dim must be 1 when matching history leading ndim, "
                    f"got {tuple(steps.shape)}"
                )
            steps = steps.squeeze(-1)
        if tuple(steps.shape) != tuple(leading_shape):
            steps = steps.expand(leading_shape)
        if steps.numel() and bool((steps < 0).any()):
            raise ValueError(
                f"imagine steps must be >= 0, got min={int(steps.min().item())}"
            )
        return steps.contiguous()

    def imagine_forward_branches(self, history: torch.Tensor, steps=1) -> torch.Tensor:
        """Branch-consistent recursion. ``history``: ``(..., 2, D)`` -> ``(..., M, D)``."""
        imaginer = self.forward_imaginer
        if not self._is_branch_preserving_forward():
            raise ValueError(
                "imagine_forward_branches requires BranchPreservingCausalLatentImaginer"
            )
        if history.size(-2) != getattr(imaginer, "history_size", 2):
            raise ValueError(
                "imagine_forward_branches expects history (..., 2, D), "
                f"got {tuple(history.shape)}"
            )
        leading = history.shape[:-2]
        num_branches = int(imaginer.num_branches)
        dim = history.size(-1)
        steps_t = self._normalize_branch_steps(steps, leading, history.device)
        max_k = int(steps_t.max().item()) if steps_t.numel() else 0
        branch_hist = (
            history.unsqueeze(-3).expand(*leading, num_branches, 2, dim).contiguous()
        )
        endpoint = (
            history[..., -1, :]
            .unsqueeze(-2)
            .expand(*leading, num_branches, dim)
            .contiguous()
        )
        if max_k == 0:
            return endpoint
        for step_index in range(max_k):
            nxt = imaginer.forward_assigned(branch_hist)
            active = steps_t > step_index
            mask_ep = active.reshape(*leading, 1, 1).expand_as(endpoint)
            endpoint = torch.where(mask_ep, nxt, endpoint)
            shifted = torch.cat([branch_hist[..., 1:, :], nxt.unsqueeze(-2)], dim=-2)
            mask_h = active.reshape(*leading, 1, 1, 1).expand_as(branch_hist)
            branch_hist = torch.where(mask_h, shifted, branch_hist)
        return endpoint

    def _is_conditional_backward(self) -> bool:
        return bool(getattr(self.backward_imaginer, "is_conditional", False))

    def _coarsen_backward_steps(self, steps, *, max_steps: int = 3, block: int = 5):
        """Cap/coarsen k only for obs-anchored now-B; pred_goal keeps the fine schedule."""
        if self._effective_backward_anchor() != "obs":
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
        """Roll goal backward. Conditional B: g <- B(anchor, g) with anchor frozen."""
        if not self._is_conditional_backward():
            if torch.is_tensor(steps) and steps.numel() != 1:
                return self._recurse_imaginer_masked(
                    self.backward_imaginer, z_goal, steps
                )
            return self._recurse_imaginer(self.backward_imaginer, z_goal, steps)

        steps = self._coarsen_backward_steps(steps)
        steps = self._apply_pred_goal_depth_cap(steps)
        if not torch.is_tensor(steps):
            if _as_int_steps(steps) == 0:
                return z_goal
        elif int(steps.max().item()) == 0:
            return z_goal

        if z_now is None:
            raise ValueError(
                "conditional backward imaginer requires an anchor latent "
                "(z_now for now-B, predicted endpoint for pred_goal); got z_now=None"
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

    @staticmethod
    def _candidate_zscore(cost: torch.Tensor, eps: float) -> torch.Tensor:
        """Standardize independently over each current CEM candidate pool."""
        if cost.ndim != 2:
            raise ValueError(f"candidate cost must have shape (B, S), got {cost.shape}")
        mean = cost.mean(dim=-1, keepdim=True)
        std = cost.std(dim=-1, keepdim=True, unbiased=False)
        return (cost - mean) / (std + float(eps))

    def _trm_terminal_cost(
        self, endpoint: torch.Tensor, z_goal: torch.Tensor
    ) -> torch.Tensor:
        head = self._trm_head
        if head is None:
            raise RuntimeError(
                f"planning_mode={self._planning_mode!r} requires set_trm_head(...)"
            )
        try:
            head_dtype = next(head.parameters()).dtype
        except StopIteration:
            head_dtype = endpoint.dtype
        cost = head(
            endpoint.to(dtype=head_dtype),
            z_goal.detach().to(dtype=head_dtype),
        )
        if cost.shape != endpoint.shape[:-1]:
            raise ValueError(
                f"TRM head returned shape {tuple(cost.shape)}, "
                f"expected {tuple(endpoint.shape[:-1])}"
            )
        return cost

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
        self, endpoint: torch.Tensor, z_goal: torch.Tensor, steps, *, history=None
    ) -> torch.Tensor:
        if not self._is_branch_preserving_forward():
            if steps is None:
                imagined = endpoint
            else:
                imagined = self.imagine_forward(endpoint, steps)
            return self._latent_mse(imagined, z_goal)
        if history is None:
            raise ValueError(
                "branch_preserving forward cost requires history (..., 2, D) "
                "from predicted_emb[..., -2:, :]"
            )
        official = self._latent_mse(endpoint, z_goal)
        if steps is None:
            return official
        branch_ep = self.imagine_forward_branches(history, steps)
        goal = z_goal
        if goal.ndim == branch_ep.ndim - 1:
            goal = goal.unsqueeze(-2)
        if goal.shape != branch_ep.shape:
            goal = goal.expand_as(branch_ep)
        dist_m = (branch_ep - goal.detach()).pow(2).sum(dim=-1)
        branch_cost = dist_m.min(dim=-1).values
        steps_t = self._normalize_branch_steps(
            steps, endpoint.shape[:-1], endpoint.device
        )
        return torch.where(steps_t == 0, official, branch_cost)

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
        predicted = info_dict["predicted_emb"]
        endpoint = predicted[..., -1, :]  # (B, S, D)
        z_goal = self._as_bs_latent(info_dict["goal_emb"], endpoint)
        history = None
        if self._is_branch_preserving_forward() and mode == "forward":
            if predicted.size(-2) < 2:
                raise ValueError(
                    "branch_preserving forward cost needs at least 2 predicted "
                    f"latents, got T={int(predicted.size(-2))}"
                )
            history = predicted[..., -2:, :]

        if mode in ("trm_replace", "trm_hybrid"):
            c_trm = self._trm_terminal_cost(endpoint, z_goal)
            if mode == "trm_replace":
                return c_trm
            c_lat = self._latent_mse(endpoint, z_goal)
            return self._candidate_zscore(
                c_lat, self._trm_eps
            ) + self._trm_weight * self._candidate_zscore(c_trm, self._trm_eps)

        if mode == "forward":
            return self._forward_cost(
                endpoint,
                z_goal,
                self._apply_forward_depth_override(imagine_steps),
                history=history,
            )

        if mode == "official":
            return self.criterion(info_dict)

        if mode in FUSION_MODES and self._is_branch_preserving_forward():
            raise ValueError(
                "branch_preserving does not support planning mode "
                f"{mode!r}; use official or forward (backward is also allowed)"
            )

        # now-B anchors on the current observation; pred_goal anchors on P.
        if self._effective_backward_anchor() == "pred":
            z_now = endpoint
        else:
            z_now = self._current_now_latent(info_dict, endpoint)

        if mode == "backward" and self._effective_backward_anchor() == "pred":
            return self._backward_cost(endpoint, z_goal, imagine_steps, z_now=endpoint)

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
