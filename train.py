"""Train FBLeWM: official LeWM + Forward/Backward imaginers in one run."""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module import SIGReg
from utils import SaveCkptCallback, count_params, get_column_normalizer, get_img_preprocessor

FORWARD_VARIANTS = frozenset(
    {"latent", "action_aligned", "sequential_action", "branch_preserving"}
)


def _forward_variant(cfg) -> str:
    variant = str(cfg.loss.forward.get("variant", "latent"))
    if variant not in FORWARD_VARIANTS:
        raise ValueError(
            "loss.forward.variant must be 'latent', 'action_aligned', "
            f"'sequential_action', or 'branch_preserving', got {variant!r}"
        )
    return variant


def _prepare_forward_action(action, action_dim: int, variant: str) -> torch.Tensor:
    """Detach and flatten blocked actions; require T>=3 and last dim=A."""
    if action is None:
        raise ValueError(f"{variant} forward loss requires an action tensor")
    action = action.detach().reshape(action.size(0), action.size(1), -1)
    if action.size(1) < 3:
        raise ValueError(
            f"{variant} forward loss needs at least 3 action blocks, "
            f"got T={int(action.size(1))}"
        )
    if action.size(-1) != action_dim:
        raise ValueError(
            f"action last dim must equal imaginer.action_dim={action_dim}, "
            f"got {int(action.size(-1))}"
        )
    return action


def _fill_sequential_action_forward_losses(
    self, output, z, p, action, cfg, stage=None
) -> None:
    """A=G(p); z'=H(p,A). Teacher uses A_tgt; autonomous uses G."""
    imaginer = self.model.forward_imaginer
    if not bool(getattr(imaginer, "is_sequential_action", False)):
        raise ValueError(
            "loss.forward.variant=sequential_action requires "
            "SequentialActionCausalLatentImaginer"
        )
    for name in ("predict_action", "transition", "forward_with_action"):
        if not hasattr(imaginer, name):
            raise ValueError(
                "loss.forward.variant=sequential_action requires "
                f"{name} on SequentialActionCausalLatentImaginer"
            )

    action_dim = int(getattr(imaginer, "action_dim", -1))
    action = _prepare_forward_action(action, action_dim, "sequential_action")
    if action.size(0) != z.size(0):
        raise ValueError(
            f"action batch {int(action.size(0))} != latent batch {int(z.size(0))}"
        )

    x = p[:, 0:2]
    a_tgt = action[:, 1:3]
    z_tgt = z[:, 2:4]
    if a_tgt.shape[:-1] != x.shape[:-1]:
        raise ValueError(
            f"action leading shape {tuple(a_tgt.shape[:-1])} "
            f"incompatible with latent {tuple(x.shape[:-1])}"
        )

    a_step = imaginer.predict_action(x)
    z_teacher = imaginer.forward_teacher_forced(x, a_tgt)
    z_auto = imaginer.transition(x, a_step)
    a_roll_1, z_roll_1 = imaginer.forward_with_action(p[:, 0:1])
    a_roll_2, z_roll_2 = imaginer.forward_with_action(z_roll_1)

    teacher_weight = float(cfg.loss.forward.get("teacher_weight", 1.0))
    action_weight = float(cfg.loss.forward.get("action_weight", 1.0))
    roll_weight = float(cfg.loss.forward.roll_weight)

    output["forward_teacher_loss"] = (z_teacher - z_tgt).pow(2).mean()
    output["forward_auto_step_loss"] = (z_auto - z_tgt).pow(2).mean()
    output["forward_roll_loss"] = (z_roll_2 - z[:, 3:4]).pow(2).mean()
    output["forward_action_step_loss"] = (a_step - a_tgt).pow(2).mean()
    output["forward_action_roll_loss"] = (a_roll_2 - action[:, 2:3]).pow(2).mean()
    output["forward_teacher_auto_gap"] = (
        output["forward_auto_step_loss"] - output["forward_teacher_loss"]
    )
    latent_loss = (
        teacher_weight * output["forward_teacher_loss"]
        + output["forward_auto_step_loss"]
        + roll_weight * output["forward_roll_loss"]
    )
    action_loss = (
        output["forward_action_step_loss"]
        + roll_weight * output["forward_action_roll_loss"]
    )
    output["forward_action_loss"] = action_loss
    output["forward_loss"] = latent_loss + action_weight * action_loss
    output["forward_out_norm"] = z_auto.detach().float().norm(dim=-1).mean()
    output["forward_tgt_norm"] = z_tgt.detach().float().norm(dim=-1).mean()
    output["forward_action_pred_norm"] = a_step.detach().float().norm(dim=-1).mean()
    output["forward_action_tgt_norm"] = a_tgt.detach().float().norm(dim=-1).mean()

    if stage in ("val", "validate"):
        mean_act = getattr(self.model, "train_action_mean", None)
        if mean_act is not None:
            mean_act = mean_act.to(device=a_tgt.device, dtype=a_tgt.dtype).reshape(
                1, 1, -1
            )
            if int(mean_act.size(-1)) != int(a_tgt.size(-1)):
                raise ValueError(
                    "train_action_mean last dim must equal action_dim="
                    f"{int(a_tgt.size(-1))}, got {int(mean_act.size(-1))}"
                )
            baseline = (mean_act - a_tgt).pow(2).mean()
            model_mse = output["forward_action_step_loss"].detach()
            output["forward_action_mean_baseline_mse"] = baseline
            output["forward_action_skill"] = 1.0 - model_mse / baseline.clamp(min=1e-8)


def _fill_branch_preserving_forward_losses(
    self, output, z, p, cfg, stage=None
) -> None:
    """WTA step/roll on two-latent histories. Action is unused."""
    imaginer = self.model.forward_imaginer
    if not bool(getattr(imaginer, "is_branch_preserving", False)):
        raise ValueError(
            "loss.forward.variant=branch_preserving requires "
            "BranchPreservingCausalLatentImaginer"
        )
    if int(getattr(imaginer, "history_size", -1)) != 2:
        raise ValueError(
            "branch_preserving imaginer history_size must be 2, "
            f"got {getattr(imaginer, 'history_size', None)!r}"
        )
    if z.size(1) < 4:
        raise ValueError(
            f"forward loss needs at least 4 latent frames, got T={int(z.size(1))}"
        )
    if p.size(1) < 2:
        raise ValueError(
            f"forward loss needs at least 2 predictor frames, got T={int(p.size(1))}"
        )

    z0, z2, z3 = z[:, 0], z[:, 2], z[:, 3]
    p0, p1 = p[:, 0], p[:, 1]
    h0 = torch.stack([z0, p0], dim=1)
    h1 = torch.stack([p0, p1], dim=1)
    y2 = imaginer.forward_branches(h0)
    y3_step = imaginer.forward_branches(h1)
    bsz, num_branches, dim = y2.shape
    p0_m = p0.unsqueeze(1).expand(bsz, num_branches, dim)
    h_roll = torch.stack([p0_m, y2], dim=2)
    y3_roll = imaginer.forward_assigned(h_roll)

    z2_t = z2.unsqueeze(1).to(dtype=torch.float32)
    z3_t = z3.unsqueeze(1).to(dtype=torch.float32)
    e2 = (y2.float() - z2_t).pow(2).mean(dim=-1)
    e3_step = (y3_step.float() - z3_t).pow(2).mean(dim=-1)
    e3_roll = (y3_roll.float() - z3_t).pow(2).mean(dim=-1)
    e_step = 0.5 * (e2 + e3_step)
    roll_weight = float(cfg.loss.forward.roll_weight)
    e_total = e_step + roll_weight * e3_roll
    winner = e_total.detach().argmin(dim=1)
    gather_idx = winner.unsqueeze(1)
    step_loss = e_step.gather(1, gather_idx).squeeze(1).mean()
    roll_loss = e3_roll.gather(1, gather_idx).squeeze(1).mean()
    output["forward_step_loss"] = step_loss
    output["forward_roll_loss"] = roll_loss
    output["forward_loss"] = step_loss + roll_weight * roll_loss

    usage = torch.zeros(num_branches, device=z.device, dtype=torch.float32)
    usage.scatter_add_(
        0, winner, torch.ones(bsz, device=z.device, dtype=torch.float32)
    )
    usage = usage / max(bsz, 1)
    log_usage = usage.clamp_min(1e-8).log()
    entropy = -(usage * log_usage).sum()
    output["forward_branch_usage_entropy"] = entropy.detach()
    output["forward_branch_effective_count"] = entropy.exp().detach()
    output["forward_branch_active_fraction"] = (usage > 0.01).float().mean().detach()
    output["forward_branch_winner_max_frac"] = usage.max().detach()
    if num_branches > 1:
        pair = y2.unsqueeze(2) - y2.unsqueeze(1)
        dist = pair.float().norm(dim=-1)
        eye = torch.eye(num_branches, dtype=torch.bool, device=y2.device)
        output["forward_branch_spread"] = dist[:, ~eye].mean().detach()
    else:
        output["forward_branch_spread"] = y2.new_zeros(())
    output["forward_out_norm"] = y2.detach().float().norm(dim=-1).mean()
    output["forward_tgt_norm"] = z2.detach().float().norm(dim=-1).mean()


def _fill_forward_losses(self, output, z, p, action, cfg, stage=None) -> None:
    """Detached Forward step/roll losses. Action is a target, never an input."""
    alpha_f = float(cfg.loss.forward.roll_weight)
    variant = _forward_variant(cfg)
    imaginer = self.model.forward_imaginer

    if z.size(1) < 4:
        raise ValueError(
            f"forward loss needs at least 4 latent frames, got T={int(z.size(1))}"
        )
    if p.size(1) < 2:
        raise ValueError(
            f"forward loss needs at least 2 predictor frames, got T={int(p.size(1))}"
        )

    f_in = p[:, 0:2]
    f_tgt = z[:, 2:4]

    if variant == "latent":
        f_pred = imaginer(f_in)
        f_roll = imaginer(imaginer(p[:, 0:1]))
        output["forward_step_loss"] = (f_pred - f_tgt).pow(2).mean()
        output["forward_roll_loss"] = (f_roll - z[:, 3:4]).pow(2).mean()
        output["forward_loss"] = (
            output["forward_step_loss"] + alpha_f * output["forward_roll_loss"]
        )
        output["forward_out_norm"] = f_pred.detach().float().norm(dim=-1).mean()
        output["forward_tgt_norm"] = f_tgt.detach().float().norm(dim=-1).mean()
        return

    if variant == "sequential_action":
        _fill_sequential_action_forward_losses(
            self, output, z, p, action, cfg, stage=stage
        )
        return

    if variant == "branch_preserving":
        _fill_branch_preserving_forward_losses(
            self, output, z, p, cfg, stage=stage
        )
        return

    if not hasattr(imaginer, "forward_with_action"):
        raise ValueError(
            "loss.forward.variant=action_aligned requires "
            "ActionAlignedCausalLatentImaginer with forward_with_action"
        )
    action_dim = int(getattr(imaginer, "action_dim", -1))
    action = _prepare_forward_action(action, action_dim, "action_aligned")

    a_step, z_step = imaginer.forward_with_action(f_in)
    _, z_roll_1 = imaginer.forward_with_action(p[:, 0:1])
    a_roll_2, z_roll_2 = imaginer.forward_with_action(z_roll_1)
    a_tgt = action[:, 1:3]
    a_roll_tgt = action[:, 2:3]

    output["forward_step_loss"] = (z_step - f_tgt).pow(2).mean()
    output["forward_roll_loss"] = (z_roll_2 - z[:, 3:4]).pow(2).mean()
    output["forward_action_step_loss"] = (a_step - a_tgt).pow(2).mean()
    output["forward_action_roll_loss"] = (a_roll_2 - a_roll_tgt).pow(2).mean()
    latent_loss = output["forward_step_loss"] + alpha_f * output["forward_roll_loss"]
    action_loss = (
        output["forward_action_step_loss"]
        + alpha_f * output["forward_action_roll_loss"]
    )
    action_weight = float(cfg.loss.forward.get("action_weight", 1.0))
    output["forward_action_loss"] = action_loss
    output["forward_loss"] = latent_loss + action_weight * action_loss
    output["forward_out_norm"] = z_step.detach().float().norm(dim=-1).mean()
    output["forward_tgt_norm"] = f_tgt.detach().float().norm(dim=-1).mean()
    output["forward_action_pred_norm"] = a_step.detach().float().norm(dim=-1).mean()
    output["forward_action_tgt_norm"] = a_tgt.detach().float().norm(dim=-1).mean()


def _compute_train_blocked_action_mean(
    train_set, action_dim: int, max_items: int = 256
) -> torch.Tensor:
    """Fixed mean of standardized blocked actions on the training split only."""
    acc = torch.zeros(int(action_dim), dtype=torch.float32)
    n = 0
    n_take = min(len(train_set), int(max_items))
    if n_take <= 0:
        return acc
    step = max(len(train_set) // n_take, 1)
    for i in range(0, len(train_set), step):
        if n >= n_take:
            break
        try:
            item = train_set[i]
        except Exception:
            continue
        act = item.get("action") if isinstance(item, dict) else None
        if act is None:
            continue
        t = torch.as_tensor(act, dtype=torch.float32)
        t = torch.nan_to_num(t, 0.0).reshape(-1, t.shape[-1] if t.ndim >= 1 else 1)
        if t.size(-1) != int(action_dim):
            try:
                t = t.reshape(-1, int(action_dim))
            except RuntimeError:
                continue
        acc = acc + t.mean(0)
        n += 1
    if n == 0:
        return acc
    return acc / n


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite value in {name}: {tensor}")


def _shuffle_batch(x: torch.Tensor) -> torch.Tensor:
    """Permute the batch axis; identity if batch size is 1."""
    if x.size(0) <= 1:
        return x
    perm = torch.randperm(x.size(0), device=x.device)
    return x[perm]


def _fill_pred_goal_backward(self, output, z, p, cfg) -> None:
    """B(P2, Z3)->Z2 and B(P1, Z2)->Z1, with identity / shuffle diagnostics."""
    p1 = p[:, 0:1]
    p2 = p[:, 1:2]
    z1 = z[:, 1:2]
    z2 = z[:, 2:3]
    z3 = z[:, 3:4]
    noise_std = float(cfg.loss.backward.get("p_noise", 0.1))
    margin = float(cfg.loss.backward.get("goal_margin", 0.1))

    if noise_std > 0:
        p1_in = p1 + noise_std * torch.randn_like(p1)
        p2_in = p2 + noise_std * torch.randn_like(p2)
    else:
        p1_in = p1
        p2_in = p2

    g2 = self.model.backward_imaginer(p2_in, z3)
    g1 = self.model.backward_imaginer(p1_in, z2)
    output["backward_step_loss"] = 0.5 * (
        (g2 - z2).pow(2).mean() + (g1 - z1).pow(2).mean()
    )
    g1_roll = self.model.backward_imaginer(p1_in, g2)
    output["backward_roll_loss"] = (g1_roll - z1).pow(2).mean()

    z3_shuf = _shuffle_batch(z3)
    z2_shuf = _shuffle_batch(z2)
    d_pos = (g2 - z2).pow(2).mean()
    d_neg = (self.model.backward_imaginer(p2_in, z3_shuf) - z2).pow(2).mean()
    d_pos_1 = (g1 - z1).pow(2).mean()
    d_neg_1 = (self.model.backward_imaginer(p1_in, z2_shuf) - z1).pow(2).mean()
    output["backward_goal_rank_loss"] = 0.5 * (
        torch.relu(margin + d_pos - d_neg) + torch.relu(margin + d_pos_1 - d_neg_1)
    )

    with torch.no_grad():
        g2_clean = self.model.backward_imaginer(p2, z3)
        g2_shuf = self.model.backward_imaginer(p2, z3_shuf)
        output["backward_pred_mse"] = (p2 - z2).pow(2).mean()
        output["backward_clean_mse"] = (g2_clean - z2).pow(2).mean()
        output["backward_shuffle_mse"] = (g2_shuf - z2).pow(2).mean()
        output["backward_identity_gap"] = (
            output["backward_pred_mse"] - output["backward_clean_mse"]
        )

    output["_b_pred"] = g2
    output["_b_tgt"] = z2


def _fill_fixed_bridge_backward(self, output, z, p) -> None:
    """B(P1, z3)->z2, B(P1, z2)->z1, B(P1, B(P1, z3))->z1. No noise/rank."""
    p1 = p[:, 0:1]
    z1 = z[:, 1:2]
    z2 = z[:, 2:3]
    z3 = z[:, 3:4]

    g2 = self.model.backward_imaginer(p1, z3)
    g1 = self.model.backward_imaginer(p1, z2)
    output["backward_step_loss"] = 0.5 * (
        (g2 - z2).pow(2).mean() + (g1 - z1).pow(2).mean()
    )

    g1_roll = self.model.backward_imaginer(p1, g2)
    output["backward_roll_loss"] = (g1_roll - z1).pow(2).mean()

    with torch.no_grad():
        output["backward_copy_mse"] = (p1 - z2).pow(2).mean()
        output["backward_clean_mse"] = (g2 - z2).pow(2).mean()
        output["backward_copy_gap"] = (
            output["backward_copy_mse"] - output["backward_clean_mse"]
        )

    output["_b_pred"] = g2
    output["_b_tgt"] = z2


def fblewm_forward(self, batch, stage, cfg):
    """Official JEPA loss + detached Forward/Backward imaginer losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    lam_f = cfg.loss.forward.weight
    lam_b = cfg.loss.backward.weight
    alpha_b = cfg.loss.backward.roll_weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # Official LeWM loss (no stop-grad on targets).
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["official_loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Detached latents / actions for imaginers. Alignment:
    # p[:,i] ~ z[:,i+1]; F(p[:,0]) -> (action[:,1], z[:,2]) when action_aligned.
    # sequential_action: A=G(p); z'=H(p,A); teacher uses action[:,1:3].
    z = emb.detach()
    p = pred_emb.detach()
    _fill_forward_losses(self, output, z, p, batch.get("action"), cfg, stage=stage)

    # Backward is selected by loss.backward.target:
    #   pred         — unary B(z_{t+1}) -> p_t
    #   encoder      — unary B(z_{t+1}) -> z_t
    #   now          — conditional g <- B(z0, g) in z-space
    #   pred_goal    — conditional G <- B(P, z_later), supervise G ≈ z_at_P
    #   fixed_bridge — conditional B(P1, z_later) with frozen P1
    b_target_kind = str(cfg.loss.backward.get("target", "pred"))
    is_cond = bool(getattr(self.model.backward_imaginer, "is_conditional", False))
    if b_target_kind == "now":
        if not is_cond:
            raise ValueError(
                "loss.backward.target=now requires ConditionalLatentImaginer "
                "(set model.backward_imaginer._target_=module.ConditionalLatentImaginer)"
            )
        z0 = z[:, 0:1]
        z1 = z[:, 1:2]
        z2 = z[:, 2:3]
        z3 = z[:, 3:4]
        b_from_z3 = self.model.backward_imaginer(z0, z3)
        b_from_z2 = self.model.backward_imaginer(z0, z2)
        output["backward_step_loss"] = (
            (b_from_z3 - z2).pow(2).mean() + (b_from_z2 - z1).pow(2).mean()
        ) * 0.5
        b_roll = self.model.backward_imaginer(z0, b_from_z3)
        output["backward_roll_loss"] = (b_roll - z1).pow(2).mean()
        b_pred = b_from_z3
        b_tgt = z2
    elif b_target_kind == "pred_goal":
        if not is_cond:
            raise ValueError(
                "loss.backward.target=pred_goal requires ConditionalLatentImaginer "
                "(set model.backward_imaginer._target_=module.ConditionalLatentImaginer)"
            )
        _fill_pred_goal_backward(self, output, z, p, cfg)
        b_pred = output.pop("_b_pred")
        b_tgt = output.pop("_b_tgt")
    elif b_target_kind == "fixed_bridge":
        if not is_cond:
            raise ValueError(
                "loss.backward.target=fixed_bridge requires ConditionalLatentImaginer "
                "(set model.backward_imaginer._target_=module.ConditionalLatentImaginer)"
            )
        _fill_fixed_bridge_backward(self, output, z, p)
        b_pred = output.pop("_b_pred")
        b_tgt = output.pop("_b_tgt")
    elif b_target_kind in ("pred", "encoder"):
        if is_cond:
            raise ValueError(
                f"loss.backward.target={b_target_kind!r} is the unary B objective; "
                "use CausalLatentImaginer, or set target=now / pred_goal / fixed_bridge"
            )
        b_in = torch.stack([z[:, 3], z[:, 2]], dim=1)
        if b_target_kind == "pred":
            b_tgt = torch.stack([p[:, 1], p[:, 0]], dim=1)  # p2, p1
            b_roll_tgt = p[:, 0:1]  # p1
        else:
            b_tgt = torch.stack([z[:, 2], z[:, 1]], dim=1)
            b_roll_tgt = z[:, 1:2]
        b_pred = self.model.backward_imaginer(b_in)
        output["backward_step_loss"] = (b_pred - b_tgt).pow(2).mean()
        b_roll = self.model.backward_imaginer(
            self.model.backward_imaginer(z[:, 3:4])
        )
        output["backward_roll_loss"] = (b_roll - b_roll_tgt).pow(2).mean()
    else:
        raise ValueError(
            "loss.backward.target must be 'pred', 'encoder', 'now', "
            "'pred_goal', or 'fixed_bridge', "
            f"got {b_target_kind!r}"
        )

    output["backward_loss"] = (
        output["backward_step_loss"] + alpha_b * output["backward_roll_loss"]
    )
    if "backward_goal_rank_loss" in output:
        rank_w = float(cfg.loss.backward.get("goal_rank_weight", 0.5))
        output["backward_loss"] = (
            output["backward_loss"] + rank_w * output["backward_goal_rank_loss"]
        )

    output["loss"] = (
        output["official_loss"]
        + lam_f * output["forward_loss"]
        + lam_b * output["backward_loss"]
    )

    # Diagnostics (detached). Forward norms are filled in _fill_forward_losses.
    output["backward_out_norm"] = b_pred.detach().float().norm(dim=-1).mean()
    output["backward_tgt_norm"] = b_tgt.detach().float().norm(dim=-1).mean()

    for key in (
        "pred_loss",
        "sigreg_loss",
        "official_loss",
        "forward_step_loss",
        "forward_roll_loss",
        "forward_teacher_loss",
        "forward_auto_step_loss",
        "forward_teacher_auto_gap",
        "forward_action_step_loss",
        "forward_action_roll_loss",
        "forward_action_loss",
        "forward_action_mean_baseline_mse",
        "forward_action_skill",
        "forward_branch_usage_entropy",
        "forward_branch_effective_count",
        "forward_branch_active_fraction",
        "forward_branch_spread",
        "forward_branch_winner_max_frac",
        "forward_loss",
        "backward_step_loss",
        "backward_roll_loss",
        "backward_goal_rank_loss",
        "backward_loss",
        "loss",
    ):
        if key not in output:
            continue
        _require_finite(key, output[key])

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if (
            "loss" in k
            or k.endswith("_norm")
            or k.endswith("_mse")
            or k.endswith("_gap")
            or k.endswith("_skill")
            or k.startswith("forward_branch_")
        )
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


# Legacy joint-train artifacts; never write new runs into these names.
_PROTECTED_OUTPUT_NAMES = frozenset(
    {"fblewm", "fblewm_bp", "fblewm_tworoom", "fblewm_cube"}
)


@hydra.main(version_base=None, config_path="./config/train", config_name="fblewm")
def run(cfg):
    out_name = str(cfg.output_model_name)
    if out_name in _PROTECTED_OUTPUT_NAMES:
        raise ValueError(
            f"Refusing to train into protected output_model_name={out_name!r} "
            "(would overwrite an existing checkpoint tree). "
            "Use a new name, e.g. output_model_name=fblewm_tworoom_v2."
        )
    b_target = str(cfg.loss.backward.get("target", "pred"))
    print(
        f"[FBLeWM] train run output_model_name={out_name} "
        f"backward.target={b_target} "
        f"(checkpoints -> $STABLEWM_HOME/checkpoints/{out_name}/)",
        flush=True,
    )

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    resolve_name = dataset_name
    if cache_dir:
        direct = Path(cache_dir) / dataset_name
        under_datasets = Path(cache_dir) / "datasets" / dataset_name
        if direct.exists():
            resolve_name = str(direct.resolve())
        elif under_datasets.exists():
            resolve_name = str(under_datasets.resolve())

    dataset = swm.data.load_dataset(
        resolve_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

    blocked_action_dim = int(cfg.model.action_encoder.input_dim)
    f_variant = _forward_variant(cfg)
    if f_variant == "action_aligned":
        with open_dict(cfg):
            cfg.model.forward_imaginer._target_ = (
                "module.ActionAlignedCausalLatentImaginer"
            )
            cfg.model.forward_imaginer.action_dim = blocked_action_dim
    elif f_variant == "sequential_action":
        with open_dict(cfg):
            cfg.model.forward_imaginer._target_ = (
                "module.SequentialActionCausalLatentImaginer"
            )
            cfg.model.forward_imaginer.action_dim = blocked_action_dim
    elif f_variant == "branch_preserving":
        with open_dict(cfg):
            cfg.model.forward_imaginer._target_ = (
                "module.BranchPreservingCausalLatentImaginer"
            )
            cfg.model.forward_imaginer.num_branches = int(
                cfg.loss.forward.get("branches", 4)
            )

    print(
        f"[FBLeWM] forward.variant={f_variant} "
        f"forward.action_weight={float(cfg.loss.forward.get('action_weight', 1.0))} "
        f"forward.roll_weight={float(cfg.loss.forward.roll_weight)} "
        f"forward.teacher_weight={float(cfg.loss.forward.get('teacher_weight', 1.0))} "
        f"forward.branches={int(cfg.loss.forward.get('branches', 4))} "
        f"blocked_action_dim={blocked_action_dim} "
        f"backward.weight={float(cfg.loss.backward.weight)}",
        flush=True,
    )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    world_model = hydra.utils.instantiate(cfg.model)
    if f_variant == "sequential_action":
        train_action_mean = _compute_train_blocked_action_mean(
            train_set, blocked_action_dim
        )
        world_model.register_buffer(
            "train_action_mean", train_action_mean, persistent=False
        )
        print(
            f"[FBLeWM] train_action_mean dim={blocked_action_dim} "
            f"norm={float(train_action_mean.norm()):.4f} "
            "(training-split blocked actions; val baseline only)",
            flush=True,
        )

    # Parameter counts for metadata.
    n_official = (
        count_params(world_model.encoder)
        + count_params(world_model.predictor)
        + count_params(world_model.action_encoder)
        + count_params(world_model.projector)
        + count_params(world_model.pred_proj)
    )
    n_f = count_params(world_model.forward_imaginer)
    n_b = count_params(world_model.backward_imaginer)
    n_g = n_h = None
    n_f_shared = n_f_heads = None
    fwd = world_model.forward_imaginer
    if bool(getattr(fwd, "is_sequential_action", False)):
        n_g = count_params(fwd.blocks) + count_params(fwd.action_head)
        n_h = (
            count_params(fwd.action_embed)
            + count_params(fwd.fuse)
            + count_params(fwd.transition_blocks)
            + count_params(fwd.out_norm)
            + count_params(fwd.out_proj)
        )
    if bool(getattr(fwd, "is_branch_preserving", False)):
        n_f_shared = (
            count_params(fwd.history_fuse)
            + count_params(fwd.blocks)
            + count_params(fwd.out_norm)
        )
        n_f_heads = count_params(fwd.branch_heads)
    extra_fwd = ""
    if n_g is not None:
        extra_fwd += f"forward_proposer={n_g} forward_transition={n_h} "
    if n_f_shared is not None:
        extra_fwd += (
            f"forward_shared={n_f_shared} forward_branch_heads={n_f_heads} "
        )
    print(
        f"[FBLeWM] params official={n_official} forward={n_f} backward={n_b} "
        + extra_fwd
        + f"total={n_official + n_f + n_b}",
        flush=True,
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(fblewm_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    with open(run_dir / "param_counts.json", "w") as f:
        import json

        counts = {
            "official": n_official,
            "forward_imaginer": n_f,
            "backward_imaginer": n_b,
            "total": n_official + n_f + n_b,
        }
        if n_g is not None:
            counts["forward_action_proposer"] = n_g
            counts["forward_transition"] = n_h
        if n_f_shared is not None:
            counts["forward_shared"] = n_f_shared
            counts["forward_branch_heads"] = n_f_heads
        json.dump(counts, f, indent=2)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
