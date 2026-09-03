"""Contracts for Branch-Preserving Forward (two-latent history, WTA, best-of-M)."""

from __future__ import annotations

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from module import BranchPreservingCausalLatentImaginer, CausalLatentImaginer
from tests.test_gradient_isolation import _has_any_grad, _nonzero_grads
from tests.test_model_contracts import _tiny_branch_model, _tiny_model
from train import _fill_forward_losses, _forward_variant


class _Holder:
    def __init__(self, model):
        self.model = model


def _cfg(variant="branch_preserving", roll_weight=1.0, branches=4):
    return OmegaConf.create(
        {
            "loss": {
                "forward": {
                    "variant": variant,
                    "roll_weight": roll_weight,
                    "branches": branches,
                }
            }
        }
    )


def test_history_api_shapes():
    imag = BranchPreservingCausalLatentImaginer(
        dim=16, hidden_dim=32, depth=1, num_branches=3
    )
    h1 = torch.randn(4, 2, 16)
    y1 = imag.forward_branches(h1)
    assert y1.shape == (4, 3, 16)
    assert torch.allclose(imag(h1), y1)
    h2 = torch.randn(2, 5, 2, 16)
    y2 = imag.forward_branches(h2)
    assert y2.shape == (2, 5, 3, 16)
    assigned = imag.forward_assigned(h2.unsqueeze(2).expand(2, 5, 3, 2, 16).contiguous())
    assert assigned.shape == (2, 5, 3, 16)


def test_forward_default_is_all_branches():
    imag = BranchPreservingCausalLatentImaginer(
        dim=8, hidden_dim=16, depth=1, num_branches=2
    )
    h = torch.randn(3, 2, 8)
    assert torch.allclose(imag(h), imag.forward_branches(h))


def test_forward_assigned_uses_matching_head():
    imag = BranchPreservingCausalLatentImaginer(
        dim=8, hidden_dim=16, depth=1, num_branches=3
    )
    h_m = torch.randn(4, 3, 2, 8)
    y_all = imag.forward_assigned(h_m)
    for m in range(3):
        y_m = imag.forward_branches(h_m[:, m])[:, m]
        assert torch.allclose(y_all[:, m], y_m, atol=1e-6, rtol=1e-5)


def test_wrong_history_shape_raises():
    imag = BranchPreservingCausalLatentImaginer(
        dim=8, hidden_dim=16, depth=1, num_branches=2
    )
    try:
        imag.forward_branches(torch.randn(3, 8))
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        imag.forward_branches(torch.randn(3, 3, 8))
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        imag.forward_assigned(torch.randn(2, 3, 2, 8))
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        imag.forward_assigned(torch.randn(2, 2, 3, 8))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_illegal_num_branches_raises():
    try:
        BranchPreservingCausalLatentImaginer(dim=8, hidden_dim=16, depth=1, num_branches=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_different_branch_heads_differ():
    imag = BranchPreservingCausalLatentImaginer(
        dim=8, hidden_dim=16, depth=1, num_branches=3
    )
    h = torch.randn(5, 2, 8)
    y = imag.forward_branches(h)
    assert not torch.allclose(y[:, 0], y[:, 1])
    assert not torch.allclose(y[:, 1], y[:, 2])


def test_four_frame_history_alignment():
    B, D = 2, 8
    z = torch.arange(B * 4 * D, dtype=torch.float32).reshape(B, 4, D)
    p = torch.arange(B * 3 * D, dtype=torch.float32).reshape(B, 3, D) + 1000
    h0 = torch.stack([z[:, 0], p[:, 0]], dim=1)
    h1 = torch.stack([p[:, 0], p[:, 1]], dim=1)
    assert torch.equal(h0[:, 0], z[:, 0])
    assert torch.equal(h0[:, 1], p[:, 0])
    assert torch.equal(h1[:, 0], p[:, 0])
    assert torch.equal(h1[:, 1], p[:, 1])
    assert not torch.equal(h0, h1)


def test_branch_consistent_roll():
    imag = BranchPreservingCausalLatentImaginer(
        dim=8, hidden_dim=16, depth=1, num_branches=3
    )
    z = torch.randn(4, 4, 8)
    p = torch.randn(4, 3, 8)
    y2 = imag.forward_branches(torch.stack([z[:, 0], p[:, 0]], dim=1))
    p0_m = p[:, 0].unsqueeze(1).expand(4, 3, 8)
    h_roll = torch.stack([p0_m, y2], dim=2)
    y3 = imag.forward_assigned(h_roll)
    y3_switched = imag.forward_assigned(h_roll[:, [1, 2, 0]])
    assert y3.shape == (4, 3, 8)
    assert not torch.allclose(y3, y3_switched)


def test_fill_loss_keys_and_weight_formula():
    torch.manual_seed(0)
    m = _tiny_branch_model(dim=16, num_branches=3)
    holder = _Holder(m)
    z = torch.randn(5, 4, 16)
    p = torch.randn(5, 3, 16)
    cfg = _cfg(roll_weight=0.5)
    output = {}
    _fill_forward_losses(holder, output, z, p, None, cfg)
    assert "forward_step_loss" in output
    assert "forward_roll_loss" in output
    assert "forward_balance_loss" not in output
    assert "forward_action_step_loss" not in output
    expected = output["forward_step_loss"] + 0.5 * output["forward_roll_loss"]
    assert torch.allclose(output["forward_loss"], expected)
    for key in (
        "forward_branch_usage_entropy",
        "forward_branch_effective_count",
        "forward_branch_active_fraction",
        "forward_branch_spread",
        "forward_branch_winner_max_frac",
    ):
        assert key in output
        assert torch.isfinite(output[key]).all()


def test_mon_step_matches_manual_wta():
    torch.manual_seed(1)
    m = _tiny_branch_model(dim=8, num_branches=3)
    holder = _Holder(m)
    z = torch.randn(4, 4, 8)
    p = torch.randn(4, 3, 8)
    cfg = _cfg(roll_weight=1.0)
    imag = m.forward_imaginer
    h0 = torch.stack([z[:, 0], p[:, 0]], dim=1)
    h1 = torch.stack([p[:, 0], p[:, 1]], dim=1)
    y2 = imag.forward_branches(h0)
    y3_step = imag.forward_branches(h1)
    p0_m = p[:, 0].unsqueeze(1).expand(4, 3, 8)
    y3_roll = imag.forward_assigned(torch.stack([p0_m, y2], dim=2))
    e2 = (y2 - z[:, 2].unsqueeze(1)).pow(2).mean(dim=-1)
    e3_step = (y3_step - z[:, 3].unsqueeze(1)).pow(2).mean(dim=-1)
    e3_roll = (y3_roll - z[:, 3].unsqueeze(1)).pow(2).mean(dim=-1)
    e_step = 0.5 * (e2 + e3_step)
    winner = (e_step + e3_roll).detach().argmin(dim=1)
    step_ref = e_step.gather(1, winner.unsqueeze(1)).mean()
    roll_ref = e3_roll.gather(1, winner.unsqueeze(1)).mean()
    output = {}
    _fill_forward_losses(holder, output, z, p, None, cfg)
    assert torch.allclose(output["forward_step_loss"], step_ref, atol=1e-6)
    assert torch.allclose(output["forward_roll_loss"], roll_ref, atol=1e-6)


def test_nonwinner_heads_have_no_wta_grad():
    torch.manual_seed(2)
    imag = BranchPreservingCausalLatentImaginer(
        dim=8, hidden_dim=16, depth=1, num_branches=3
    )
    z = torch.randn(1, 4, 8)
    p = torch.randn(1, 3, 8)
    holder = _Holder(type("M", (), {"forward_imaginer": imag})())
    output = {}
    _fill_forward_losses(holder, output, z, p, None, _cfg())
    output["forward_loss"].backward()
    e_total_winner = None
    with torch.no_grad():
        h0 = torch.stack([z[:, 0], p[:, 0]], dim=1)
        h1 = torch.stack([p[:, 0], p[:, 1]], dim=1)
        y2 = imag.forward_branches(h0)
        y3_step = imag.forward_branches(h1)
        p0_m = p[:, 0].unsqueeze(1).expand(1, 3, 8)
        y3_roll = imag.forward_assigned(torch.stack([p0_m, y2], dim=2))
        e2 = (y2 - z[:, 2].unsqueeze(1)).pow(2).mean(dim=-1)
        e3_step = (y3_step - z[:, 3].unsqueeze(1)).pow(2).mean(dim=-1)
        e3_roll = (y3_roll - z[:, 3].unsqueeze(1)).pow(2).mean(dim=-1)
        e_total_winner = int((0.5 * (e2 + e3_step) + e3_roll).argmin(dim=1).item())
    for m, head in enumerate(imag.branch_heads):
        g = head.weight.grad
        if m == e_total_winner:
            assert g is not None and g.abs().sum().item() > 0
        else:
            assert g is None or g.abs().sum().item() == 0


def test_illegal_variant_mentions_branch_preserving():
    try:
        _forward_variant(OmegaConf.create({"loss": {"forward": {"variant": "nope"}}}))
        assert False, "expected ValueError"
    except ValueError as err:
        assert "branch_preserving" in str(err)


def test_too_few_latent_frames_raises():
    m = _tiny_branch_model(dim=8, num_branches=2)
    try:
        _fill_forward_losses(_Holder(m), {}, torch.randn(2, 3, 8), torch.randn(2, 2, 8), None, _cfg())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_imagine_forward_branches_shapes():
    m = _tiny_branch_model(dim=16, num_branches=3)
    h = torch.randn(4, 2, 16)
    out0 = m.imagine_forward_branches(h, 0)
    assert out0.shape == (4, 3, 16)
    assert torch.allclose(out0, h[:, -1].unsqueeze(1).expand(4, 3, 16))
    for k in (1, 3, 15):
        out = m.imagine_forward_branches(h, k)
        assert out.shape == (4, 3, 16)
    hs = torch.randn(2, 5, 2, 16)
    assert m.imagine_forward_branches(hs, 2).shape == (2, 5, 3, 16)


def test_masked_branch_recursion():
    torch.manual_seed(0)
    m = _tiny_branch_model(dim=8, num_branches=2)
    h = torch.randn(3, 2, 8)
    steps = torch.tensor([0, 1, 3], dtype=torch.long)
    out = m.imagine_forward_branches(h, steps)
    assert out.shape == (3, 2, 8)
    assert torch.allclose(out[0], h[0, -1].unsqueeze(0).expand(2, 8))
    one = m.imagine_forward_branches(h[1:2], 1)
    three = m.imagine_forward_branches(h[2:3], 3)
    assert torch.allclose(out[1], one[0], atol=1e-6)
    assert torch.allclose(out[2], three[0], atol=1e-6)


def test_m1_hardmin_is_single_branch_mse():
    torch.manual_seed(0)
    m = _tiny_branch_model(dim=8, num_branches=1)
    B, S, H, A = 2, 3, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    goal = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)
    z_goal = m.encode({"pixels": goal[:, 0].clone()})["emb"]
    goal_emb = z_goal.unsqueeze(1).expand(B, S, 1, z_goal.size(-1)).contiguous()
    info = {
        "pixels": pixels.clone(),
        "goal": goal.clone(),
        "goal_emb": goal_emb.clone(),
        "imagine_steps": torch.full((B, S, 1), 2, dtype=torch.int64),
    }
    m.set_planning_mode("forward")
    cost = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    rolled = m.rollout(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    hist = rolled["predicted_emb"][..., -2:, :]
    z_g = m._as_bs_latent(goal_emb, rolled["predicted_emb"][..., -1, :])
    ep = m.imagine_forward_branches(hist, info["imagine_steps"])
    ref = (ep.squeeze(-2) - z_g.detach()).pow(2).sum(dim=-1)
    assert torch.allclose(cost, ref, atol=1e-5, rtol=1e-4)


def test_goal_change_does_not_change_branch_predictions():
    m = _tiny_branch_model(dim=8, num_branches=3)
    h = torch.randn(4, 2, 8)
    y_a = m.imagine_forward_branches(h, 3)
    y_b = m.imagine_forward_branches(h, 3)
    assert torch.allclose(y_a, y_b)
    B, S, H, A = 1, 2, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)
    g1 = torch.randn(B, S, 1, 3, 8, 8)
    g2 = torch.randn(B, S, 1, 3, 8, 8)
    e1 = m.encode({"pixels": g1[:, 0]})["emb"].unsqueeze(1).expand(B, S, 1, 8)
    e2 = m.encode({"pixels": g2[:, 0]})["emb"].unsqueeze(1).expand(B, S, 1, 8)
    steps = torch.full((B, S, 1), 2, dtype=torch.int64)
    captured = []

    def _wrap(history, steps_i):
        captured.append(history.detach().clone())
        return orig(history, steps_i)

    orig = m.imagine_forward_branches
    m.imagine_forward_branches = _wrap
    m.set_planning_mode("forward")
    c1 = m.get_cost(
        {"pixels": pixels.clone(), "goal": g1.clone(), "goal_emb": e1.clone(), "imagine_steps": steps},
        actions.clone(),
    )
    c2 = m.get_cost(
        {"pixels": pixels.clone(), "goal": g2.clone(), "goal_emb": e2.clone(), "imagine_steps": steps},
        actions.clone(),
    )
    m.imagine_forward_branches = orig
    assert len(captured) == 2
    assert torch.allclose(captured[0], captured[1])
    assert not torch.allclose(c1, c2)


def test_eval_reads_last_two_predicted_latents():
    m = _tiny_branch_model(dim=8, num_branches=2)
    B, S, H, A = 1, 2, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    goal = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)
    z_goal = m.encode({"pixels": goal[:, 0]})["emb"].unsqueeze(1).expand(B, S, 1, 8)
    info = {
        "pixels": pixels.clone(),
        "goal": goal.clone(),
        "goal_emb": z_goal.clone(),
        "imagine_steps": torch.ones(B, S, 1, dtype=torch.int64),
    }
    rolled = m.rollout(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    expected = rolled["predicted_emb"][..., -2:, :]
    seen = {}

    def _wrap(history, steps):
        seen["history"] = history.detach().clone()
        return orig(history, steps)

    orig = m.imagine_forward_branches
    m.imagine_forward_branches = _wrap
    m.set_planning_mode("forward")
    m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    m.imagine_forward_branches = orig
    assert torch.allclose(seen["history"], expected)


def test_meet_and_fusion_rejected_on_branch_model():
    m = _tiny_branch_model(dim=8, num_branches=2)
    B, S, H, A = 1, 2, 5, 10
    info = {
        "pixels": torch.randn(B, S, 1, 3, 8, 8),
        "goal": torch.randn(B, S, 1, 3, 8, 8),
        "imagine_steps": torch.ones(B, S, 1, dtype=torch.int64),
    }
    actions = torch.randn(B, S, H, A)
    for mode in ("meet", "fusion_avg05"):
        m.set_planning_mode(mode)
        try:
            m.get_cost(
                {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
                actions.clone(),
            )
            assert False, f"expected ValueError for {mode}"
        except ValueError as err:
            assert "branch_preserving" in str(err)


def test_legacy_latent_fill_unchanged():
    torch.manual_seed(0)
    m = _tiny_model(dim=16)
    z = torch.randn(3, 4, 16)
    p = torch.randn(3, 3, 16)
    output = {}
    _fill_forward_losses(_Holder(m), output, z, p, None, _cfg(variant="latent"))
    f_pred = m.forward_imaginer(p[:, 0:2])
    f_roll = m.forward_imaginer(m.forward_imaginer(p[:, 0:1]))
    assert torch.allclose(output["forward_step_loss"], (f_pred - z[:, 2:4]).pow(2).mean())
    assert torch.allclose(output["forward_roll_loss"], (f_roll - z[:, 3:4]).pow(2).mean())


def test_checkpoint_param_tokens_and_missing_heads():
    imag = BranchPreservingCausalLatentImaginer(
        dim=8, hidden_dim=16, depth=1, num_branches=3
    )
    state = imag.state_dict()
    assert any("branch_heads" in k for k in state)
    assert any("history_fuse" in k for k in state)
    rebuilt = instantiate(
        {
            "_target_": "module.BranchPreservingCausalLatentImaginer",
            "dim": 8,
            "hidden_dim": 16,
            "depth": 1,
            "num_branches": 3,
        }
    )
    rebuilt.load_state_dict(state)
    h = torch.randn(2, 2, 8)
    assert torch.allclose(rebuilt.forward_branches(h), imag.forward_branches(h))
    incomplete = {k: v for k, v in state.items() if "branch_heads.0" not in k}
    try:
        BranchPreservingCausalLatentImaginer(
            dim=8, hidden_dim=16, depth=1, num_branches=3
        ).load_state_dict(incomplete, strict=True)
        assert False, "expected missing branch head error"
    except RuntimeError:
        pass
    missing = [k for k in imag.state_dict() if k not in incomplete]
    missing_imag = [k for k in missing if "branch" in k or "imaginer" in k]
    assert missing_imag


def test_train_override_sets_branch_target():
    cfg = OmegaConf.create(
        {
            "loss": {"forward": {"variant": "branch_preserving", "branches": 4}},
            "model": {"forward_imaginer": {"_target_": "module.CausalLatentImaginer"}},
        }
    )
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.model.forward_imaginer._target_ = (
            "module.BranchPreservingCausalLatentImaginer"
        )
        cfg.model.forward_imaginer.num_branches = int(cfg.loss.forward.branches)
    assert cfg.model.forward_imaginer._target_.endswith(
        "BranchPreservingCausalLatentImaginer"
    )
    assert int(cfg.model.forward_imaginer.num_branches) == 4
    assert isinstance(CausalLatentImaginer(), CausalLatentImaginer)
