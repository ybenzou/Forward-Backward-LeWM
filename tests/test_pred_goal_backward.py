"""Contracts for pred_goal: B(P, z_later) -> z at P's time."""

import torch
from omegaconf import OmegaConf

from fblewm import FBLeWM
from tests.test_model_contracts import _tiny_conditional_model
from train import _fill_pred_goal_backward


def test_pred_goal_anchor_skips_coarsen():
    m = _tiny_conditional_model()
    m.backward_anchor = "pred"
    z_now = torch.randn(3, 192)
    z_goal = torch.randn(3, 192)
    out = m.imagine_backward(z_goal, 3, z_now=z_now)
    one = m.backward_imaginer(z_now, z_goal)
    two = m.backward_imaginer(z_now, one)
    three = m.backward_imaginer(z_now, two)
    assert torch.allclose(out, three)


def test_pred_goal_k0_matches_official():
    torch.manual_seed(0)
    m = _tiny_conditional_model()
    m.backward_anchor = "pred"
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
        "imagine_steps": torch.zeros(B, S, 1, dtype=torch.int64),
    }
    m.set_planning_mode("official")
    c_o = m.get_cost({k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()}, actions.clone())
    m.set_planning_mode("backward")
    c_b = m.get_cost({k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()}, actions.clone())
    assert torch.allclose(c_o, c_b, atol=1e-5, rtol=1e-4)


def test_pred_goal_cost_depends_on_candidate_endpoint():
    torch.manual_seed(1)
    m = _tiny_conditional_model()
    m.backward_anchor = "pred"
    B, S, H, A = 1, 4, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    goal = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)
    z_goal = m.encode({"pixels": goal[:, 0].clone()})["emb"]
    goal_emb = z_goal.unsqueeze(1).expand(B, S, 1, z_goal.size(-1)).contiguous()
    m.set_planning_mode("backward")
    costs = m.get_cost(
        {
            "pixels": pixels,
            "goal": goal,
            "goal_emb": goal_emb,
            "imagine_steps": torch.full((B, S, 1), 2, dtype=torch.int64),
        },
        actions,
    )
    assert costs.shape == (B, S)
    assert costs.std() > 0


def test_pred_goal_depth_cap_none_matches_uncapped():
    torch.manual_seed(0)
    m = _tiny_conditional_model()
    m.backward_anchor = "pred"
    z_now = torch.randn(4, 192)
    z_goal = torch.randn(4, 192)
    uncapped = m.imagine_backward(z_goal, 4, z_now=z_now)
    m.set_backward_depth_cap(None)
    still = m.imagine_backward(z_goal, 4, z_now=z_now)
    assert torch.allclose(uncapped, still)


def test_pred_goal_depth_cap_maps_to_min_k():
    torch.manual_seed(1)
    m = _tiny_conditional_model()
    m.backward_anchor = "pred"
    z_now = torch.randn(3, 192)
    z_goal = torch.randn(3, 192)
    one = m.imagine_backward(z_goal, 1, z_now=z_now)
    two = m.imagine_backward(z_goal, 2, z_now=z_now)
    m.set_backward_depth_cap(1)
    capped = m.imagine_backward(z_goal, 5, z_now=z_now)
    assert torch.allclose(capped, one)
    assert not torch.allclose(capped, two)


def test_pred_goal_depth_cap_keeps_k0():
    m = _tiny_conditional_model()
    m.backward_anchor = "pred"
    m.set_backward_depth_cap(1)
    z_now = torch.randn(2, 192)
    z_goal = torch.randn(2, 192)
    assert torch.equal(m.imagine_backward(z_goal, 0, z_now=z_now), z_goal)


def test_obs_anchor_ignores_pred_goal_depth_cap():
    torch.manual_seed(2)
    m = _tiny_conditional_model()
    m.backward_anchor = "obs"
    z_now = torch.randn(3, 192)
    z_goal = torch.randn(3, 192)
    # now-B coarsens k=3 -> 1, so cap must not change the obs path.
    base = m.imagine_backward(z_goal, 3, z_now=z_now)
    m.set_backward_depth_cap(2)
    after = m.imagine_backward(z_goal, 3, z_now=z_now)
    assert torch.allclose(base, after)


def test_pred_goal_train_fill_logs_identity_gap():
    m = _tiny_conditional_model()
    z = torch.randn(8, 4, 192)
    p = torch.randn(8, 3, 192)
    cfg = OmegaConf.create(
        {"loss": {"backward": {"p_noise": 0.1, "goal_margin": 0.1, "goal_rank_weight": 0.5}}}
    )
    output = {}
    dummy = type("M", (), {"model": m})()
    _fill_pred_goal_backward(dummy, output, z, p, cfg)
    assert output["backward_step_loss"].ndim == 0
    assert output["backward_roll_loss"].ndim == 0
    assert "backward_pred_mse" in output
    assert "backward_shuffle_mse" in output
    assert "backward_identity_gap" in output
    assert "backward_goal_rank_loss" in output
