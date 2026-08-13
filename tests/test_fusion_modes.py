"""Fusion / switch / meet planning-mode contracts."""

import torch

from planning import FUSION_MODES, split_meet_steps
from policy import compute_switch_remain_branch
from tests.test_model_contracts import _tiny_model


def test_split_meet_steps():
    assert split_meet_steps(0) == (0, 0)
    assert split_meet_steps(5) == (2, 3)
    assert split_meet_steps(10) == (5, 5)
    assert split_meet_steps(15) == (7, 8)


def test_switch_remain_branch_schedule():
    # remaining = offset - elapsed; threshold 50
    assert compute_switch_remain_branch(100, 0, 50) == 0  # forward
    assert compute_switch_remain_branch(100, 50, 50) == 1  # backward
    assert compute_switch_remain_branch(75, 0, 50) == 0
    assert compute_switch_remain_branch(50, 0, 50) == 1
    assert compute_switch_remain_branch(25, 0, 50) == 1


def test_fusion_modes_cost_shape_and_k0_match_official():
    torch.manual_seed(0)
    m = _tiny_model()
    B, S, H, A = 2, 3, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    goal = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)
    goal_info = {"pixels": goal[:, 0].clone()}
    z_goal = m.encode(goal_info)["emb"]
    goal_emb = z_goal.unsqueeze(1).expand(B, S, 1, z_goal.size(-1)).contiguous()

    m.set_planning_mode("official")
    c_official = m.get_cost(
        {"pixels": pixels.clone(), "goal": goal.clone(), "goal_emb": goal_emb.clone()},
        actions.clone(),
    )

    for mode in FUSION_MODES:
        m.set_planning_mode(mode)
        m.set_goal_offset(100)
        info = {
            "pixels": pixels.clone(),
            "goal": goal.clone(),
            "goal_emb": goal_emb.clone(),
            "imagine_steps": torch.zeros(B, S, 1, dtype=torch.int64),
        }
        if mode == "switch_remain":
            info["plan_branch"] = torch.zeros(B, S, 1, dtype=torch.int64)
        cost = m.get_cost(info, actions.clone())
        assert cost.shape == (B, S), (mode, cost.shape)
        # k=0: all fusion strategies reduce to official terminal MSE
        assert torch.allclose(cost, c_official, atol=1e-5, rtol=1e-4), mode


def test_fusion_avg_is_convex_combination():
    torch.manual_seed(1)
    m = _tiny_model()
    B, S, H, A = 2, 4, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    goal = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)
    steps = torch.full((B, S, 1), 5, dtype=torch.int64)
    base = {
        "pixels": pixels,
        "goal": goal,
        "imagine_steps": steps,
    }
    m.set_planning_mode("forward")
    # Build shared rollout endpoint costs via public API pieces.
    info_f = {k: v.clone() if torch.is_tensor(v) else v for k, v in base.items()}
    # Use dual-cost path internals after one encode
    info_f["goal_emb"] = m._encode_goal(info_f)
    info_f = m.rollout(info_f, actions.clone())
    endpoint = info_f["predicted_emb"][..., -1, :]
    z_goal = m._as_bs_latent(info_f["goal_emb"], endpoint)
    c_f = m._forward_cost(endpoint, z_goal, steps)
    c_b = m._backward_cost(endpoint, z_goal, steps, z_now=endpoint.detach())

    m.set_planning_mode("fusion_avg05")
    c_avg = m._combine_fusion("fusion_avg05", c_f, c_b)
    assert torch.allclose(c_avg, 0.5 * c_f + 0.5 * c_b)

    m.set_planning_mode("fusion_avg07")
    assert m._fusion_alpha == 0.7
    c_avg07 = m._combine_fusion("fusion_avg07", c_f, c_b)
    assert torch.allclose(c_avg07, 0.7 * c_f + 0.3 * c_b)

    assert torch.allclose(
        m._combine_fusion("fusion_max", c_f, c_b), torch.maximum(c_f, c_b)
    )
    assert torch.allclose(
        m._combine_fusion("fusion_min", c_f, c_b), torch.minimum(c_f, c_b)
    )

    c_o = m._latent_mse(endpoint, z_goal)
    m.set_planning_mode("fusion_ofb")
    info_ofb = {k: v.clone() if torch.is_tensor(v) else v for k, v in base.items()}
    info_ofb["goal_emb"] = m._encode_goal(info_ofb)
    c_ofb = m.get_cost(info_ofb, actions.clone())
    assert torch.allclose(c_ofb, (c_o + c_f + c_b) / 3.0, atol=1e-5, rtol=1e-4)
