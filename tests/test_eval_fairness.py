"""Fairness contracts for matrix eval starts / k=0 cost equality."""

import hashlib
import json
from pathlib import Path

import torch

import inspect

import train
from policy import compute_imagine_steps
from tests.test_model_contracts import (
    _tiny_branch_model,
    _tiny_conditional_model,
    _tiny_model,
)


def test_shared_manifest_hash_stable():
    manifest = {
        "row_indices": [1, 2, 3],
        "episodes": [10, 11, 12],
        "start_steps": [0, 1, 2],
        "seed": 42,
        "num_eval": 3,
        "goal_offset_for_sampling": 100,
    }
    h1 = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert h1 == h2
    assert len({h1 for _ in range(12)}) == 1


def test_k0_forward_backward_match_official_cost():
    torch.manual_seed(0)
    m = _tiny_model()
    B, S, H, A = 2, 3, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    goal = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)

    # Shared goal embedding for all modes.
    goal_info = {"pixels": goal[:, 0].clone()}
    z_goal = m.encode(goal_info)["emb"]  # (B, 1, D)
    goal_emb = z_goal.unsqueeze(1).expand(B, S, 1, z_goal.size(-1)).contiguous()

    m.set_planning_mode("official")
    c_official = m.get_cost(
        {
            "pixels": pixels.clone(),
            "goal": goal.clone(),
            "goal_emb": goal_emb.clone(),
        },
        actions.clone(),
    )

    m.set_planning_mode("forward")
    c_forward = m.get_cost(
        {
            "pixels": pixels.clone(),
            "goal": goal.clone(),
            "goal_emb": goal_emb.clone(),
            "imagine_steps": torch.zeros(B, S, 1, dtype=torch.int64),
        },
        actions.clone(),
    )

    m.set_planning_mode("backward")
    c_backward = m.get_cost(
        {
            "pixels": pixels.clone(),
            "goal": goal.clone(),
            "goal_emb": goal_emb.clone(),
        },
        actions.clone(),
    )

    assert torch.allclose(c_official, c_forward, atol=1e-6, rtol=1e-5)
    assert torch.allclose(c_official, c_backward, atol=1e-6, rtol=1e-5)


def test_offset25_depth_is_zero():
    assert compute_imagine_steps(25, 0, 25, 5) == 0


def test_pred_goal_k0_fair_with_depth_cap_set():
    torch.manual_seed(0)
    m = _tiny_conditional_model()
    m.backward_anchor = "pred"
    m.set_backward_depth_cap(2)
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
    c_o = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    m.set_planning_mode("backward")
    c_b = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    assert torch.allclose(c_o, c_b, atol=1e-5, rtol=1e-4)


def test_train_does_not_set_depth_cap():
    src = inspect.getsource(train)
    assert "backward_depth_cap" not in src
    assert "set_backward_depth_cap" not in src
    assert "forward_depth_override" not in src
    assert "set_forward_depth_override" not in src


def test_forward_depth_override_none_keeps_k0_match():
    torch.manual_seed(0)
    m = _tiny_model()
    m.set_forward_depth_override(None)
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
    c_official = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    m.set_planning_mode("forward")
    c_forward = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    assert torch.allclose(c_official, c_forward, atol=1e-6, rtol=1e-5)


def test_forward_depth_override_ignored_by_official():
    torch.manual_seed(0)
    m = _tiny_model()
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
    c_plain = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    m.set_forward_depth_override(5)
    c_capped = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    assert torch.allclose(c_plain, c_capped, atol=1e-6, rtol=1e-5)


def test_forward_depth_override_changes_forward_cost():
    torch.manual_seed(0)
    m = _tiny_model()
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
    m.set_planning_mode("forward")
    c_k0 = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    m.set_forward_depth_override(3)
    c_k3 = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    assert c_k0.shape == c_k3.shape == (B, S)
    assert not torch.allclose(c_k0, c_k3, atol=1e-5, rtol=1e-4)


def test_starts_per_offset_flag_rejects_shared_manifest():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "eval_fblewm_matrix.py").read_text()
    assert "--starts-per-offset" in src
    assert "cannot be combined with --starts-manifest" in src
    assert "--forward-depth-override" in src


def test_k0_branch_forward_matches_official_cost():
    torch.manual_seed(0)
    m = _tiny_branch_model(dim=16, num_branches=3)
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
    c_official = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    m.set_planning_mode("forward")
    c_forward = m.get_cost(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()},
        actions.clone(),
    )
    assert c_official.shape == (B, S)
    assert torch.allclose(c_official, c_forward, atol=1e-6, rtol=1e-5)
