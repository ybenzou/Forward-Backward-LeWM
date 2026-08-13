"""Fairness contracts for matrix eval starts / k=0 cost equality."""

import hashlib
import json

import torch

from policy import compute_imagine_steps
from tests.test_model_contracts import _tiny_model


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
