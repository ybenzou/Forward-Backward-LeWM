"""Opt-in TRM planning and backward-compatibility contracts."""

import torch
from torch import nn

from planning import BASE_MODES, TRM_MODES
from scripts.eval_fblewm_matrix import _expand_modes, parse_args
from scripts.run_trm_baseline import parse_args as parse_runner_args
from scripts.run_trm_tworoom import parse_args as parse_tworoom_args
from tests.test_model_contracts import _tiny_model


class _SquaredDistanceHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        return (z_i - z_j).pow(2).sum(dim=-1) + 0.0 * self.anchor


def _inputs(dim: int = 16):
    batch, samples, horizon, action_dim = 2, 5, 5, 10
    pixels = torch.randn(batch, samples, 1, 3, 8, 8)
    goal = torch.randn(batch, samples, 1, 3, 8, 8)
    actions = torch.randn(batch, samples, horizon, action_dim)
    return pixels, goal, actions


def _cost(model, mode, pixels, goal, actions):
    model.set_planning_mode(mode)
    return model.get_cost(
        {"pixels": pixels.clone(), "goal": goal.clone()},
        actions.clone(),
    )


def test_trm_modes_are_opt_in():
    assert "trm_replace" not in BASE_MODES
    assert "trm_hybrid" not in BASE_MODES
    assert TRM_MODES == ("trm_replace", "trm_hybrid")
    assert "trm_replace" not in _expand_modes("all")
    assert parse_args(["--policy=dummy.pt"]).trm_head is None
    runner = parse_runner_args(["--trm-head=dummy.pt"])
    assert runner.task == "tworoom"
    assert runner.mode is None
    assert runner.seeds == "42,43,44,45,46,47,48,49,50,51"
    launcher = parse_tworoom_args([])
    assert launcher.stage == "all"
    assert launcher.seeds == "42,43,44,45,46,47,48,49,50,51"
    assert parse_tworoom_args(["--stage", "train"]).stage == "train"


def test_trm_requires_attached_head():
    model = _tiny_model(dim=16)
    pixels, goal, actions = _inputs()
    model.set_planning_mode("trm_replace")
    try:
        model.get_cost({"pixels": pixels, "goal": goal}, actions)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "set_trm_head" in str(exc)


def test_trm_replace_matches_squared_latent_head():
    torch.manual_seed(0)
    model = _tiny_model(dim=16)
    pixels, goal, actions = _inputs()
    before_keys = tuple(model.state_dict())
    model.set_trm_head(_SquaredDistanceHead(16))
    after_keys = tuple(model.state_dict())

    official = _cost(model, "official", pixels, goal, actions)
    trm = _cost(model, "trm_replace", pixels, goal, actions)
    assert before_keys == after_keys
    assert official.shape == trm.shape == (2, 5)
    assert torch.allclose(official, trm, atol=1e-6, rtol=1e-5)


def test_trm_hybrid_standardizes_only_candidate_axis():
    torch.manual_seed(1)
    model = _tiny_model(dim=16)
    pixels, goal, actions = _inputs()
    model.set_trm_head(_SquaredDistanceHead(16), weight=1.0, eps=1e-8)

    official = _cost(model, "official", pixels, goal, actions)
    hybrid = _cost(model, "trm_hybrid", pixels, goal, actions)
    expected = 2.0 * model._candidate_zscore(official, 1e-8)
    assert torch.allclose(hybrid, expected, atol=1e-5, rtol=1e-5)
    probe = torch.tensor([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    standardized = model._candidate_zscore(probe, 1e-8)
    assert torch.allclose(
        standardized.mean(dim=-1), torch.zeros(2), atol=1e-6, rtol=0
    )


def test_attaching_trm_does_not_change_existing_modes():
    torch.manual_seed(2)
    model = _tiny_model(dim=16)
    pixels, goal, actions = _inputs()
    before = _cost(model, "official", pixels, goal, actions)
    model.set_trm_head(_SquaredDistanceHead(16))
    after = _cost(model, "official", pixels, goal, actions)
    model.clear_trm_head()
    cleared = _cost(model, "official", pixels, goal, actions)
    assert torch.equal(before, after)
    assert torch.equal(before, cleared)


def test_candidate_zscore_is_stable_for_one_or_constant_candidate():
    one = torch.tensor([[4.0], [7.0]])
    constant = torch.full((2, 4), 3.0)
    assert torch.equal(
        _tiny_model(dim=16)._candidate_zscore(one, 1e-8),
        torch.zeros_like(one),
    )
    assert torch.equal(
        _tiny_model(dim=16)._candidate_zscore(constant, 1e-8),
        torch.zeros_like(constant),
    )
