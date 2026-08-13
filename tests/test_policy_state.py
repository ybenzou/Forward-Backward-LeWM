"""FBWorldModelPolicy elapsed / flush / terminated contracts."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
from stable_worldmodel.policy import PlanConfig

from policy import FBWorldModelPolicy


class _DummySolver:
    def __init__(self):
        self.model = SimpleNamespace(
            set_planning_mode=lambda mode: None,
            parameters=lambda: iter([torch.zeros(1, requires_grad=False)]),
            encode=lambda info: {"emb": torch.zeros(info["pixels"].shape[0], 1, 192)},
            imagine_backward=lambda z, steps, z_now=None: z,
        )
        self.n_calls = 0
        self.last_info = None
        self._n_envs = 2
        self._horizon = 5
        self._action_dim = 2

    def configure(self, *, action_space=None, n_envs=1, config=None):
        self._n_envs = n_envs
        return None

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def horizon(self) -> int:
        return self._horizon

    def solve(self, info_dict, init_action=None):
        return self.__call__(info_dict, init_action=init_action)

    def __call__(self, info, init_action=None):
        self.n_calls += 1
        self.last_info = info
        B = next(v for v in info.values() if torch.is_tensor(v)).shape[0]
        # (B, horizon, action_block * raw_dim) so reshape -> (B, 25, 2)
        actions = torch.zeros(B, 5, 10)
        return {"actions": actions}


class _DummyEnv:
    num_envs = 2
    action_space = SimpleNamespace(shape=(2, 2))
    single_action_space = SimpleNamespace(shape=(2,))


def _make_policy():
    cfg = PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver = _DummySolver()
    pol = FBWorldModelPolicy(
        solver=solver,
        config=cfg,
        goal_offset=75,
        planning_mode="official",
        process={},
        transform={},
    )
    pol.set_env(_DummyEnv())
    return pol, solver


def _obs(n=2):
    return {
        "pixels": torch.zeros(n, 1, 3, 8, 8),
        "goal": torch.zeros(n, 1, 3, 8, 8),
        "terminated": np.array([False, False]),
    }


def test_first_replan_uses_elapsed_zero():
    pol, solver = _make_policy()
    # Monkeypatch prepare_info to identity
    pol._prepare_info = lambda d: d
    pol.get_action(_obs())
    assert solver.n_calls == 1
    assert int(pol._elapsed_steps[0]) == 1  # incremented after action return
    # At replan time, imagine_steps computed with elapsed=0 before increment.
    # For official mode we still attach imagine_steps.
    assert "imagine_steps" in solver.last_info
    assert int(solver.last_info["imagine_steps"][0, 0]) == 10  # offset 75


def test_elapsed_reaches_25_after_25_actions():
    pol, solver = _make_policy()
    pol._prepare_info = lambda d: d
    # First call fills buffer with 25 actions and pops one -> elapsed=1, buffer=24
    pol.get_action(_obs())
    # Drain remaining 24 buffered actions without replan
    for _ in range(24):
        pol.get_action(_obs())
    assert int(pol._elapsed_steps[0]) == 25
    assert len(pol._action_buffer[0]) == 0
    # Next call replans with elapsed=25 -> k=5 for offset 75
    before = solver.n_calls
    pol.get_action(_obs())
    assert solver.n_calls == before + 1
    assert int(solver.last_info["imagine_steps"][0, 0]) == 5


def test_needs_flush_resets_elapsed():
    pol, solver = _make_policy()
    pol._prepare_info = lambda d: d
    for _ in range(5):
        pol.get_action(_obs())
    assert int(pol._elapsed_steps[0]) > 0
    obs = _obs()
    obs["_needs_flush"] = np.array([True, False])
    pol.get_action(obs)
    assert int(pol._elapsed_steps[0]) == 1  # reset then +1
    assert int(pol._elapsed_steps[1]) >= 1


def test_terminated_env_does_not_replan_or_increment():
    pol, solver = _make_policy()
    pol._prepare_info = lambda d: d
    pol.get_action(_obs())
    calls = solver.n_calls
    # Mark both terminated; should not replan even if buffers empty.
    pol._action_buffer = [deque(), deque()]
    elapsed_before = pol._elapsed_steps.copy()
    obs = _obs()
    obs["terminated"] = np.array([True, True])
    pol.get_action(obs)
    assert solver.n_calls == calls
    assert np.array_equal(pol._elapsed_steps, elapsed_before)


def test_encode_goal_pixels_ignores_goal_action():
    """Regression: goal_action is raw dim-2 and must not enter action_encoder."""
    from tests.test_model_contracts import _tiny_model

    pol, _ = _make_policy()
    model = _tiny_model()
    captured = {}

    def _encode(info):
        captured["keys"] = set(info.keys())
        assert "action" not in info
        b = info["pixels"].shape[0]
        return {"emb": torch.zeros(b, 1, 192)}

    model.encode = _encode
    pol.solver.model = model
    sliced = {
        "goal": torch.zeros(4, 1, 3, 8, 8),
        "goal_action": torch.zeros(4, 1, 2),  # env raw action
        "goal_state": torch.zeros(4, 1, 5),
        "action": torch.zeros(4, 1, 2),
    }
    emb = pol._encode_goal_pixels(sliced)
    assert emb.shape == (4, 1, 192)
    assert "action" not in captured["keys"]
    assert "state" in captured["keys"]
