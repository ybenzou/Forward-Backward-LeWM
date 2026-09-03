"""Shape / recursion contracts for CausalLatentImaginer and FBLeWM costs."""

import torch
from torch import nn

from fblewm import FBLeWM
from module import (
    ActionAlignedCausalLatentImaginer,
    BranchPreservingCausalLatentImaginer,
    CausalLatentImaginer,
    ConditionalLatentImaginer,
    SequentialActionCausalLatentImaginer,
)


class _TinyEnc(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(3 * 8 * 8, dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        b = pixels.size(0)
        x = pixels.reshape(b, -1)[:, : 3 * 8 * 8]
        if x.size(1) < 3 * 8 * 8:
            x = torch.nn.functional.pad(x, (0, 3 * 8 * 8 - x.size(1)))
        emb = self.lin(x)

        class Out:
            pass

        out = Out()
        # fake last_hidden_state with CLS at index 0
        out.last_hidden_state = emb.unsqueeze(1)
        return out


class _TinyPred(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, c):
        return self.lin(x)


class _TinyAct(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.lin = nn.Linear(10, dim)

    def forward(self, x):
        return self.lin(x.float())


def _tiny_model(dim=192):
    return FBLeWM(
        encoder=_TinyEnc(dim),
        predictor=_TinyPred(dim),
        action_encoder=_TinyAct(dim),
        forward_imaginer=CausalLatentImaginer(dim=dim, hidden_dim=64, depth=1),
        backward_imaginer=CausalLatentImaginer(dim=dim, hidden_dim=64, depth=1),
    )


def _tiny_action_aligned_model(dim=192, action_dim=10):
    return FBLeWM(
        encoder=_TinyEnc(dim),
        predictor=_TinyPred(dim),
        action_encoder=_TinyAct(dim),
        forward_imaginer=ActionAlignedCausalLatentImaginer(
            dim=dim, hidden_dim=64, depth=1, action_dim=action_dim
        ),
        backward_imaginer=CausalLatentImaginer(dim=dim, hidden_dim=64, depth=1),
    )


def _tiny_sequential_model(dim=192, action_dim=10):
    return FBLeWM(
        encoder=_TinyEnc(dim),
        predictor=_TinyPred(dim),
        action_encoder=_TinyAct(dim),
        forward_imaginer=SequentialActionCausalLatentImaginer(
            dim=dim, hidden_dim=64, depth=1, action_dim=action_dim
        ),
        backward_imaginer=CausalLatentImaginer(dim=dim, hidden_dim=64, depth=1),
    )


def _tiny_branch_model(dim=192, num_branches=4):
    return FBLeWM(
        encoder=_TinyEnc(dim),
        predictor=_TinyPred(dim),
        action_encoder=_TinyAct(dim),
        forward_imaginer=BranchPreservingCausalLatentImaginer(
            dim=dim, hidden_dim=64, depth=1, num_branches=num_branches
        ),
        backward_imaginer=CausalLatentImaginer(dim=dim, hidden_dim=64, depth=1),
    )


def _tiny_conditional_model(dim=192):
    return FBLeWM(
        encoder=_TinyEnc(dim),
        predictor=_TinyPred(dim),
        action_encoder=_TinyAct(dim),
        forward_imaginer=CausalLatentImaginer(dim=dim, hidden_dim=64, depth=1),
        backward_imaginer=ConditionalLatentImaginer(dim=dim, hidden_dim=64, depth=1),
    )


def test_imaginer_conditional_flag():
    assert CausalLatentImaginer.is_conditional is False
    assert CausalLatentImaginer.is_action_aligned is False
    assert CausalLatentImaginer.is_sequential_action is False
    assert ConditionalLatentImaginer.is_conditional is True
    assert ActionAlignedCausalLatentImaginer.is_action_aligned is True
    assert ActionAlignedCausalLatentImaginer.is_conditional is False
    assert ActionAlignedCausalLatentImaginer.is_sequential_action is False
    assert SequentialActionCausalLatentImaginer.is_action_aligned is True
    assert SequentialActionCausalLatentImaginer.is_sequential_action is True
    assert SequentialActionCausalLatentImaginer.is_conditional is False
    assert BranchPreservingCausalLatentImaginer.is_branch_preserving is True
    assert BranchPreservingCausalLatentImaginer.is_action_aligned is False
    assert BranchPreservingCausalLatentImaginer.is_sequential_action is False
    assert BranchPreservingCausalLatentImaginer.is_conditional is False
    assert BranchPreservingCausalLatentImaginer.history_size == 2


def test_imaginer_preserves_leading_shape():
    imag = CausalLatentImaginer(dim=192, hidden_dim=64, depth=1)
    for shape in [(2, 192), (2, 3, 192), (2, 4, 192)]:
        x = torch.randn(*shape)
        y = imag(x)
        assert y.shape == x.shape


def test_recursive_zero_steps_is_identity():
    m = _tiny_model()
    z = torch.randn(3, 192)
    assert torch.equal(m.imagine_forward(z, 0), z)
    assert torch.equal(m.imagine_backward(z, 0), z)


def test_recursive_negative_steps_raises():
    m = _tiny_model()
    z = torch.randn(2, 192)
    try:
        m.imagine_forward(z, -1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_training_tensor_alignment():
    # p has length history_size=3, targets z[:,1:]
    B, T, D = 4, 4, 192
    emb = torch.randn(B, T, D)
    pred = emb[:, :3]  # pretend p1,p2,p3
    assert pred.shape[1] == 3
    f_in = pred[:, 0:2]
    f_tgt = emb[:, 2:4]
    assert f_in.shape == f_tgt.shape == (B, 2, D)
    # Legacy encoder targets
    b_in = torch.stack([emb[:, 3], emb[:, 2]], dim=1)
    b_tgt_z = torch.stack([emb[:, 2], emb[:, 1]], dim=1)
    assert b_in.shape == b_tgt_z.shape == (B, 2, D)
    # New pred-manifold targets: B(z3)->p2, B(z2)->p1
    b_tgt_p = torch.stack([pred[:, 1], pred[:, 0]], dim=1)
    assert b_tgt_p.shape == (B, 2, D)
    # Conditional B: (z0, z3)->z2 and (z0, z2)->z1
    z0 = emb[:, 0:1]
    assert z0.shape == (B, 1, D)
    assert emb[:, 2:3].shape == (B, 1, D)


def test_conditional_imaginer_preserves_goal_shape():
    imag = ConditionalLatentImaginer(dim=192, hidden_dim=64, depth=1)
    z_now = torch.randn(4, 192)
    z_goal = torch.randn(4, 3, 192)
    y = imag(z_now, z_goal)
    assert y.shape == z_goal.shape


def test_conditional_backward_freezes_now_and_coarsens_k():
    from planning import coarsen_backward_steps

    assert coarsen_backward_steps(0) == 0
    assert coarsen_backward_steps(5) == 1
    assert coarsen_backward_steps(10) == 2
    assert coarsen_backward_steps(15) == 3

    m = _tiny_conditional_model()
    z_now = torch.randn(3, 192)
    z_goal = torch.randn(3, 192)
    assert torch.equal(m.imagine_backward(z_goal, 0, z_now=z_now), z_goal)
    out = m.imagine_backward(z_goal, 15, z_now=z_now)
    assert out.shape == z_goal.shape
    # Fine k=15 coarsens to 3 pulls; all share the same z_now.
    one = m.backward_imaginer(z_now, z_goal)
    two = m.backward_imaginer(z_now, one)
    three = m.backward_imaginer(z_now, two)
    assert torch.allclose(out, three)


def test_conditional_k0_cost_matches_official():
    torch.manual_seed(0)
    m = _tiny_conditional_model()
    B, S, H, A = 2, 3, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    goal = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)
    goal_info = {"pixels": goal[:, 0].clone()}
    z_goal = m.encode(goal_info)["emb"]
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
    m.set_planning_mode("backward")
    c_backward = m.get_cost(
        {
            "pixels": pixels.clone(),
            "goal": goal.clone(),
            "goal_emb": goal_emb.clone(),
        },
        actions.clone(),
    )
    assert torch.allclose(c_official, c_backward, atol=1e-6, rtol=1e-5)


def test_cost_shape_for_all_modes():
    m = _tiny_model()
    B, S, H, A = 2, 4, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    goal = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)
    info = {"pixels": pixels, "goal": goal}
    for mode in ("official", "forward", "backward"):
        m.set_planning_mode(mode)
        info_i = dict(info)
        if mode == "forward":
            info_i["imagine_steps"] = torch.tensor([[0], [5]], dtype=torch.int64)
            # expand like CEM
            info_i["imagine_steps"] = info_i["imagine_steps"].unsqueeze(1).expand(B, S, 1)
        if mode == "backward":
            # inject precomputed subgoal
            info_i["goal_emb"] = torch.randn(B, S, 1, 192)
        cost = m.get_cost(info_i, actions)
        assert cost.shape == (B, S), (mode, cost.shape)


def test_branch_preserving_cost_shape_and_k0():
    torch.manual_seed(0)
    m = _tiny_branch_model(dim=16, num_branches=3)
    B, S, H, A = 2, 4, 5, 10
    pixels = torch.randn(B, S, 1, 3, 8, 8)
    goal = torch.randn(B, S, 1, 3, 8, 8)
    actions = torch.randn(B, S, H, A)
    z_goal = m.encode({"pixels": goal[:, 0].clone()})["emb"]
    goal_emb = z_goal.unsqueeze(1).expand(B, S, 1, z_goal.size(-1)).contiguous()
    info = {"pixels": pixels, "goal": goal, "goal_emb": goal_emb}
    m.set_planning_mode("official")
    c_o = m.get_cost(dict(info), actions)
    m.set_planning_mode("forward")
    info_f = dict(info)
    info_f["imagine_steps"] = torch.zeros(B, S, 1, dtype=torch.int64)
    c_f = m.get_cost(info_f, actions)
    assert c_o.shape == c_f.shape == (B, S)
    assert torch.allclose(c_o, c_f, atol=1e-6, rtol=1e-5)


def test_action_aligned_forward_returns_latent_only():
    imag = ActionAlignedCausalLatentImaginer(dim=192, hidden_dim=64, depth=1, action_dim=10)
    x = torch.randn(2, 3, 192)
    y = imag(x)
    a, z = imag.forward_with_action(x)
    assert y.shape == x.shape
    assert a.shape == (2, 3, 10)
    assert z.shape == x.shape
    assert torch.allclose(y, z)


def test_sequential_action_forward_returns_latent_only():
    imag = SequentialActionCausalLatentImaginer(
        dim=192, hidden_dim=64, depth=1, action_dim=10
    )
    x = torch.randn(2, 3, 192)
    y = imag(x)
    a, z = imag.forward_with_action(x)
    assert y.shape == x.shape
    assert a.shape == (2, 3, 10)
    assert z.shape == x.shape
    assert torch.allclose(y, z)
