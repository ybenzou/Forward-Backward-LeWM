"""Contracts for Sequential Action-Aligned Forward (A=G(p), z'=H(p,A))."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from module import SequentialActionCausalLatentImaginer
from tests.test_gradient_isolation import _has_any_grad, _nonzero_grads
from tests.test_model_contracts import (
    _tiny_action_aligned_model,
    _tiny_sequential_model,
    _tiny_model,
)
from train import _fill_forward_losses, _forward_variant


class _Holder:
    def __init__(self, model):
        self.model = model


def _cfg(
    variant="sequential_action",
    action_weight=1.0,
    roll_weight=1.0,
    teacher_weight=1.0,
):
    return OmegaConf.create(
        {
            "loss": {
                "forward": {
                    "variant": variant,
                    "roll_weight": roll_weight,
                    "action_weight": action_weight,
                    "teacher_weight": teacher_weight,
                }
            }
        }
    )


def test_sequential_api_shapes_bd_and_btd():
    imag = SequentialActionCausalLatentImaginer(
        dim=16, hidden_dim=32, depth=1, action_dim=10
    )
    z1 = torch.randn(4, 16)
    a1 = imag.predict_action(z1)
    y1 = imag.transition(z1, a1)
    a1b, y1b = imag.forward_with_action(z1)
    assert a1.shape == (4, 10) and y1.shape == (4, 16)
    assert a1b.shape == (4, 10) and y1b.shape == (4, 16)
    z2 = torch.randn(3, 5, 16)
    a2 = imag.predict_action(z2)
    y2 = imag.transition(z2, a2)
    a2b, y2b = imag.forward_with_action(z2)
    assert a2.shape == (3, 5, 10) and y2.shape == (3, 5, 16)
    assert a2b.shape == (3, 5, 10) and y2b.shape == (3, 5, 16)
    assert imag(z2).shape == z2.shape


def test_forward_matches_latent_head():
    imag = SequentialActionCausalLatentImaginer(
        dim=16, hidden_dim=32, depth=1, action_dim=6
    )
    z = torch.randn(2, 4, 16)
    assert torch.allclose(imag(z), imag.forward_with_action(z)[1])


def test_transition_depends_on_action():
    imag = SequentialActionCausalLatentImaginer(
        dim=16, hidden_dim=32, depth=1, action_dim=10
    )
    z = torch.randn(3, 16)
    a = imag.predict_action(z)
    y0 = imag.transition(z, a)
    y1 = imag.transition(z, a + 1.0)
    assert y0.shape == z.shape
    assert not torch.allclose(y0, y1)


def test_teacher_forced_is_transition_alias():
    imag = SequentialActionCausalLatentImaginer(
        dim=16, hidden_dim=32, depth=1, action_dim=10
    )
    z = torch.randn(2, 3, 16)
    a = torch.randn(2, 3, 10)
    assert torch.allclose(imag.forward_teacher_forced(z, a), imag.transition(z, a))


def test_imagine_forward_returns_latent_only():
    m = _tiny_sequential_model(dim=32, action_dim=10)
    z = torch.randn(3, 32)
    out0 = m.imagine_forward(z, 0)
    out1 = m.imagine_forward(z, 1)
    out3 = m.imagine_forward(z, 3)
    out15 = m.imagine_forward(z, 15)
    assert out0.shape == z.shape and torch.equal(out0, z)
    assert out1.shape == z.shape
    assert out3.shape == z.shape
    assert out15.shape == z.shape
    assert out1.dtype == z.dtype


def test_four_frame_action_index_contract():
    B, T, D, A = 2, 4, 8, 10
    action = torch.arange(B * T * A, dtype=torch.float32).reshape(B, T, A)
    assert torch.equal(action[:, 1:3], torch.stack([action[:, 1], action[:, 2]], dim=1))
    assert torch.equal(action[:, 2:3], action[:, 2:3])
    assert not torch.equal(action[:, 0:2], action[:, 1:3])


def test_fill_uses_action_1_3_and_roll_2():
    torch.manual_seed(0)
    dim, action_dim = 16, 10
    m = _tiny_sequential_model(dim=dim, action_dim=action_dim)
    holder = _Holder(m)
    z = torch.randn(3, 4, dim)
    p = torch.randn(3, 3, dim)
    action = torch.arange(3 * 4 * action_dim, dtype=torch.float32).reshape(
        3, 4, action_dim
    )
    output = {}
    _fill_forward_losses(holder, output, z, p, action, _cfg())

    imag = m.forward_imaginer
    x = p[:, 0:2]
    a_step = imag.predict_action(x)
    z_teacher = imag.forward_teacher_forced(x, action[:, 1:3])
    z_auto = imag.transition(x, a_step)
    a1, z2 = imag.forward_with_action(p[:, 0:1])
    a2, z3 = imag.forward_with_action(z2)
    assert torch.allclose(
        output["forward_action_step_loss"], (a_step - action[:, 1:3]).pow(2).mean()
    )
    assert torch.allclose(
        output["forward_action_roll_loss"], (a2 - action[:, 2:3]).pow(2).mean()
    )
    assert torch.allclose(output["forward_teacher_loss"], (z_teacher - z[:, 2:4]).pow(2).mean())
    assert torch.allclose(output["forward_auto_step_loss"], (z_auto - z[:, 2:4]).pow(2).mean())
    assert torch.allclose(output["forward_roll_loss"], (z3 - z[:, 3:4]).pow(2).mean())


def test_teacher_uses_target_action_auto_uses_predicted():
    torch.manual_seed(1)
    dim, action_dim = 16, 10
    m = _tiny_sequential_model(dim=dim, action_dim=action_dim)
    holder = _Holder(m)
    z = torch.randn(2, 4, dim)
    p = torch.randn(2, 3, dim)
    action = torch.randn(2, 4, action_dim)
    output = {}
    _fill_forward_losses(holder, output, z, p, action, _cfg())

    action2 = action.clone()
    action2[:, 1:3] = action2[:, 1:3] + 10.0
    output2 = {}
    _fill_forward_losses(holder, output2, z, p, action2, _cfg())
    assert not torch.allclose(output2["forward_teacher_loss"], output["forward_teacher_loss"])
    assert torch.allclose(output2["forward_auto_step_loss"], output["forward_auto_step_loss"])


def test_two_step_roll_equals_manual_g_h():
    torch.manual_seed(2)
    imag = SequentialActionCausalLatentImaginer(
        dim=16, hidden_dim=32, depth=1, action_dim=10
    )
    p0 = torch.randn(3, 1, 16)
    a1 = imag.predict_action(p0)
    z2 = imag.transition(p0, a1)
    a2 = imag.predict_action(z2)
    z3 = imag.transition(z2, a2)
    a1b, z2b = imag.forward_with_action(p0)
    a2b, z3b = imag.forward_with_action(z2b)
    assert torch.allclose(a1, a1b)
    assert torch.allclose(z2, z2b)
    assert torch.allclose(a2, a2b)
    assert torch.allclose(z3, z3b)


def test_fill_isolates_official_grads():
    dim, action_dim = 16, 10
    m = _tiny_sequential_model(dim=dim, action_dim=action_dim)
    m.zero_grad(set_to_none=True)
    holder = _Holder(m)
    z = torch.randn(2, 4, dim)
    p = torch.randn(2, 3, dim)
    action = torch.randn(2, 4, action_dim)
    output = {}
    _fill_forward_losses(holder, output, z, p, action, _cfg())
    output["forward_loss"].backward()
    assert _nonzero_grads(m.forward_imaginer)
    assert not _has_any_grad(m.encoder)
    assert not _has_any_grad(m.predictor)
    assert not _has_any_grad(m.action_encoder)
    assert not _has_any_grad(m.backward_imaginer)


def test_action_weight_zero_still_grads_proposer():
    dim, action_dim = 16, 10
    m = _tiny_sequential_model(dim=dim, action_dim=action_dim)
    m.zero_grad(set_to_none=True)
    holder = _Holder(m)
    z = torch.randn(2, 4, dim)
    p = torch.randn(2, 3, dim)
    action = torch.randn(2, 4, action_dim)
    output = {}
    _fill_forward_losses(holder, output, z, p, action, _cfg(action_weight=0.0))
    output["forward_loss"].backward()
    assert _nonzero_grads(m.forward_imaginer.action_head)
    assert _nonzero_grads(m.forward_imaginer.action_embed)
    assert not _has_any_grad(m.encoder)
    assert not _has_any_grad(m.backward_imaginer)


def test_weight_formula_and_key_names():
    torch.manual_seed(3)
    dim, action_dim = 16, 10
    m = _tiny_sequential_model(dim=dim, action_dim=action_dim)
    holder = _Holder(m)
    z = torch.randn(2, 4, dim)
    p = torch.randn(2, 3, dim)
    action = torch.randn(2, 4, action_dim)
    output = {}
    cfg = _cfg(teacher_weight=0.5, roll_weight=2.0, action_weight=0.25)
    _fill_forward_losses(holder, output, z, p, action, cfg)
    for key in (
        "forward_teacher_loss",
        "forward_auto_step_loss",
        "forward_roll_loss",
        "forward_action_step_loss",
        "forward_action_roll_loss",
        "forward_action_loss",
        "forward_teacher_auto_gap",
        "forward_loss",
    ):
        assert key in output
    assert "forward_step_loss" not in output
    latent = (
        0.5 * output["forward_teacher_loss"]
        + output["forward_auto_step_loss"]
        + 2.0 * output["forward_roll_loss"]
    )
    action_loss = (
        output["forward_action_step_loss"] + 2.0 * output["forward_action_roll_loss"]
    )
    expect = latent + 0.25 * action_loss
    assert torch.allclose(output["forward_action_loss"], action_loss)
    assert torch.allclose(output["forward_loss"], expect)
    assert torch.allclose(
        output["forward_teacher_auto_gap"],
        output["forward_auto_step_loss"] - output["forward_teacher_loss"],
    )


def test_val_mean_action_baseline_uses_train_mean():
    dim, action_dim = 16, 10
    m = _tiny_sequential_model(dim=dim, action_dim=action_dim)
    train_mean = torch.linspace(-1.0, 1.0, action_dim)
    m.register_buffer("train_action_mean", train_mean, persistent=False)
    holder = _Holder(m)
    z = torch.randn(2, 4, dim)
    p = torch.randn(2, 3, dim)
    action = torch.randn(2, 4, action_dim)
    train_out = {}
    _fill_forward_losses(holder, train_out, z, p, action, _cfg(), stage="train")
    assert "forward_action_mean_baseline_mse" not in train_out

    val_out = {}
    _fill_forward_losses(holder, val_out, z, p, action, _cfg(), stage="val")
    a_tgt = action[:, 1:3]
    baseline = (train_mean.reshape(1, 1, -1) - a_tgt).pow(2).mean()
    skill = 1.0 - val_out["forward_action_step_loss"].detach() / baseline.clamp(min=1e-8)
    assert torch.allclose(val_out["forward_action_mean_baseline_mse"], baseline)
    assert torch.allclose(val_out["forward_action_skill"], skill)


def test_dynamic_action_dims_pusht_and_cube():
    for action_dim in (10, 25):
        imag = SequentialActionCausalLatentImaginer(
            dim=16, hidden_dim=32, depth=1, action_dim=action_dim
        )
        z = torch.randn(2, 3, 16)
        a = imag.predict_action(z)
        y = imag.transition(z, a)
        assert a.shape == (2, 3, action_dim)
        assert y.shape == z.shape


def test_illegal_variant_mentions_sequential():
    try:
        _forward_variant(OmegaConf.create({"loss": {"forward": {"variant": "policy"}}}))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "sequential_action" in str(exc)
        assert "action_aligned" in str(exc)


def test_wrong_action_dim_raises():
    m = _tiny_sequential_model(dim=16, action_dim=10)
    holder = _Holder(m)
    try:
        _fill_forward_losses(
            holder,
            {},
            torch.randn(2, 4, 16),
            torch.randn(2, 3, 16),
            torch.randn(2, 4, 7),
            _cfg(),
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "action_dim=10" in str(exc)


def test_too_few_latent_frames_raises():
    m = _tiny_sequential_model(dim=16, action_dim=10)
    holder = _Holder(m)
    try:
        _fill_forward_losses(
            holder,
            {},
            torch.randn(2, 3, 16),
            torch.randn(2, 3, 16),
            torch.randn(2, 4, 10),
            _cfg(),
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "4 latent frames" in str(exc)


def test_action_leading_shape_raises():
    imag = SequentialActionCausalLatentImaginer(
        dim=16, hidden_dim=32, depth=1, action_dim=10
    )
    z = torch.randn(2, 4, 16)
    a = torch.randn(2, 10)
    try:
        imag.transition(z, a)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "leading shape" in str(exc)


def test_wrong_class_for_sequential_variant_raises():
    holder = _Holder(_tiny_model(dim=16))
    try:
        _fill_forward_losses(
            holder,
            {},
            torch.randn(2, 4, 16),
            torch.randn(2, 3, 16),
            torch.randn(2, 4, 10),
            _cfg(),
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "SequentialActionCausalLatentImaginer" in str(exc)


def test_parallel_aaf_is_unchanged_under_action_aligned_variant():
    """Old parallel AAF path must not pick up sequential keys."""
    m = _tiny_action_aligned_model(dim=16, action_dim=10)
    holder = _Holder(m)
    output = {}
    _fill_forward_losses(
        holder,
        output,
        torch.randn(2, 4, 16),
        torch.randn(2, 3, 16),
        torch.randn(2, 4, 10),
        _cfg(variant="action_aligned"),
    )
    assert "forward_step_loss" in output
    assert "forward_teacher_loss" not in output
    assert "forward_auto_step_loss" not in output


def test_checkpoint_missing_sequential_params_rejected():
    dim, action_dim = 16, 10
    m = _tiny_sequential_model(dim=dim, action_dim=action_dim)
    state = m.state_dict()
    for token in ("action_head", "action_embed", "fuse", "transition_blocks"):
        assert any(token in k for k in state), token

    for token in ("action_head", "action_embed", "fuse", "transition_blocks"):
        fresh = _tiny_sequential_model(dim=dim, action_dim=action_dim)
        bad = {k: v for k, v in state.items() if token not in k}
        missing, _unexpected = fresh.load_state_dict(bad, strict=False)
        missing_imag = [x for x in missing if "imaginer" in x]
        assert missing_imag, token
