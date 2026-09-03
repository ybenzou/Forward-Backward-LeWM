"""Contracts for Action-aligned autonomous Forward rollout."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from module import ActionAlignedCausalLatentImaginer
from tests.test_gradient_isolation import _has_any_grad, _nonzero_grads
from tests.test_model_contracts import _tiny_action_aligned_model, _tiny_model
from train import _fill_forward_losses, _forward_variant


class _Holder:
    def __init__(self, model):
        self.model = model


def _cfg(variant="action_aligned", action_weight=1.0, roll_weight=1.0):
    return OmegaConf.create(
        {
            "loss": {
                "forward": {
                    "variant": variant,
                    "roll_weight": roll_weight,
                    "action_weight": action_weight,
                }
            }
        }
    )


def test_forward_with_action_shapes():
    imag = ActionAlignedCausalLatentImaginer(dim=16, hidden_dim=32, depth=1, action_dim=10)
    z1 = torch.randn(4, 16)
    a1, y1 = imag.forward_with_action(z1)
    assert a1.shape == (4, 10)
    assert y1.shape == (4, 16)
    z2 = torch.randn(3, 5, 16)
    a2, y2 = imag.forward_with_action(z2)
    assert a2.shape == (3, 5, 10)
    assert y2.shape == (3, 5, 16)


def test_forward_matches_latent_head():
    imag = ActionAlignedCausalLatentImaginer(dim=16, hidden_dim=32, depth=1, action_dim=6)
    z = torch.randn(2, 4, 16)
    assert torch.allclose(imag(z), imag.forward_with_action(z)[1])


def test_imagine_forward_returns_latent_only():
    m = _tiny_action_aligned_model(dim=32, action_dim=10)
    z = torch.randn(3, 32)
    out0 = m.imagine_forward(z, 0)
    out1 = m.imagine_forward(z, 1)
    out3 = m.imagine_forward(z, 3)
    assert out0.shape == z.shape and torch.equal(out0, z)
    assert out1.shape == z.shape
    assert out3.shape == z.shape
    assert out1.dtype == z.dtype


def test_four_frame_action_index_contract():
    B, T, D, A = 2, 4, 8, 10
    action = torch.arange(B * T * A, dtype=torch.float32).reshape(B, T, A)
    assert torch.equal(action[:, 1:3], torch.stack([action[:, 1], action[:, 2]], dim=1))
    assert torch.equal(action[:, 2:3], action[:, 2:3])
    # Sentinel: first block is NOT the step target.
    assert not torch.equal(action[:, 0:2], action[:, 1:3])


def test_fill_forward_uses_action_1_3_and_roll_2():
    torch.manual_seed(0)
    dim, action_dim = 16, 10
    m = _tiny_action_aligned_model(dim=dim, action_dim=action_dim)
    holder = _Holder(m)
    z = torch.randn(3, 4, dim)
    p = torch.randn(3, 3, dim)
    action = torch.arange(3 * 4 * action_dim, dtype=torch.float32).reshape(3, 4, action_dim)
    output = {}
    _fill_forward_losses(holder, output, z, p, action, _cfg())

    imag = m.forward_imaginer
    a_step, z_step = imag.forward_with_action(p[:, 0:2])
    _, z_roll_1 = imag.forward_with_action(p[:, 0:1])
    a_roll_2, z_roll_2 = imag.forward_with_action(z_roll_1)
    expect_step = (a_step - action[:, 1:3]).pow(2).mean()
    expect_roll = (a_roll_2 - action[:, 2:3]).pow(2).mean()
    assert torch.allclose(output["forward_action_step_loss"], expect_step)
    assert torch.allclose(output["forward_action_roll_loss"], expect_roll)
    assert torch.allclose(output["forward_step_loss"], (z_step - z[:, 2:4]).pow(2).mean())
    assert torch.allclose(output["forward_roll_loss"], (z_roll_2 - z[:, 3:4]).pow(2).mean())


def test_action_aligned_fill_isolates_official_grads():
    dim, action_dim = 16, 10
    m = _tiny_action_aligned_model(dim=dim, action_dim=action_dim)
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


def test_latent_variant_has_no_action_loss_keys():
    holder = _Holder(_tiny_model(dim=16))
    z = torch.randn(2, 4, 16)
    p = torch.randn(2, 3, 16)
    output = {}
    _fill_forward_losses(holder, output, z, p, torch.randn(2, 4, 10), _cfg("latent"))
    assert "forward_action_step_loss" not in output
    assert "forward_action_loss" not in output
    alpha = 1.0
    expect = output["forward_step_loss"] + alpha * output["forward_roll_loss"]
    assert torch.allclose(output["forward_loss"], expect)


def test_action_weight_zero_matches_latent_formula():
    dim, action_dim = 16, 10
    m = _tiny_action_aligned_model(dim=dim, action_dim=action_dim)
    holder = _Holder(m)
    z = torch.randn(2, 4, dim)
    p = torch.randn(2, 3, dim)
    action = torch.randn(2, 4, action_dim)
    output = {}
    _fill_forward_losses(holder, output, z, p, action, _cfg(action_weight=0.0))
    latent_only = output["forward_step_loss"] + output["forward_roll_loss"]
    assert torch.allclose(output["forward_loss"], latent_only)
    assert "forward_action_loss" in output


def test_illegal_variant_raises():
    try:
        _forward_variant(OmegaConf.create({"loss": {"forward": {"variant": "policy"}}}))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "action_aligned" in str(exc)


def test_wrong_action_dim_raises():
    m = _tiny_action_aligned_model(dim=16, action_dim=10)
    holder = _Holder(m)
    output = {}
    try:
        _fill_forward_losses(
            holder,
            output,
            torch.randn(2, 4, 16),
            torch.randn(2, 3, 16),
            torch.randn(2, 4, 7),
            _cfg(),
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "action_dim=10" in str(exc)


def test_too_few_latent_frames_raises():
    m = _tiny_action_aligned_model(dim=16, action_dim=10)
    holder = _Holder(m)
    output = {}
    try:
        _fill_forward_losses(
            holder,
            output,
            torch.randn(2, 3, 16),
            torch.randn(2, 3, 16),
            torch.randn(2, 4, 10),
            _cfg(),
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "4 latent frames" in str(exc)
