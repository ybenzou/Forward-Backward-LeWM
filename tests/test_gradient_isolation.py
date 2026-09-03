"""Gradient isolation: Imaginer losses must not update official modules."""

import torch
from torch import nn

from fblewm import FBLeWM
from module import CausalLatentImaginer
from tests.test_model_contracts import (
    _tiny_action_aligned_model,
    _tiny_branch_model,
    _tiny_model,
    _tiny_sequential_model,
)


def _nonzero_grads(module: nn.Module) -> bool:
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return False
    return any(g.abs().sum().item() > 0 for g in grads)


def _has_any_grad(module: nn.Module) -> bool:
    return any(p.grad is not None for p in module.parameters())


def test_forward_loss_updates_only_forward():
    m = _tiny_model()
    m.zero_grad(set_to_none=True)
    z = torch.randn(4, 192, requires_grad=False)
    p = torch.randn(4, 192)
    # create a graph through forward imaginer only
    pred = m.forward_imaginer(p)
    loss = (pred - z).pow(2).mean()
    loss.backward()
    assert _nonzero_grads(m.forward_imaginer)
    assert not _has_any_grad(m.encoder)
    assert not _has_any_grad(m.predictor)
    assert not _has_any_grad(m.action_encoder)
    assert not _has_any_grad(m.backward_imaginer)


def test_backward_loss_updates_only_backward():
    m = _tiny_model()
    m.zero_grad(set_to_none=True)
    z = torch.randn(4, 192)
    pred = m.backward_imaginer(z)
    loss = (pred - z).pow(2).mean()
    loss.backward()
    assert _nonzero_grads(m.backward_imaginer)
    assert not _has_any_grad(m.encoder)
    assert not _has_any_grad(m.predictor)
    assert not _has_any_grad(m.action_encoder)
    assert not _has_any_grad(m.forward_imaginer)


def test_official_loss_does_not_update_imaginers():
    m = _tiny_model()
    m.zero_grad(set_to_none=True)
    B, T, D = 2, 4, 192
    pixels = torch.randn(B, T, 3, 8, 8)
    action = torch.randn(B, T, 10)
    out = m.encode({"pixels": pixels, "action": action})
    emb = out["emb"]
    act = out["act_emb"]
    pred = m.predict(emb[:, :3], act[:, :3])
    loss = (pred - emb[:, 1:]).pow(2).mean()
    loss.backward()
    assert _nonzero_grads(m.encoder) or _nonzero_grads(m.predictor) or _nonzero_grads(
        m.action_encoder
    )
    assert not _has_any_grad(m.forward_imaginer)
    assert not _has_any_grad(m.backward_imaginer)


def test_detached_imaginer_inputs_block_official_grads():
    """Mirrors train.py: detach before imaginer."""
    m = _tiny_model()
    m.zero_grad(set_to_none=True)
    B, T, D = 2, 4, 192
    pixels = torch.randn(B, T, 3, 8, 8)
    action = torch.randn(B, T, 10)
    out = m.encode({"pixels": pixels, "action": action})
    emb = out["emb"]
    act = out["act_emb"]
    pred = m.predict(emb[:, :3], act[:, :3])
    # imaginer on detached tensors
    f_loss = (m.forward_imaginer(pred.detach()[:, 0:2]) - emb.detach()[:, 2:4]).pow(2).mean()
    f_loss.backward()
    assert _nonzero_grads(m.forward_imaginer)
    assert not _has_any_grad(m.encoder)
    assert not _has_any_grad(m.predictor)
    assert not _has_any_grad(m.action_encoder)
    assert not _has_any_grad(m.backward_imaginer)


def test_action_aligned_loss_updates_only_forward():
    """Action+latent Forward loss must stay isolated from official modules."""
    m = _tiny_action_aligned_model()
    m.zero_grad(set_to_none=True)
    p = torch.randn(4, 192)
    z = torch.randn(4, 192)
    a_tgt = torch.randn(4, 10)
    a_hat, z_hat = m.forward_imaginer.forward_with_action(p)
    loss = (z_hat - z).pow(2).mean() + (a_hat - a_tgt).pow(2).mean()
    loss.backward()
    assert _nonzero_grads(m.forward_imaginer)
    assert not _has_any_grad(m.encoder)
    assert not _has_any_grad(m.predictor)
    assert not _has_any_grad(m.action_encoder)
    assert not _has_any_grad(m.backward_imaginer)


def test_sequential_action_autonomous_loss_updates_proposer():
    """H(p, G(p)) must send grad to G even without an action MSE term."""
    m = _tiny_sequential_model()
    m.zero_grad(set_to_none=True)
    p = torch.randn(4, 192)
    z = torch.randn(4, 192)
    a_hat, z_hat = m.forward_imaginer.forward_with_action(p)
    loss = (z_hat - z).pow(2).mean()
    loss.backward()
    assert _nonzero_grads(m.forward_imaginer)
    assert _nonzero_grads(m.forward_imaginer.action_head)
    assert _nonzero_grads(m.forward_imaginer.action_embed)
    assert not _has_any_grad(m.encoder)
    assert not _has_any_grad(m.predictor)
    assert not _has_any_grad(m.action_encoder)
    assert not _has_any_grad(m.backward_imaginer)


def test_branch_preserving_loss_updates_only_forward():
    from omegaconf import OmegaConf

    from train import _fill_branch_preserving_forward_losses

    m = _tiny_branch_model(dim=16, num_branches=3)
    m.zero_grad(set_to_none=True)
    z = torch.randn(4, 4, 16)
    p = torch.randn(4, 3, 16)
    cfg = OmegaConf.create({"loss": {"forward": {"roll_weight": 1.0}}})
    output = {}
    holder = type("H", (), {"model": m})()
    _fill_branch_preserving_forward_losses(holder, output, z, p, cfg)
    output["forward_loss"].backward()
    assert _nonzero_grads(m.forward_imaginer)
    assert not _has_any_grad(m.encoder)
    assert not _has_any_grad(m.predictor)
    assert not _has_any_grad(m.action_encoder)
    assert not _has_any_grad(m.backward_imaginer)
