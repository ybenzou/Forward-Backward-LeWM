"""Gradient isolation: Imaginer losses must not update official modules."""

import torch
from torch import nn

from fblewm import FBLeWM
from module import CausalLatentImaginer
from tests.test_model_contracts import _tiny_model


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
