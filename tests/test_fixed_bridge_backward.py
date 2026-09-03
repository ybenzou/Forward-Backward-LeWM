"""Contracts for fixed_bridge: B(P1, z_later) with a frozen endpoint."""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from train import _fill_fixed_bridge_backward, fblewm_forward
from tests.test_model_contracts import _tiny_model


class _RecordingB(nn.Module):
    is_conditional = True

    def __init__(self, mix: float = 0.25):
        super().__init__()
        self.mix = mix
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, z_now: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        self.calls.append((z_now.detach().clone(), z_goal.detach().clone()))
        return self.mix * z_now + (1.0 - self.mix) * z_goal


def _fill(z, p, imaginer):
    dummy = type("M", (), {"model": type("N", (), {"backward_imaginer": imaginer})()})()
    output = {}
    _fill_fixed_bridge_backward(dummy, output, z, p)
    return output


def test_fixed_bridge_calls_use_the_same_p1():
    torch.manual_seed(0)
    z = torch.randn(4, 4, 8)
    p = torch.randn(4, 3, 8)
    imag = _RecordingB()
    _fill(z, p, imag)

    assert len(imag.calls) == 3
    p1, z3 = imag.calls[0]
    p1_b, z2 = imag.calls[1]
    p1_c, g2 = imag.calls[2]
    assert torch.equal(p1, p[:, 0:1])
    assert torch.equal(p1_b, p[:, 0:1])
    assert torch.equal(p1_c, p[:, 0:1])
    assert torch.equal(z3, z[:, 3:4])
    assert torch.equal(z2, z[:, 2:3])
    expected_g2 = imag.mix * p[:, 0:1] + (1.0 - imag.mix) * z[:, 3:4]
    assert torch.allclose(g2, expected_g2)


def test_fixed_bridge_step_and_roll_match_formula():
    torch.manual_seed(1)
    z = torch.randn(5, 4, 8)
    p = torch.randn(5, 3, 8)
    imag = _RecordingB(mix=0.3)
    out = _fill(z, p, imag)

    p1 = p[:, 0:1]
    z1 = z[:, 1:2]
    z2 = z[:, 2:3]
    z3 = z[:, 3:4]
    g2 = imag.mix * p1 + (1.0 - imag.mix) * z3
    g1 = imag.mix * p1 + (1.0 - imag.mix) * z2
    g1_roll = imag.mix * p1 + (1.0 - imag.mix) * g2
    step = 0.5 * ((g2 - z2).pow(2).mean() + (g1 - z1).pow(2).mean())
    roll = (g1_roll - z1).pow(2).mean()
    assert torch.allclose(out["backward_step_loss"], step)
    assert torch.allclose(out["backward_roll_loss"], roll)
    assert torch.allclose(out["_b_pred"], g2)
    assert torch.equal(out["_b_tgt"], z2)


def test_fixed_bridge_logs_copy_metrics_without_v3_keys():
    z = torch.randn(3, 4, 8)
    p = torch.randn(3, 3, 8)
    out = _fill(z, p, _RecordingB())
    assert "backward_copy_mse" in out
    assert "backward_clean_mse" in out
    assert "backward_copy_gap" in out
    for key in (
        "backward_goal_rank_loss",
        "backward_shuffle_mse",
        "backward_identity_gap",
        "backward_pred_mse",
    ):
        assert key not in out


def test_fixed_bridge_ignores_p2():
    torch.manual_seed(2)
    z = torch.randn(4, 4, 8)
    p = torch.randn(4, 3, 8)
    imag = _RecordingB()
    first = _fill(z, p.clone(), imag)
    p_mut = p.clone()
    p_mut[:, 1:2] = p_mut[:, 1:2] + 10.0
    imag.calls.clear()
    second = _fill(z, p_mut, imag)
    assert torch.equal(first["backward_step_loss"], second["backward_step_loss"])
    assert torch.equal(first["backward_roll_loss"], second["backward_roll_loss"])
    assert torch.equal(first["_b_pred"], second["_b_pred"])
    assert torch.equal(first["_b_tgt"], second["_b_tgt"])


def test_fixed_bridge_requires_conditional_imaginer():
    dummy = type(
        "M",
        (),
        {
            "model": _tiny_model(),
            "sigreg": lambda *a, **k: torch.zeros(()),
            "log_dict": lambda *a, **k: None,
        },
    )()
    cfg = OmegaConf.create(
        {
            "history_size": 3,
            "num_preds": 1,
            "loss": {
                "sigreg": {"weight": 0.0},
                "forward": {"weight": 1.0, "roll_weight": 1.0},
                "backward": {
                    "weight": 1.0,
                    "roll_weight": 1.0,
                    "target": "fixed_bridge",
                },
            },
        }
    )
    batch = {
        "pixels": torch.randn(2, 4, 3, 8, 8),
        "action": torch.randn(2, 4, 10),
    }
    with pytest.raises(ValueError, match="fixed_bridge requires ConditionalLatentImaginer"):
        fblewm_forward(dummy, batch, "train", cfg)
