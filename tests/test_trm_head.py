"""Focused CPU contracts for the isolated TRM head."""

from __future__ import annotations

import io

import pytest
import torch

from baselines.trm import TRMHead, pair_features
from checkpoint_utils import load_trm_head


def test_pair_feature_order_is_exact():
    z_i = torch.tensor([[1.0, -2.0]])
    z_j = torch.tensor([[4.0, -5.0]])
    expected = torch.tensor([[1.0, -2.0, 4.0, -5.0, -3.0, 3.0, 3.0, 3.0]])
    assert torch.equal(pair_features(z_i, z_j), expected)


def test_head_shape_nonnegative_and_parameter_count():
    head = TRMHead(latent_dim=192)
    z_i = torch.randn(2, 3, 192)
    z_j = torch.randn(2, 3, 192)
    output = head(z_i, z_j)
    assert output.shape == (2, 3)
    assert torch.all(output >= 0)
    assert sum(parameter.numel() for parameter in head.parameters()) == 262_913


def test_head_state_dict_round_trip():
    torch.manual_seed(7)
    source = TRMHead(5)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    restored = TRMHead(5)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    z_i = torch.randn(4, 5)
    z_j = torch.randn(4, 5)
    assert torch.equal(source(z_i, z_j), restored(z_i, z_j))


def test_head_rejects_mismatched_inputs():
    head = TRMHead(4)
    with pytest.raises(ValueError, match="identical shapes"):
        head(torch.zeros(2, 4), torch.zeros(3, 4))
    with pytest.raises(ValueError, match="identical dtypes"):
        head(torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="expected latent dim"):
        head(torch.zeros(2, 3), torch.zeros(2, 3))


def test_standalone_head_artifact_round_trip(tmp_path):
    source = TRMHead(latent_dim=7)
    artifact = tmp_path / "trm.pt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "latent_dim": 7,
            "metadata": {
                "task": "tworoom",
                "label_type": "temporal_delta",
                "base_checkpoint_sha256": "abc",
            },
        },
        artifact,
    )
    restored, metadata = load_trm_head(str(artifact))
    z_i = torch.randn(4, 7)
    z_j = torch.randn(4, 7)
    assert torch.equal(source(z_i, z_j), restored(z_i, z_j))
    assert metadata["task"] == "tworoom"
    assert metadata["base_checkpoint_sha256"] == "abc"
    assert metadata["artifact_path"] == str(artifact.resolve())
