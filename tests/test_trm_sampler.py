"""Focused CPU contracts for TwoRoom TRM pair sampling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.train_trm import (
    annotate_exact_pair_overlap,
    build_episode_table,
    load_excluded_episode_ids,
    load_pair_manifest,
    sample_temporal_pairs,
    save_pair_manifest,
    temporal_labels,
    validate_pair_manifest,
)


def _synthetic_rows():
    episode = np.asarray([10] * 5 + [20] * 4 + [30] * 6, dtype=np.int64)
    step = np.asarray(list(range(5)) + list(range(4)) + list(range(6)), dtype=np.int64)
    return episode, step


def test_balanced_full_delta_same_episode_and_random_order():
    episode, step = _synthetic_rows()
    table = build_episode_table(episode, step, excluded_episode_ids=[30])
    manifest = sample_temporal_pairs(table, 400, 3072, split="train")

    assert set(manifest.delta.tolist()) == {1, 2, 3, 4}
    counts = np.bincount(manifest.delta)[1:]
    assert counts.max() - counts.min() <= 1
    assert np.array_equal(episode[manifest.row_i], episode[manifest.row_j])
    assert np.array_equal(
        np.abs(step[manifest.row_i] - step[manifest.row_j]), manifest.delta
    )
    assert np.all(step[manifest.row_i[manifest.swapped]] > step[manifest.row_j[manifest.swapped]])
    assert np.all(step[manifest.row_i[~manifest.swapped]] < step[manifest.row_j[~manifest.swapped]])
    assert manifest.swapped.any() and (~manifest.swapped).any()
    assert not np.any(episode[manifest.row_i] == 30)
    assert manifest.metadata["evaluation_episode_overlap_count"] == 0
    assert manifest.metadata["endpoint_recurrence_allowed"] is True
    assert len(np.unique(np.concatenate([manifest.row_i, manifest.row_j]))) < 2 * len(
        manifest
    )
    assert np.allclose(temporal_labels(manifest), manifest.delta / 224.0)


def test_train_val_rng_are_independent_and_overlap_is_recorded():
    episode, step = _synthetic_rows()
    table = build_episode_table(episode, step, excluded_episode_ids=[30])
    train = sample_temporal_pairs(table, 400, 3072, split="train")
    val = sample_temporal_pairs(table, 120, 3073, split="validation")
    assert not np.array_equal(train.row_i[: len(val)], val.row_i)
    overlap = annotate_exact_pair_overlap(train, val)
    assert overlap > 0
    assert train.metadata["train_validation_exact_pair_overlap"] == overlap
    assert val.metadata["train_validation_exact_pair_overlap"] == overlap
    validate_pair_manifest(train, episode, step, excluded_episode_ids=[30])
    validate_pair_manifest(val, episode, step, excluded_episode_ids=[30])


def test_pair_manifest_round_trip(tmp_path):
    episode, step = _synthetic_rows()
    table = build_episode_table(episode, step, excluded_episode_ids=[30])
    manifest = sample_temporal_pairs(table, 80, 8, split="train")
    prefix = tmp_path / "pairs" / "train"
    save_pair_manifest(manifest, prefix)
    restored = load_pair_manifest(prefix)
    assert np.array_equal(restored.row_i, manifest.row_i)
    assert np.array_equal(restored.row_j, manifest.row_j)
    assert np.array_equal(restored.delta, manifest.delta)
    assert np.array_equal(restored.swapped, manifest.swapped)
    assert restored.metadata["pair_manifest_sha256"] == manifest.metadata[
        "pair_manifest_sha256"
    ]
    with pytest.raises(FileExistsError):
        save_pair_manifest(manifest, prefix)


def test_evaluation_manifest_exclusion_and_invalid_episode_steps(tmp_path):
    first = tmp_path / "group42.json"
    second = tmp_path / "group43.json"
    first.write_text(json.dumps({"episodes": [30, 31]}))
    second.write_text(json.dumps({"by_offset": {"100": {"episodes": [31, 32]}}}))
    assert load_excluded_episode_ids([first, second]) == (30, 31, 32)

    with pytest.raises(ValueError, match="duplicate"):
        build_episode_table([1, 1], [0, 0])
    with pytest.raises(ValueError, match="non-contiguous"):
        build_episode_table([1, 1], [0, 2])
