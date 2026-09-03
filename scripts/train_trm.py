#!/usr/bin/env python3
"""Build TwoRoom TRM pairs/latents and train an isolated pairwise head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.trm import TRMHead


DEFAULT_OUTPUT = ROOT / "outputs" / "baselines" / "trm_compare" / "v1"
LABEL_SCALE = 224.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EpisodeTable:
    """Ordered global row ids for each retained episode."""

    rows_by_episode: Mapping[int, np.ndarray]
    steps_by_episode: Mapping[int, np.ndarray]
    excluded_episode_ids: tuple[int, ...]
    total_episode_count: int

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.rows_by_episode))

    @property
    def max_delta(self) -> int:
        return max((len(v) - 1 for v in self.rows_by_episode.values()), default=0)


def build_episode_table(
    episode_idx: Sequence[int] | np.ndarray,
    step_idx: Sequence[int] | np.ndarray,
    excluded_episode_ids: Iterable[int] = (),
) -> EpisodeTable:
    """Build episode row tables, asserting unique contiguous steps."""
    episodes = np.asarray(episode_idx)
    steps = np.asarray(step_idx)
    if episodes.ndim != 1 or steps.ndim != 1 or episodes.shape != steps.shape:
        raise ValueError("episode_idx and step_idx must be same-length 1-D arrays")
    if not np.issubdtype(episodes.dtype, np.integer):
        raise TypeError("episode ids must be integer-valued")
    if not np.issubdtype(steps.dtype, np.integer):
        raise TypeError("step ids must be integer-valued")

    excluded = frozenset(int(v) for v in excluded_episode_ids)
    rows_by_episode: dict[int, np.ndarray] = {}
    steps_by_episode: dict[int, np.ndarray] = {}
    row_order = np.lexsort((steps, episodes))
    sorted_episodes = episodes[row_order]
    group_starts = np.r_[
        0, np.flatnonzero(sorted_episodes[1:] != sorted_episodes[:-1]) + 1
    ]
    group_ends = np.r_[group_starts[1:], len(row_order)]
    unique_episodes = sorted_episodes[group_starts]
    for raw_episode, group_start, group_end in zip(
        unique_episodes, group_starts, group_ends
    ):
        episode = int(raw_episode)
        row_ids = row_order[group_start:group_end].astype(np.int64, copy=False)
        episode_steps = steps[row_ids].astype(np.int64, copy=False)
        if len(np.unique(episode_steps)) != len(episode_steps):
            raise ValueError(f"episode {episode} has duplicate step_idx values")
        if len(episode_steps) > 1 and not np.all(np.diff(episode_steps) == 1):
            raise ValueError(f"episode {episode} has non-contiguous step_idx values")
        if episode in excluded:
            continue
        rows_by_episode[episode] = row_ids
        steps_by_episode[episode] = episode_steps

    if not rows_by_episode:
        raise ValueError("no training episodes remain after evaluation exclusion")
    return EpisodeTable(
        rows_by_episode=rows_by_episode,
        steps_by_episode=steps_by_episode,
        excluded_episode_ids=tuple(sorted(excluded)),
        total_episode_count=int(len(unique_episodes)),
    )


@dataclass
class PairManifest:
    row_i: np.ndarray
    row_j: np.ndarray
    delta: np.ndarray
    swapped: np.ndarray
    episode_id: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.row_i = np.asarray(self.row_i, dtype=np.int64)
        self.row_j = np.asarray(self.row_j, dtype=np.int64)
        self.delta = np.asarray(self.delta, dtype=np.int32)
        self.swapped = np.asarray(self.swapped, dtype=np.bool_)
        self.episode_id = np.asarray(self.episode_id, dtype=np.int64)
        sizes = {
            len(self.row_i),
            len(self.row_j),
            len(self.delta),
            len(self.swapped),
            len(self.episode_id),
        }
        if len(sizes) != 1:
            raise ValueError("pair manifest arrays must have equal lengths")

    def __len__(self) -> int:
        return len(self.row_i)


def temporal_labels(
    manifest: PairManifest, scale: float = LABEL_SCALE
) -> np.ndarray:
    """Return TwoRoom temporal targets in full-horizon units."""
    scale = float(scale)
    if scale <= 0:
        raise ValueError("label scale must be positive")
    return manifest.delta.astype(np.float32) / scale


def _pair_digest(manifest: PairManifest) -> str:
    digest = hashlib.sha256()
    for array in (
        manifest.row_i,
        manifest.row_j,
        manifest.delta,
        manifest.swapped,
        manifest.episode_id,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    metadata = {
        k: v for k, v in manifest.metadata.items() if k != "pair_manifest_sha256"
    }
    digest.update(_canonical_json(metadata))
    return digest.hexdigest()


def sample_temporal_pairs(
    table: EpisodeTable,
    pair_count: int,
    seed: int,
    *,
    split: str,
    max_delta: int | None = None,
    label_scale: float = LABEL_SCALE,
) -> PairManifest:
    """Sample balanced-full-delta, same-episode ordered temporal pairs."""
    pair_count = int(pair_count)
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    available_max = table.max_delta
    if available_max < 1:
        raise ValueError("at least one episode must contain two rows")
    if max_delta is None:
        max_delta = available_max
    max_delta = int(max_delta)
    if max_delta < 1 or max_delta > available_max:
        raise ValueError(
            f"max_delta must be in [1, {available_max}], got {max_delta}"
        )
    if pair_count < max_delta:
        raise ValueError(
            f"balanced full-delta sampling needs pair_count >= {max_delta}, "
            f"got {pair_count}"
        )

    deltas_available = np.arange(1, max_delta + 1, dtype=np.int32)
    base, remainder = divmod(pair_count, len(deltas_available))
    delta_counts = np.full(len(deltas_available), base, dtype=np.int64)
    delta_counts[:remainder] += 1
    deltas = np.repeat(deltas_available, delta_counts).astype(np.int32, copy=False)

    rng = np.random.default_rng(int(seed))
    rng.shuffle(deltas)
    row_start = np.empty(pair_count, dtype=np.int64)
    row_end = np.empty(pair_count, dtype=np.int64)
    pair_episodes = np.empty(pair_count, dtype=np.int64)

    lengths = {episode: len(rows) for episode, rows in table.rows_by_episode.items()}
    for delta in deltas_available:
        positions = np.flatnonzero(deltas == delta)
        eligible = np.asarray(
            [episode for episode in table.episode_ids if lengths[episode] > delta],
            dtype=np.int64,
        )
        if not len(eligible):
            raise RuntimeError(f"no episode can realize delta={int(delta)}")
        chosen = eligible[rng.integers(0, len(eligible), size=len(positions))]
        pair_episodes[positions] = chosen
        for output_pos, episode in zip(positions, chosen):
            rows = table.rows_by_episode[int(episode)]
            t = int(rng.integers(0, len(rows) - int(delta)))
            row_start[output_pos] = rows[t]
            row_end[output_pos] = rows[t + int(delta)]

    swapped = rng.random(pair_count) < 0.5
    row_i = np.where(swapped, row_end, row_start)
    row_j = np.where(swapped, row_start, row_end)
    actual_counts = np.bincount(deltas, minlength=max_delta + 1)[1:]
    retained = set(table.episode_ids)
    evaluation_overlap = retained.intersection(table.excluded_episode_ids)
    if evaluation_overlap:
        raise AssertionError("evaluation episodes survived exclusion")

    metadata: dict[str, Any] = {
        "format_version": 1,
        "task": "tworoom",
        "split": str(split),
        "sampler": "balanced_full_delta_same_episode",
        "seed": int(seed),
        "pair_count": pair_count,
        "delta_min": 1,
        "delta_max": max_delta,
        "delta_counts": {
            str(delta): int(count)
            for delta, count in zip(deltas_available, actual_counts)
        },
        "pair_order": "independent Bernoulli(0.5) endpoint swap",
        "swapped_count": int(swapped.sum()),
        "swapped_fraction": float(swapped.mean()),
        "same_episode": True,
        "endpoint_recurrence_allowed": True,
        "exact_pair_deduplication": False,
        "label_type": "temporal_delta",
        "label_formula": "target = delta / 224.0",
        "label_scale": float(label_scale),
        "retained_episode_count": len(table.episode_ids),
        "dataset_episode_count": table.total_episode_count,
        "excluded_episode_ids": list(table.excluded_episode_ids),
        "excluded_episode_count": len(table.excluded_episode_ids),
        "evaluation_episode_overlap_count": 0,
    }
    return PairManifest(row_i, row_j, deltas, swapped, pair_episodes, metadata)


def _canonical_pair_set(manifest: PairManifest) -> set[tuple[int, int]]:
    return {
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in zip(manifest.row_i, manifest.row_j)
    }


def annotate_exact_pair_overlap(
    train_manifest: PairManifest, val_manifest: PairManifest
) -> int:
    """Record unordered exact endpoint-pair overlap in both manifests."""
    overlap = len(
        _canonical_pair_set(train_manifest).intersection(
            _canonical_pair_set(val_manifest)
        )
    )
    train_manifest.metadata["train_validation_exact_pair_overlap"] = overlap
    val_manifest.metadata["train_validation_exact_pair_overlap"] = overlap
    return overlap


def _manifest_paths(prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(prefix)
    if prefix.suffix in {".npz", ".json"}:
        prefix = prefix.with_suffix("")
    return prefix.with_suffix(".npz"), prefix.with_suffix(".json")


def save_pair_manifest(
    manifest: PairManifest, prefix: str | Path, *, force: bool = False
) -> tuple[Path, Path]:
    npz_path, json_path = _manifest_paths(prefix)
    if not force and (npz_path.exists() or json_path.exists()):
        raise FileExistsError(f"pair manifest already exists: {prefix}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.metadata["pair_manifest_sha256"] = _pair_digest(manifest)
    with npz_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            row_i=manifest.row_i,
            row_j=manifest.row_j,
            delta=manifest.delta,
            swapped=manifest.swapped,
            episode_id=manifest.episode_id,
        )
    json_path.write_text(
        json.dumps(_jsonable(manifest.metadata), indent=2, sort_keys=True) + "\n"
    )
    return npz_path, json_path


def load_pair_manifest(prefix: str | Path) -> PairManifest:
    npz_path, json_path = _manifest_paths(prefix)
    metadata = json.loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as arrays:
        manifest = PairManifest(
            arrays["row_i"],
            arrays["row_j"],
            arrays["delta"],
            arrays["swapped"],
            arrays["episode_id"],
            metadata,
        )
    expected = metadata.get("pair_manifest_sha256")
    actual = _pair_digest(manifest)
    if expected != actual:
        raise ValueError(
            f"pair manifest hash mismatch for {prefix}: expected={expected} actual={actual}"
        )
    return manifest


def _collect_episode_values(value: Any, output: set[int]) -> None:
    if isinstance(value, Mapping):
        episodes = value.get("episodes")
        if isinstance(episodes, (list, tuple)):
            output.update(int(v) for v in episodes)
        for child in value.values():
            _collect_episode_values(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_episode_values(child, output)


def load_excluded_episode_ids(paths: Iterable[str | Path]) -> tuple[int, ...]:
    """Collect source episode ids from one or more evaluation JSON manifests."""
    episodes: set[int] = set()
    for path in paths:
        payload = json.loads(Path(path).read_text())
        _collect_episode_values(payload, episodes)
    return tuple(sorted(episodes))


def validate_pair_manifest(
    manifest: PairManifest,
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    excluded_episode_ids: Iterable[int] = (),
) -> None:
    """Reject crossing, wrong deltas, invalid rows, or excluded episodes."""
    n_rows = len(episode_idx)
    if (
        np.any(manifest.row_i < 0)
        or np.any(manifest.row_j < 0)
        or np.any(manifest.row_i >= n_rows)
        or np.any(manifest.row_j >= n_rows)
    ):
        raise ValueError("pair manifest contains out-of-range row ids")
    ep_i = episode_idx[manifest.row_i]
    ep_j = episode_idx[manifest.row_j]
    if not np.array_equal(ep_i, ep_j):
        raise ValueError("pair manifest crosses episode boundaries")
    actual_delta = np.abs(step_idx[manifest.row_i] - step_idx[manifest.row_j])
    if not np.array_equal(actual_delta.astype(np.int32), manifest.delta):
        raise ValueError("pair manifest delta does not match endpoint steps")
    excluded = np.asarray(tuple(int(v) for v in excluded_episode_ids), dtype=np.int64)
    if len(excluded) and np.any(np.isin(ep_i, excluded)):
        raise ValueError("pair manifest overlaps evaluation source episodes")


def _resolve_checkpoint_path(checkpoint: str | Path, cache_dir: str | Path) -> Path:
    candidate = Path(checkpoint).expanduser()
    searched = (
        candidate,
        Path(cache_dir).expanduser() / "checkpoints" / candidate,
        ROOT / ".stable-wm" / "checkpoints" / candidate,
    )
    for path in searched:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "checkpoint not found; searched:\n" + "\n".join(f"  - {p}" for p in searched)
    )


def load_local_dataset(data_path: str | Path):
    """Open a local HDF5 file through the repository's dataset API."""
    from stable_worldmodel import data as swm_data

    path = Path(data_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    return swm_data.HDF5Dataset(path=path, keys_to_cache=[])


def _checkpoint_img_size(checkpoint_path: Path) -> int:
    config_path = checkpoint_path.parent / "config.json"
    config = json.loads(config_path.read_text())
    return int(config.get("encoder", {}).get("image_size", 224))


def _latent_digest(rows: np.ndarray, latents: np.ndarray, metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(latents, dtype=np.float32).tobytes())
    digest.update(
        _canonical_json({k: v for k, v in metadata.items() if k != "cache_sha256"})
    )
    return digest.hexdigest()


def save_latent_cache(
    prefix: str | Path,
    rows: np.ndarray,
    latents: np.ndarray,
    metadata: dict[str, Any],
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    npz_path, json_path = _manifest_paths(prefix)
    if not force and (npz_path.exists() or json_path.exists()):
        raise FileExistsError(f"latent cache already exists: {prefix}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=np.int64)
    latents = np.asarray(latents, dtype=np.float32)
    metadata["cache_sha256"] = _latent_digest(rows, latents, metadata)
    with npz_path.open("wb") as handle:
        np.savez(handle, row_id=rows, latent=latents)
    json_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n")
    return npz_path, json_path


def load_latent_cache(
    prefix: str | Path, expected: Mapping[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    npz_path, json_path = _manifest_paths(prefix)
    metadata = json.loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as arrays:
        rows = np.asarray(arrays["row_id"], dtype=np.int64)
        latents = np.asarray(arrays["latent"], dtype=np.float32)
    actual = _latent_digest(rows, latents, metadata)
    if metadata.get("cache_sha256") != actual:
        raise ValueError(f"latent cache hash mismatch: {prefix}")
    for key, expected_value in (expected or {}).items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"latent cache {key} mismatch: "
                f"expected={expected_value!r}, found={metadata.get(key)!r}"
            )
    return rows, latents, metadata


def extract_projected_latents(
    dataset,
    rows: np.ndarray,
    checkpoint: str | Path,
    cache_dir: str | Path,
    *,
    device: torch.device,
    img_size: int,
    batch_size: int,
) -> np.ndarray:
    """Encode unique raw image rows into frozen projector-output embeddings."""
    from checkpoint_utils import load_fblewm_checkpoint
    from utils import get_img_preprocessor

    model = load_fblewm_checkpoint(str(checkpoint), cache_dir=str(cache_dir))
    model = model.to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    transform = get_img_preprocessor("pixels", "pixels", img_size=int(img_size))
    chunks: list[np.ndarray] = []
    total_batches = (len(rows) + int(batch_size) - 1) // int(batch_size)
    with torch.no_grad():
        for batch_index, start in enumerate(
            range(0, len(rows), int(batch_size)), start=1
        ):
            batch_rows = rows[start : start + int(batch_size)].tolist()
            raw_pixels = dataset.get_row_data(batch_rows)["pixels"]
            pixels = transform({"pixels": raw_pixels})["pixels"]
            pixels = torch.as_tensor(pixels, dtype=torch.float32, device=device)
            embedding = model.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]
            chunks.append(embedding.float().cpu().numpy())
            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % 10 == 0
            ):
                print(
                    f"batch {batch_index}/{total_batches} | encoding projected latents",
                    flush=True,
                )
    if not chunks:
        raise ValueError("cannot extract an empty latent cache")
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


class _PairDataset(Dataset):
    def __init__(
        self, latents: torch.Tensor, index_i: np.ndarray, index_j: np.ndarray, labels
    ) -> None:
        self.latents = latents
        self.index_i = torch.as_tensor(index_i, dtype=torch.long)
        self.index_j = torch.as_tensor(index_j, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.latents[self.index_i[index]],
            self.latents[self.index_j[index]],
            self.labels[index],
        )


def _row_positions(cache_rows: np.ndarray, requested_rows: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(cache_rows, requested_rows)
    if np.any(positions >= len(cache_rows)) or not np.array_equal(
        cache_rows[positions], requested_rows
    ):
        raise ValueError("latent cache is missing rows required by pair manifest")
    return positions


def _epoch_metrics(
    total_loss: float, total_abs: float, total_sq: float, count: int
) -> dict[str, float]:
    return {
        "smooth_l1": total_loss / count,
        "mae_unscaled": (total_abs / count) * LABEL_SCALE,
        "rmse_unscaled": (total_sq / count) ** 0.5 * LABEL_SCALE,
    }


@torch.no_grad()
def _evaluate(
    head: TRMHead, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    head.eval()
    total_loss = total_abs = total_sq = 0.0
    count = 0
    for z_i, z_j, target in loader:
        z_i = z_i.to(device, non_blocking=False)
        z_j = z_j.to(device, non_blocking=False)
        target = target.to(device, non_blocking=False)
        prediction = head(z_i, z_j)
        total_loss += float(
            F.smooth_l1_loss(prediction, target, beta=1.0, reduction="sum").item()
        )
        error = prediction - target
        total_abs += float(error.abs().sum().item())
        total_sq += float(error.square().sum().item())
        count += len(target)
    return _epoch_metrics(total_loss, total_abs, total_sq, count)


def train_head(
    train_manifest: PairManifest,
    val_manifest: PairManifest,
    cache_rows: np.ndarray,
    cache_latents: np.ndarray,
    *,
    device: torch.device,
    head_seed: int = 3072,
    batch_size: int = 1024,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    workers: int = 0,
) -> tuple[TRMHead, list[dict[str, Any]], dict[str, Any]]:
    if workers != 0:
        raise ValueError("paper-faithful TRM training requires workers=0")
    if cache_latents.dtype != np.float32:
        raise ValueError("TRM latent cache must be raw float32")
    latent_tensor = torch.from_numpy(cache_latents)
    train_set = _PairDataset(
        latent_tensor,
        _row_positions(cache_rows, train_manifest.row_i),
        _row_positions(cache_rows, train_manifest.row_j),
        temporal_labels(train_manifest),
    )
    val_set = _PairDataset(
        latent_tensor,
        _row_positions(cache_rows, val_manifest.row_i),
        _row_positions(cache_rows, val_manifest.row_j),
        temporal_labels(val_manifest),
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    torch.manual_seed(int(head_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(head_seed))
    head = TRMHead(latent_dim=int(cache_latents.shape[1])).to(
        device=device, dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    metrics: list[dict[str, Any]] = []
    optimizer_steps = 0
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        generator = torch.Generator().manual_seed(int(head_seed) + epoch - 1)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=generator,
        )
        head.train()
        total_loss = total_abs = total_sq = 0.0
        count = 0
        epoch_started = time.perf_counter()
        for z_i, z_j, target in train_loader:
            z_i = z_i.to(device, non_blocking=False)
            z_j = z_j.to(device, non_blocking=False)
            target = target.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(z_i, z_j)
            loss = F.smooth_l1_loss(prediction, target, beta=1.0)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            error = prediction.detach() - target
            total_loss += float(loss.detach().item()) * len(target)
            total_abs += float(error.abs().sum().item())
            total_sq += float(error.square().sum().item())
            count += len(target)

        train_metrics = _epoch_metrics(total_loss, total_abs, total_sq, count)
        val_metrics = _evaluate(head, val_loader, device)
        record = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "train": train_metrics,
            "validation": val_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        metrics.append(record)
        if val_metrics["smooth_l1"] < best_loss:
            best_loss = val_metrics["smooth_l1"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"train={train_metrics['smooth_l1']:.7f} "
            f"val={val_metrics['smooth_l1']:.7f} "
            f"val_mae_steps={val_metrics['mae_unscaled']:.3f}",
            flush=True,
        )

    assert best_state is not None
    head.load_state_dict(best_state)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_smooth_l1": best_loss,
        "optimizer_steps": optimizer_steps,
        "wall_seconds": time.perf_counter() - started,
    }
    return head, metrics, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated paper-faithful TRM training core (TwoRoom)"
    )
    parser.add_argument("--task", choices=("tworoom",), default="tworoom")
    parser.add_argument("--checkpoint", required=True, help="FBLeWM .pt or cache name")
    parser.add_argument("--data", required=True, help="Local TwoRoom HDF5 path")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("STABLEWM_HOME", str(ROOT / ".stable-wm")),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Evaluation starts_manifest.json; repeat for every evaluation group",
    )
    parser.add_argument("--train-manifest", default=None, help="Existing train prefix")
    parser.add_argument("--val-manifest", default=None, help="Existing val prefix")
    parser.add_argument("--latent-cache", default=None, help="Existing/output cache prefix")
    parser.add_argument("--head-output", default=None, help="Output .pt path")
    parser.add_argument(
        "--stage", choices=("all", "pairs", "latents", "train"), default="all"
    )
    parser.add_argument("--train-pairs", type=int, default=100_000)
    parser.add_argument("--val-pairs", type=int, default=10_000)
    parser.add_argument("--train-pair-seed", type=int, default=3072)
    parser.add_argument("--val-pair-seed", type=int, default=3073)
    parser.add_argument("--head-seed", type=int, default=3072)
    parser.add_argument("--max-delta", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_or_create_pairs(
    args: argparse.Namespace,
    dataset,
    data_hash: str,
    excluded: tuple[int, ...],
    train_prefix: Path,
    val_prefix: Path,
) -> tuple[PairManifest, PairManifest]:
    episode_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = np.asarray(dataset.get_col_data(episode_col), dtype=np.int64)
    steps = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    table = build_episode_table(episodes, steps, excluded)
    existing = all(path.exists() for prefix in (train_prefix, val_prefix) for path in _manifest_paths(prefix))
    if existing and not args.force:
        train_manifest = load_pair_manifest(train_prefix)
        val_manifest = load_pair_manifest(val_prefix)
    else:
        train_manifest = sample_temporal_pairs(
            table,
            args.train_pairs,
            args.train_pair_seed,
            split="train",
            max_delta=args.max_delta,
        )
        val_manifest = sample_temporal_pairs(
            table,
            args.val_pairs,
            args.val_pair_seed,
            split="validation",
            max_delta=args.max_delta,
        )
        if args.train_pair_seed == args.val_pair_seed:
            raise ValueError("train and validation pair seeds must be independent")
        overlap = annotate_exact_pair_overlap(train_manifest, val_manifest)
        for manifest in (train_manifest, val_manifest):
            manifest.metadata["data_sha256"] = data_hash
            manifest.metadata["exact_pair_overlap_definition"] = (
                "unique unordered endpoint row pairs shared across train/validation"
            )
        save_pair_manifest(train_manifest, train_prefix, force=args.force)
        save_pair_manifest(val_manifest, val_prefix, force=args.force)
        print(f"saved pair manifests; exact train/val overlap={overlap}", flush=True)
    validate_pair_manifest(train_manifest, episodes, steps, excluded)
    validate_pair_manifest(val_manifest, episodes, steps, excluded)
    if train_manifest.metadata.get("data_sha256") != data_hash:
        raise ValueError("train pair manifest data hash does not match --data")
    if val_manifest.metadata.get("data_sha256") != data_hash:
        raise ValueError("validation pair manifest data hash does not match --data")
    return train_manifest, val_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workers != 0:
        raise ValueError("paper-faithful default and required worker count is 0")
    if args.train_pair_seed == args.val_pair_seed:
        raise ValueError("train and validation pair seeds must differ")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device=cpu")

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, args.cache_dir)
    data_path = Path(args.data).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    train_prefix = Path(args.train_manifest) if args.train_manifest else (
        output_root / "pairs" / args.task / "train"
    )
    val_prefix = Path(args.val_manifest) if args.val_manifest else (
        output_root / "pairs" / args.task / "val"
    )
    latent_prefix = Path(args.latent_cache) if args.latent_cache else (
        output_root / "latents" / args.task / "projected"
    )
    head_path = Path(args.head_output) if args.head_output else (
        output_root / "heads" / args.task / "true.pt"
    )

    print("hashing checkpoint and dataset", flush=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    data_hash = sha256_file(data_path)
    excluded = load_excluded_episode_ids(args.exclude_manifest)
    dataset = load_local_dataset(data_path)

    if args.stage in ("all", "pairs"):
        train_manifest, val_manifest = _load_or_create_pairs(
            args, dataset, data_hash, excluded, train_prefix, val_prefix
        )
        if args.stage == "pairs":
            return 0
    else:
        train_manifest = load_pair_manifest(train_prefix)
        val_manifest = load_pair_manifest(val_prefix)
        episode_col = (
            "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        )
        episodes = np.asarray(dataset.get_col_data(episode_col), dtype=np.int64)
        steps = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
        validate_pair_manifest(train_manifest, episodes, steps, excluded)
        validate_pair_manifest(val_manifest, episodes, steps, excluded)

    pair_hashes = {
        "train_pair_sha256": train_manifest.metadata["pair_manifest_sha256"],
        "val_pair_sha256": val_manifest.metadata["pair_manifest_sha256"],
    }
    img_size = int(args.img_size or _checkpoint_img_size(checkpoint_path))
    expected_cache = {
        "checkpoint_sha256": checkpoint_hash,
        "data_sha256": data_hash,
        "img_size": img_size,
        **pair_hashes,
    }
    cache_exists = all(path.exists() for path in _manifest_paths(latent_prefix))
    if args.stage in ("all", "latents"):
        if cache_exists and not args.force:
            cache_rows, cache_latents, cache_metadata = load_latent_cache(
                latent_prefix, expected_cache
            )
            print(f"reused {len(cache_rows)} cached raw projected latents", flush=True)
        else:
            cache_rows = np.unique(
                np.concatenate(
                    [
                        train_manifest.row_i,
                        train_manifest.row_j,
                        val_manifest.row_i,
                        val_manifest.row_j,
                    ]
                )
            )
            cache_latents = extract_projected_latents(
                dataset,
                cache_rows,
                checkpoint_path,
                args.cache_dir,
                device=device,
                img_size=img_size,
                batch_size=args.encode_batch_size,
            )
            cache_metadata = {
                "format_version": 1,
                "task": args.task,
                "row_count": len(cache_rows),
                "latent_dim": int(cache_latents.shape[1]),
                "dtype": "float32",
                "normalization": "none",
                "latent_source": "model.encode({'pixels': ...})['emb'] projector output",
                "preprocessing": {
                    "factory": "utils.get_img_preprocessor",
                    "source": "pixels",
                    "target": "pixels",
                    "img_size": img_size,
                },
                "checkpoint_path": str(checkpoint_path),
                "data_path": str(data_path),
                **expected_cache,
            }
            save_latent_cache(
                latent_prefix,
                cache_rows,
                cache_latents,
                cache_metadata,
                force=args.force,
            )
            print(f"saved {len(cache_rows)} raw projected latents", flush=True)
        if args.stage == "latents":
            return 0
    else:
        cache_rows, cache_latents, cache_metadata = load_latent_cache(
            latent_prefix, expected_cache
        )

    metadata_path = head_path.with_suffix(".json")
    metrics_path = head_path.with_name(head_path.stem + "_metrics.jsonl")
    existing_head_artifacts = [
        path for path in (head_path, metadata_path, metrics_path) if path.exists()
    ]
    if existing_head_artifacts and not args.force:
        raise FileExistsError(
            "head artifact already exists: "
            + ", ".join(str(path) for path in existing_head_artifacts)
        )
    head, metrics, summary = train_head(
        train_manifest,
        val_manifest,
        cache_rows,
        cache_latents,
        device=device,
        head_seed=args.head_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
    )
    head_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "task": args.task,
        "method": "trm_v2_temporal_replace_local",
        "latent_dim": head.latent_dim,
        "architecture": head.architecture(),
        "parameter_count": sum(p.numel() for p in head.parameters()),
        "label_type": "temporal_delta",
        "label_formula": "target = delta / 224.0",
        "label_scale": LABEL_SCALE,
        "seed": args.head_seed,
        "pair_seeds": {
            "train": args.train_pair_seed,
            "validation": args.val_pair_seed,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "loss": {"name": "smooth_l1_loss", "beta": 1.0},
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "workers": args.workers,
        "precision": "float32",
        "amp": False,
        "gradient_clipping": None,
        "scheduler": None,
        "model_selection": "minimum validation Smooth-L1",
        "train_pair_count": len(train_manifest),
        "validation_pair_count": len(val_manifest),
        "unique_pair_counts": {
            "train": len(_canonical_pair_set(train_manifest)),
            "validation": len(_canonical_pair_set(val_manifest)),
        },
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(data_path),
        "base_checkpoint_sha256": checkpoint_hash,
        "checkpoint_sha256": checkpoint_hash,
        "data_sha256": data_hash,
        "latent_cache_sha256": cache_metadata["cache_sha256"],
        **pair_hashes,
        **summary,
        "appendix_selection": {
            "width": {"tried": [128, 256, 512], "selected": 256},
            "depth": {"tried": [1, 2, 3], "selected": 2},
            "lr": {"tried": [3e-4, 1e-3, 3e-3], "selected": 1e-3},
            "weight_decay": {"tried": [0, 1e-5, 1e-4], "selected": 1e-4},
            "batch_size": {"tried": [512, 1024, 2048], "selected": 1024},
            "distance_scale": {"tried": [128, 224, 256], "selected": 224},
            "epochs": {"tried": [15, 20, 30], "selected": 20},
            "selection_rule": "validation pair loss/stability, not evaluation success",
        },
    }
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()
            },
            "latent_dim": head.latent_dim,
            "architecture": head.architecture(),
            "metadata": metadata,
        },
        head_path,
    )
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n")
    metrics_path.write_text(
        "".join(json.dumps(_jsonable(record), sort_keys=True) + "\n" for record in metrics)
    )
    print(
        f"saved best epoch {summary['best_epoch']} head to {head_path}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
