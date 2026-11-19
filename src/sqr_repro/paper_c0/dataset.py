"""Synthetic/real-compatible Paper C0 WAV dataset boundary.

The loader consumes the existing JSONL manifest contract. It does not perform
normalization and uses one deterministic crop offset for a mixture and all four
targets. Real SQR use remains gated by the verified manifest/evidence layer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    Dataset = object

from ..data import PARTS, DataContractError, iter_manifest, target_ids, validate_manifest_record


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for Paper C0 Dataset")
    return torch


def adapt_manifest_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize R1C/legacy manifest names without deriving source IDs from positions."""
    work_id = record.get("work_id", record.get("composition_id"))
    movement_id = record.get("movement_id", record.get("performance_id"))
    chunk_id = record.get("chunk_id", record.get("item_id"))
    layout_id = record.get("layout_id")
    seating_type = record.get("seating_type")
    if seating_type is None and isinstance(layout_id, str) and layout_id.lower().startswith("type"):
        seating_type = int(layout_id.lower().removeprefix("type"))
    if not work_id or not movement_id or not chunk_id or seating_type not in {1, 2}:
        raise DataContractError("manifest identity requires work/movement/chunk and seating type")
    if set(record.get("stems", {})) != set(PARTS):
        raise DataContractError("manifest identity requires all four canonical parts")
    return {
        "work_id": str(work_id),
        "movement_id": str(movement_id),
        "chunk_id": str(chunk_id),
        "seating_type": int(seating_type),
        "layout_id": str(layout_id or f"type{seating_type}"),
        "source_index_by_part": {part: index for index, part in enumerate(PARTS)},
        "position_ids": {part: int(record["position_ids"][part]) for part in PARTS},
    }


def _crop_start(
    length: int,
    crop_samples: int | None,
    global_seed: int,
    item_id: str,
    epoch: int = 0,
    crop_slot: int = 0,
) -> int:
    if crop_samples is None:
        return 0
    if crop_samples <= 0 or crop_samples > length:
        raise DataContractError("crop_samples must be positive and no longer than the audio")
    token = f"{global_seed}|{epoch}|{item_id}|{crop_slot}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return int(np.random.default_rng(local_seed).integers(0, length - crop_samples + 1))


class PaperC0WavDataset(Dataset):
    """Four-query Dataset over one or more manifest records.

    Each index is `(mixture_record, target_part)` and all target queries for a
    record reuse exactly the same loaded/cropped mixture waveform.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path,
        split: str,
        crop_samples: int | None,
        crop_seed: int = 2026,
        epoch: int = 0,
        crop_slot: int = 0,
        all_targets: bool = False,
        require_fixture: bool | None = None,
        crop_offset_override: int | None = None,
    ):
        _require_torch()
        self.manifest_path = Path(manifest_path)
        self.root = Path(root)
        self.split = str(split)
        self.crop_samples = crop_samples
        self.crop_seed = int(crop_seed)
        self.epoch = int(epoch)
        self.crop_slot = int(crop_slot)
        self.all_targets = bool(all_targets)
        self.crop_offset_override = crop_offset_override
        self.last_read_log: list[dict[str, Any]] = []
        self.records = []
        for record in iter_manifest(self.manifest_path):
            if str(record.get("split")) != self.split:
                continue
            validate_manifest_record(record, self.root, require_audio=True)
            if require_fixture is True and not bool(record.get("is_fixture")):
                raise DataContractError("synthetic dataset requires is_fixture=true")
            if require_fixture is False and bool(record.get("is_fixture")):
                raise DataContractError("real dataset loader cannot consume fixture records")
            self.records.append(record)
        if not self.records:
            raise DataContractError(f"no records found for split={self.split}")

    def __len__(self) -> int:
        return len(self.records) * len(PARTS)

    def _read_window(self, relative_path: str, start: int, frames: int, expected_samples: int, expected_rate: int) -> np.ndarray:
        path = self.root / relative_path
        with sf.SoundFile(str(path), mode="r") as sound_file:
            if sound_file.samplerate != expected_rate or sound_file.channels != 2 or len(sound_file) != expected_samples:
                raise DataContractError(f"audio contract mismatch: {path}")
            sound_file.seek(start)
            wave = sound_file.read(frames, dtype="float32", always_2d=True)
            rate = sound_file.samplerate
        if rate != expected_rate or wave.shape[1] != 2 or wave.shape[0] != frames:
            raise DataContractError(f"audio contract mismatch: {path}")
        if not np.isfinite(wave).all():
            raise DataContractError(f"non-finite audio: {path}")
        self.last_read_log.append({"path": str(path), "start": start, "frames": int(wave.shape[0])})
        return np.ascontiguousarray(wave.T)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index, target_index = divmod(index, len(PARTS))
        record = self.records[record_index]
        length = int(record["num_samples"])
        if self.crop_offset_override is not None:
            start = int(self.crop_offset_override)
            if self.crop_samples is not None and start + self.crop_samples > length:
                raise DataContractError(
                    f"crop_offset_override {start} + crop_samples {self.crop_samples} exceeds audio length {length}"
                )
        else:
            start = _crop_start(
                length, self.crop_samples, self.crop_seed, str(record["item_id"]), self.epoch, self.crop_slot
            )
        crop_length = self.crop_samples or length
        self.last_read_log = []
        mixture = self._read_window(record["mixture"], start, crop_length, length, int(record["sample_rate"]))
        target_part = PARTS[target_index]
        target = self._read_window(record["stems"][target_part], start, crop_length, length, int(record["sample_rate"]))
        timbre_id, position_id = target_ids(target_part, record["position_ids"])
        identity = adapt_manifest_identity(record)
        output = {
            "mixture": torch.from_numpy(mixture.copy()),
            "target": torch.from_numpy(target.copy()),
            "item_id": str(record["item_id"]),
            "composition_id": str(record["composition_id"]),
            "performance_id": str(record["performance_id"]),
            "work_id": identity["work_id"],
            "movement_id": identity["movement_id"],
            "chunk_id": identity["chunk_id"],
            "split": str(record["split"]),
            "layout_id": str(record["layout_id"]),
            "target_source_index": target_index,
            "part_id": target_part,
            "timbre_id": timbre_id,
            "position_id": position_id,
            "crop_offset": start,
            "crop_samples": crop_length,
            "crop_seed": self.crop_seed,
            "target_waveform_hash": str(record.get("stem_hashes", {}).get(target_part, "")),
            "target_waveform_identity": f"{record['item_id']}::{target_part}::{record['stems'][target_part]}",
            "mixture_waveform_hash": str(record.get("mixture_hash", "")),
            "sample_rate": int(record["sample_rate"]),
        }
        if self.all_targets:
            all_targets = {target_part: torch.from_numpy(target.copy())}
            for part in PARTS:
                if part != target_part:
                    all_targets[part] = torch.from_numpy(
                        self._read_window(record["stems"][part], start, crop_length, length, int(record["sample_rate"]))
                    )
            output["all_targets"] = all_targets
        return output


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_manifest(path))
