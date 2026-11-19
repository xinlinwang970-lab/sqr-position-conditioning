from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


INSTRUMENT_TO_ID = {"violin": 0, "viola": 1, "cello": 2}
PART_TO_INSTRUMENT = {"violin1": "violin", "violin2": "violin", "viola": "viola", "cello": "cello"}
PARTS = tuple(PART_TO_INSTRUMENT)
LAYOUTS = {
    "type1": {"violin1": 3, "violin2": 0, "viola": 1, "cello": 2},
    "type2": {"violin1": 3, "violin2": 2, "viola": 1, "cello": 0},
}


class DataContractError(ValueError):
    pass


def target_ids(part: str, position_ids: dict[str, int]) -> tuple[int, int]:
    if part not in PART_TO_INSTRUMENT:
        raise DataContractError(f"Unknown part: {part}")
    if part not in position_ids:
        raise DataContractError(f"Missing position id for {part}")
    position_id = int(position_ids[part])
    if position_id not in {0, 1, 2, 3}:
        raise DataContractError(f"Invalid position id for {part}: {position_id}")
    return INSTRUMENT_TO_ID[PART_TO_INSTRUMENT[part]], position_id


def iter_manifest(path: str | Path) -> Iterable[dict]:
    manifest_path = Path(path)
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataContractError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        yield record


def validate_manifest_record(record: dict, root: str | Path, require_audio: bool = True) -> None:
    required = {
        "item_id",
        "composition_id",
        "performance_id",
        "split",
        "layout_id",
        "mixture",
        "stems",
        "position_ids",
        "sample_rate",
        "channels",
        "num_samples",
        "is_fixture",
    }
    missing = sorted(required - set(record))
    if missing:
        raise DataContractError(f"Missing manifest keys: {', '.join(missing)}")
    if set(record["stems"]) != set(PARTS):
        raise DataContractError("Manifest stems must contain violin1, violin2, viola, cello")
    for part in PARTS:
        target_ids(part, record["position_ids"])
    if int(record["channels"]) != 2:
        raise DataContractError("Manifest record must be stereo")
    if int(record["num_samples"]) <= 0:
        raise DataContractError("Manifest num_samples must be positive")
    if require_audio:
        _validate_audio_files(record, Path(root))


def crop_same_start(arrays: list[np.ndarray], start: int, length: int) -> list[np.ndarray]:
    if start < 0 or length <= 0:
        raise DataContractError("start must be non-negative and length must be positive")
    out = []
    for array in arrays:
        if start + length > array.shape[0]:
            raise DataContractError("Crop exceeds array length")
        out.append(array[start : start + length])
    return out


def split_leakage_keys(records: Iterable[dict]) -> dict[tuple[str, str], set[str]]:
    seen: dict[tuple[str, str], set[str]] = {}
    for record in records:
        key = (str(record["composition_id"]), str(record["performance_id"]))
        seen.setdefault(key, set()).add(str(record["split"]))
    return {key: splits for key, splits in seen.items() if len(splits) > 1}


def _validate_audio_files(record: dict, root: Path) -> None:
    paths = [root / record["mixture"]] + [root / record["stems"][part] for part in PARTS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise DataContractError(f"Missing audio file(s): {missing}")

    infos = [sf.info(str(path)) for path in paths]
    sample_rates = {info.samplerate for info in infos}
    channels = {info.channels for info in infos}
    frames = {info.frames for info in infos}
    if len(sample_rates) != 1 or int(record["sample_rate"]) not in sample_rates:
        raise DataContractError("Audio sample rates are inconsistent")
    if channels != {2}:
        raise DataContractError("All audio files must be stereo")
    if len(frames) != 1:
        raise DataContractError("Mixture and stems must have equal length")

