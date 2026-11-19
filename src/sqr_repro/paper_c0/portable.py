"""Portable, zero-update training preflight for external real-data machines."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from ..config import load_config, write_resolved_config
from ..data import DataContractError, iter_manifest, validate_manifest_record


class PortablePreflightError(RuntimeError):
    pass


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_portable_preflight(
    config_path: str | Path,
    manifest_path: str | Path,
    data_root: str | Path,
    split: str = "train",
    require_audio: bool = False,
    resolved_config_out: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the external training contract without constructing an optimizer or updating a model."""
    config = load_config(config_path)
    manifest = Path(manifest_path)
    root = Path(data_root)
    if not manifest.is_file():
        raise PortablePreflightError(f"manifest does not exist: {manifest}")
    if not root.exists():
        raise PortablePreflightError(f"data root does not exist: {root}")
    records = list(iter_manifest(manifest))
    selected = [record for record in records if str(record.get("split")) == split]
    if not selected:
        raise PortablePreflightError(f"manifest contains no records for split={split}")
    errors: list[str] = []
    mixture_ids: set[str] = set()
    identity_keys: set[tuple[str, str, str]] = set()
    for index, record in enumerate(selected, start=1):
        try:
            validate_manifest_record(record, root, require_audio=require_audio)
        except DataContractError as exc:
            errors.append(f"row {index}: {exc}")
        item_id = str(record.get("item_id"))
        if item_id in mixture_ids:
            errors.append(f"row {index}: duplicate mixture/item_id {item_id}")
        mixture_ids.add(item_id)
        identity_keys.add((str(record.get("composition_id")), str(record.get("performance_id")), item_id))
    if errors:
        raise PortablePreflightError("manifest contract failed: " + "; ".join(errors[:8]))
    resolved_path = None
    if resolved_config_out is not None:
        resolved_path = str(Path(resolved_config_out).resolve())
        write_resolved_config(config, resolved_path)
    return {
        "status": "FULL_AUDIO_PASS" if require_audio else "SCHEMA_ONLY_PASS",
        "optimizer_updates": 0,
        "backward_calls": 0,
        "config_path": str(Path(config_path).resolve()),
        "resolved_config_path": resolved_path,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "manifest_semantic_hash": _digest(records),
        "data_root": str(root.resolve()),
        "split": split,
        "record_count": len(selected),
        "mixture_ids_unique": len(mixture_ids) == len(selected),
        "identity_count": len(identity_keys),
        "audio_audit": "EXECUTED" if require_audio else "NOT_RUN",
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "training_entrypoint": "external_machine_only",
    }
