from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path, model, config: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "config": config,
        "metadata": metadata or {},
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


def load_checkpoint(
    path: str | Path,
    model,
    map_location: str | None = "cpu",
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    required = {"format_version", "model_state_dict", "config", "metadata"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Invalid checkpoint, missing keys: {', '.join(missing)}")
    if checkpoint["format_version"] != 1:
        raise ValueError(f"Unsupported checkpoint format_version: {checkpoint['format_version']}")
    if expected_config is not None and checkpoint["config"] != expected_config:
        raise ValueError("Checkpoint config does not match the expected config")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def capture_rng_state() -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    import random

    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def save_training_checkpoint(
    path: str | Path,
    model,
    optimizer,
    config: dict[str, Any],
    completed_updates: int,
    rng_state: dict[str, Any],
    sampler_state: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    scaler=None,
    attempted_updates: int | None = None,
    max_attempted_updates: int | None = None,
) -> None:
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "completed_updates": int(completed_updates),
        "attempted_updates": None if attempted_updates is None else int(attempted_updates),
        "max_attempted_updates": None if max_attempted_updates is None else int(max_attempted_updates),
        "rng_state": rng_state,
        "sampler_state": sampler_state,
        "metadata": metadata or {},
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


def load_training_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    map_location: str | None = "cpu",
    expected_config: dict[str, Any] | None = None,
    scaler=None,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    required = {
        "format_version",
        "model_state_dict",
        "optimizer_state_dict",
        "config",
        "completed_updates",
        "rng_state",
        "sampler_state",
        "metadata",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Invalid training checkpoint, missing keys: {', '.join(missing)}")
    if checkpoint["format_version"] != 2:
        raise ValueError(f"Unsupported training checkpoint format_version: {checkpoint['format_version']}")
    if expected_config is not None and checkpoint["config"] != expected_config:
        raise ValueError("Checkpoint config does not match the expected config")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint

