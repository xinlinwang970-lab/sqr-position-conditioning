"""Full-waveform C0 inference with deterministic chunked overlap-add."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from ..audio import AudioConfig, AudioTransform
from ..checkpoint import capture_rng_state, restore_rng_state
from .model_factory import build_model


class PaperC0WaveformError(RuntimeError):
    pass


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_waveform(waveform: torch.Tensor, config: AudioConfig) -> None:
    if waveform.ndim != 2 or waveform.shape[0] != config.channels:
        raise PaperC0WaveformError("waveform must have shape [2, samples]")
    if waveform.shape[-1] < 1 or not torch.isfinite(waveform).all():
        raise PaperC0WaveformError("waveform must be finite and non-empty")


def _model_state_hash(model) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def infer_full_waveform(
    model,
    mixture: torch.Tensor,
    config: dict[str, Any],
    timbre_id: int,
    position_id: int,
    device: str = "cpu",
    chunk_samples: int | None = None,
    overlap_samples: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Infer one stereo waveform; chunk overlap is normalized by sample count."""
    audio_config = AudioConfig.from_dict(config["audio"])
    _validate_waveform(mixture, audio_config)
    if device == "cuda" and not torch.cuda.is_available():
        raise PaperC0WaveformError("CUDA requested but unavailable")
    chunk = int(chunk_samples or audio_config.segment_samples)
    if chunk <= 0 or chunk % audio_config.hop_length != 0:
        raise PaperC0WaveformError("chunk_samples must be a positive hop-aligned value")
    if overlap_samples < 0 or overlap_samples >= chunk or overlap_samples % audio_config.hop_length != 0:
        raise PaperC0WaveformError("overlap_samples must be hop-aligned and smaller than chunk_samples")
    step = chunk - overlap_samples
    length = int(mixture.shape[-1])
    starts = list(range(0, max(length - chunk, 0) + 1, step))
    if not starts or starts[-1] + chunk < length:
        starts.append(max(length - chunk, 0))
    transform = AudioTransform(audio_config)
    before_mode = bool(model.training)
    before_state_hash = _model_state_hash(model)
    before_rng = capture_rng_state()
    model.eval()
    output = torch.zeros((audio_config.channels, length), dtype=torch.float32, device="cpu")
    weights = torch.zeros(length, dtype=torch.float32)
    try:
        with torch.inference_mode():
            for start in starts:
                window = mixture[:, start : min(start + chunk, length)].float()
                valid = window.shape[-1]
                if valid < chunk:
                    window = F.pad(window, (0, chunk - valid))
                batch = window.unsqueeze(0).to(device)
                packed = transform.stft(batch)
                timbre = torch.tensor([int(timbre_id)], device=device, dtype=torch.long)
                position = torch.tensor([int(position_id)], device=device, dtype=torch.long)
                predicted = model(packed, timbre, position)
                if predicted.shape != packed.shape or not torch.isfinite(predicted).all():
                    raise PaperC0WaveformError("model output shape or finite check failed")
                reconstructed = transform.istft(predicted, length=chunk)[0].detach().cpu()
                end = min(start + chunk, length)
                output[:, start:end] += reconstructed[:, : end - start]
                weights[start:end] += 1.0
    finally:
        model.train(before_mode)
        restore_rng_state(before_rng)
    if _model_state_hash(model) != before_state_hash:
        raise PaperC0WaveformError("waveform evaluation changed model parameters or buffers")
    if torch.any(weights <= 0):
        raise PaperC0WaveformError("overlap-add left uncovered samples")
    output = output / weights.unsqueeze(0)
    metadata = {
        "sample_rate": audio_config.sample_rate,
        "channels": audio_config.channels,
        "input_samples": length,
        "chunk_samples": chunk,
        "overlap_samples": overlap_samples,
        "step_samples": step,
        "chunk_starts": starts,
        "timbre_id": int(timbre_id),
        "position_id": int(position_id),
        "packed_complex": True,
        "overlap_add_rule": "uniform sample-count normalization",
        "overlap_add_assumption": "ENGINEERING_ASSUMPTION",
        "post_processing": "none",
        "deterministic": True,
    }
    return output, metadata


def infer_wav_file(
    model,
    mixture_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    timbre_id: int,
    position_id: int,
    device: str = "cpu",
    chunk_samples: int | None = None,
    overlap_samples: int = 0,
) -> dict[str, Any]:
    mixture, sample_rate = sf.read(str(mixture_path), dtype="float32", always_2d=True)
    expected_rate = int(config["audio"]["sample_rate"])
    if sample_rate != expected_rate or mixture.shape[1] != 2:
        raise PaperC0WaveformError("input WAV sample rate or channel count does not match config")
    waveform, metadata = infer_full_waveform(
        model, torch.from_numpy(mixture.T.copy()), config, timbre_id, position_id, device, chunk_samples, overlap_samples
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), waveform.T.numpy(), sample_rate, subtype="FLOAT")
    metadata.update({"mixture_path": str(Path(mixture_path).resolve()), "mixture_sha256": _sha256_file(mixture_path), "output_path": str(out.resolve())})
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def infer_checkpoint_wav(
    checkpoint_path: str | Path,
    mixture_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    timbre_id: int,
    position_id: int,
    device: str = "cpu",
    chunk_samples: int | None = None,
    overlap_samples: int = 0,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 2 or checkpoint.get("config") != config:
        raise PaperC0WaveformError("checkpoint format or config mismatch")
    model, report = build_model(config, "synthetic_smoke")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    metadata = infer_wav_file(model, mixture_path, output_path, config, timbre_id, position_id, device, chunk_samples, overlap_samples)
    metadata.update({"checkpoint_path": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": _sha256_file(checkpoint_path), "architecture_identity": report["architecture_version"], "optimizer_updates": 0})
    Path(output_path).with_suffix(Path(output_path).suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
