"""Read-only synthetic evaluator for the Paper C0 training boundary."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from ..audio import AudioConfig, AudioTransform
from ..checkpoint import capture_rng_state, restore_rng_state
from .trainer import PaperC0TrainerError, _dataset_digest


class PaperC0EvaluatorError(RuntimeError):
    pass


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_batch(batch: dict[str, Any], expected_batch: int, target_count: int = 4) -> None:
    required = {"mixture", "target", "all_targets", "item_id", "part_id", "target_source_index", "sample_rate"}
    missing = required - set(batch)
    if missing:
        raise PaperC0EvaluatorError(f"evaluation batch missing fields: {sorted(missing)}")
    if batch["mixture"].ndim != 3 or batch["target"].ndim != 3:
        raise PaperC0EvaluatorError("evaluation waveforms must have shape [B, C, samples]")
    if batch["mixture"].shape != batch["target"].shape or batch["mixture"].shape[0] != expected_batch:
        raise PaperC0EvaluatorError("evaluation waveform shape mismatch")
    if set(batch["all_targets"]) != {"violin1", "violin2", "viola", "cello"} or len(batch["all_targets"]) != target_count:
        raise PaperC0EvaluatorError("all_targets must contain the four canonical parts")
    for part, waveform in batch["all_targets"].items():
        if waveform.shape != batch["target"].shape or not torch.isfinite(waveform).all():
            raise PaperC0EvaluatorError(f"invalid target waveform for {part}")
    if not torch.isfinite(batch["mixture"]).all() or not torch.isfinite(batch["target"]).all():
        raise PaperC0EvaluatorError("evaluation input contains NaN or Inf")


def _selection(errors: list[float], target_index: int, atol: float = 1.0e-8, rtol: float = 1.0e-6) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    nearest = int(np.argmin(values))
    minimum = float(values[nearest])
    tolerance = atol + rtol * max(1.0, abs(minimum))
    ties = np.flatnonzero(np.abs(values - minimum) <= tolerance).astype(int).tolist()
    return {
        "nearest_source_index": nearest,
        "nearest_source_correct": nearest == int(target_index),
        "unique_selected_correct": len(ties) == 1 and nearest == int(target_index),
        "tie_count": len(ties),
    }


def evaluate_model(model, dataset, config: dict[str, Any], device: str = "cpu", batch_size: int = 1) -> dict[str, Any]:
    """Evaluate a C0 model without updates and restore mode/RNG on exit."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not getattr(dataset, "all_targets", False):
        raise PaperC0EvaluatorError("evaluation dataset must enable all_targets=True")
    if device == "cuda" and not torch.cuda.is_available():
        raise PaperC0EvaluatorError("CUDA requested but unavailable")
    before_hash = _state_hash(model)
    before_mode = bool(model.training)
    before_rng = capture_rng_state()
    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    rows: list[dict[str, Any]] = []
    try:
        model.to(device)
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                indices = list(range(start, min(start + batch_size, len(dataset))))
                batch = default_collate([dataset[index] for index in indices])
                _validate_batch(batch, len(indices))
                mixture = batch["mixture"].to(device)
                target = batch["target"].to(device)
                timbre = batch["timbre_id"].to(device).long()
                position = batch["position_id"].to(device).long()
                predicted = model(transform.stft(mixture), timbre, position)
                if predicted.ndim != 4 or predicted.shape[0] != len(indices) or not torch.isfinite(predicted).all():
                    raise PaperC0EvaluatorError("model output shape or finite check failed")
                output = transform.istft(predicted, length=target.shape[-1]).detach().cpu().numpy().astype(np.float64)
                mixture_np = mixture.detach().cpu().numpy().astype(np.float64)
                target_np = target.detach().cpu().numpy().astype(np.float64)
                sources_np = {part: value.detach().cpu().numpy().astype(np.float64) for part, value in batch["all_targets"].items()}
                parts = ("violin1", "violin2", "viola", "cello")
                for local_index, index in enumerate(indices):
                    target_index = int(batch["target_source_index"][local_index])
                    sources = [sources_np[part][local_index] for part in parts]
                    target_energy = float(np.sum(target_np[local_index] ** 2))
                    if target_energy <= 0.0 or not np.isfinite(target_energy):
                        raise PaperC0EvaluatorError("target energy must be positive and finite")
                    prediction = output[local_index]
                    errors = [float(np.sum((prediction - source) ** 2)) for source in sources]
                    blind = mixture_np[local_index] / float(len(parts))
                    blind_error = float(np.sum((blind - target_np[local_index]) ** 2))
                    zero_error = target_energy
                    selection = _selection(errors, target_index)
                    rows.append({
                        "dataset_index": index,
                        "item_id": str(batch["item_id"][local_index]),
                        "part_id": str(batch["part_id"][local_index]),
                        "target_source_index": target_index,
                        "target_error": errors[target_index] / target_energy,
                        "blind_error": blind_error / target_energy,
                        "zero_error": zero_error / target_energy,
                        "all_source_errors": errors,
                        "output_norm": float(np.linalg.norm(prediction)),
                        "target_norm": float(np.linalg.norm(target_np[local_index])),
                        **selection,
                    })
    finally:
        model.train(before_mode)
        restore_rng_state(before_rng)
    after_hash = _state_hash(model)
    if after_hash != before_hash:
        raise PaperC0EvaluatorError("evaluation changed model parameters or buffers")
    return {
        "rows": rows,
        "summary": summarize_rows(rows),
        "dataset_digest": _dataset_digest(dataset),
        "model_state_hash_before": before_hash,
        "model_state_hash_after": after_hash,
        "model_training_mode_restored": bool(model.training) == before_mode,
        "optimizer_updates": 0,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise PaperC0EvaluatorError("cannot summarize an empty evaluation")
    query = np.asarray([row["target_error"] for row in rows], dtype=np.float64)
    blind = np.asarray([row["blind_error"] for row in rows], dtype=np.float64)
    zero = np.asarray([row["zero_error"] for row in rows], dtype=np.float64)
    return {
        "query_count": len(rows),
        "query_mean_R": float(np.mean(query)),
        "query_median_R": float(np.median(query)),
        "query_p90_R": float(np.percentile(query, 90)),
        "blind_R": float(np.mean(blind)),
        "zero_R": float(np.mean(zero)),
        "nearest_accuracy": float(np.mean([row["nearest_source_correct"] for row in rows])),
        "unique_accuracy": float(np.mean([row["unique_selected_correct"] for row in rows])),
        "tie_ratio": float(np.mean([row["tie_count"] > 1 for row in rows])),
    }


def write_evaluation(result: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "query_detail.json").write_text(json.dumps(result["rows"], indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    (out / "evaluation_evidence.json").write_text(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2), encoding="utf-8")
