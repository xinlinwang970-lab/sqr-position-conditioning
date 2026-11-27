"""Non-result-bearing Paper C0 diagnostics.

The reported experiments use ``sqr_repro.evaluation`` and its versioned
``legacy_v1`` WaveSDR path.  This NumPy-only module is retained for isolated
diagnostic probes and is not an alternative result-evaluation entrypoint.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


PAPER_SDR_PROTOCOL = "PAPER_SDR_PROTOCOL_UNRESOLVED"
SDR_DEFINITION = "10*log10(sum(reference**2) / (sum((estimate-reference)**2) + eps))"
SDR_BACKEND = "numpy.float64"
SDR_EPS = 1.0e-12
SILENCE_ENERGY_THRESHOLD = 1.0e-12


def _arrays(estimate, reference) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError(f"metric shape mismatch: {predicted.shape} != {target.shape}")
    if predicted.ndim < 1 or not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("metrics require finite non-empty arrays")
    return predicted, target


def waveform_mae(estimate, reference) -> float:
    predicted, target = _arrays(estimate, reference)
    return float(np.mean(np.abs(predicted - target)))


def waveform_sdr(estimate, reference, eps: float = SDR_EPS) -> float:
    predicted, target = _arrays(estimate, reference)
    energy = float(np.sum(target * target))
    if energy <= SILENCE_ENERGY_THRESHOLD:
        return float("nan")
    error = float(np.sum((predicted - target) ** 2))
    return float(10.0 * math.log10(energy / (error + eps)))


def evaluate_waveform_variants(estimate, target, sources: list[np.ndarray] | None = None) -> dict[str, Any]:
    predicted, reference = _arrays(estimate, target)
    result: dict[str, Any] = {
        "mae": waveform_mae(predicted, reference),
        "sdr": waveform_sdr(predicted, reference),
        "sdr_definition": SDR_DEFINITION,
        "sdr_backend": SDR_BACKEND,
        "paper_sdr_protocol": PAPER_SDR_PROTOCOL,
        "silent_reference": bool(np.sum(reference * reference) <= SILENCE_ENERGY_THRESHOLD),
    }
    if sources is not None:
        if not sources:
            raise ValueError("sources must not be empty")
        source_arrays = [_arrays(source, reference)[0] for source in sources]
        result["oracle"] = {"mae": 0.0, "sdr": waveform_sdr(reference, reference)}
        result["zero"] = {"mae": waveform_mae(np.zeros_like(reference), reference), "sdr": waveform_sdr(np.zeros_like(reference), reference)}
        result["wrong_source"] = []
        for source in source_arrays:
            if np.array_equal(source, reference):
                continue
            result["wrong_source"].append({"mae": waveform_mae(source, reference), "sdr": waveform_sdr(source, reference)})
        scaled = 0.5 * reference
        result["scaled_estimate"] = {"scale": 0.5, "mae": waveform_mae(scaled, reference), "sdr": waveform_sdr(scaled, reference)}
    return result


def aggregate_track_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty metric rows")
    sdr_values = np.asarray([row["sdr"] for row in rows], dtype=np.float64)
    sdr_values = sdr_values[np.isfinite(sdr_values)]
    mae_values = np.asarray([row["mae"] for row in rows], dtype=np.float64)
    return {
        "track_count": len(rows),
        "mae_mean": float(np.mean(mae_values)),
        "mae_median": float(np.median(mae_values)),
        "sdr_valid_track_count": int(sdr_values.size),
        "sdr_mean": float(np.mean(sdr_values)) if sdr_values.size else float("nan"),
        "sdr_median": float(np.median(sdr_values)) if sdr_values.size else float("nan"),
        "sdr_protocol": PAPER_SDR_PROTOCOL,
    }
