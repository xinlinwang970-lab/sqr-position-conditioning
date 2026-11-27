from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from .audio import AudioConfig, AudioTransform
from .paper_c0.dataset import PARTS


LEGACY_V1 = "legacy_v1"
PART_TO_TIMBRE = {"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}


def compute_wavesdr(estimate, reference):
    """Historical waveform-domain, scale-dependent SDR used by this repository."""

    ref_energy = torch.sum(reference**2)
    err = estimate - reference
    return 10 * torch.log10(ref_energy / (torch.sum(err**2) + 1e-8))


def legacy_v1_settings(config: dict[str, Any]) -> tuple[float, float]:
    """Return the executable legacy-v1 settings after checking the config contract."""

    inference = config["inference"]
    if inference["evaluation_protocol"] != LEGACY_V1:
        raise ValueError("Only evaluation_protocol=legacy_v1 is implemented")
    if float(inference["overlap_ratio_requested"]) != 0.5:
        raise ValueError("legacy_v1 requires overlap_ratio_requested=0.5")
    if bool(inference["align_stride_to_stft_hop"]):
        raise ValueError("legacy_v1 uses the unaligned half-segment stride")
    if inference["blend_window"] != "hann":
        raise ValueError("legacy_v1 requires blend_window=hann")
    if float(inference["blend_weight_floor"]) != 1.0e-8:
        raise ValueError("legacy_v1 requires blend_weight_floor=1e-8")
    if not bool(inference["keep_original_length"]):
        raise ValueError("legacy_v1 evaluation trims estimates to the reference length")
    return (
        float(inference["overlap_ratio_requested"]),
        float(inference["blend_weight_floor"]),
    )


def inference_full_length_legacy_v1(
    model,
    mixture,
    transform,
    timbre_id,
    position_id,
    device,
    num_frames,
    hop_length,
    overlap_ratio=0.5,
    *,
    weight_floor=1.0e-8,
    pad_time_to_multiple_of_8=True,
):
    """Run the exact historical overlap-add inference path.

    This deliberately preserves the original unaligned half-segment stride,
    periodic Hann window, CPU accumulation, and weight clamp.  It is not a
    corrected or alternative inference protocol.
    """

    channels, total_samples = mixture.shape
    segment_samples = (num_frames - 1) * hop_length
    hop_samples = int(segment_samples * (1 - overlap_ratio))
    if total_samples < segment_samples:
        mixture = torch.nn.functional.pad(
            mixture, (0, segment_samples - total_samples)
        )
        total_samples = mixture.shape[-1]
    output = torch.zeros(channels, total_samples, dtype=torch.float32)
    weight = torch.zeros(total_samples, dtype=torch.float32)
    window = torch.hann_window(segment_samples, periodic=True)
    starts = list(
        range(0, max(total_samples - segment_samples + 1, 1), hop_samples)
    )
    if not starts or starts[-1] + segment_samples < total_samples:
        starts.append(max(total_samples - segment_samples, 0))
    for start in starts:
        end = min(start + segment_samples, total_samples)
        chunk = mixture[:, start:end]
        if chunk.shape[-1] < segment_samples:
            chunk = torch.nn.functional.pad(
                chunk, (0, segment_samples - chunk.shape[-1])
            )
        spec = transform.stft(chunk.unsqueeze(0).to(device))
        if pad_time_to_multiple_of_8:
            time_frames = spec.shape[2]
            pad_frames = (8 - time_frames % 8) % 8
            if pad_frames:
                spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_frames))
        timbre = torch.tensor([timbre_id], dtype=torch.long, device=device)
        position = torch.tensor([position_id], dtype=torch.long, device=device)
        with torch.no_grad():
            estimated_spec = model(spec, timbre, position)
        wave = transform.istft(estimated_spec, length=segment_samples)[0].cpu()
        blend = window[: end - start]
        output[:, start:end] += wave[:, : end - start] * blend
        weight[start:end] += blend
    return (output / torch.clamp(weight, min=weight_floor))[:, :total_samples]


def evaluate_full_length_legacy_v1(
    model,
    manifest_path,
    data_root,
    device,
    config,
    part_filter=None,
    *,
    include_composition_id=True,
    pad_time_to_multiple_of_8=True,
):
    """Evaluate a manifest with the exact legacy-v1 inference and WaveSDR path."""

    overlap_ratio, weight_floor = legacy_v1_settings(config)
    records = [json.loads(line) for line in open(manifest_path)]
    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    num_frames = config["audio"]["num_frames"]
    hop_length = config["audio"]["hop_length"]
    results = []
    model.eval()
    for record in records:
        mixture_array, _ = sf.read(
            str(Path(data_root) / record["mixture"]),
            dtype="float32",
            always_2d=True,
        )
        mixture = torch.from_numpy(mixture_array.T.copy())
        for part_id in PARTS:
            if part_filter == "type2" and record["layout_id"] != "type2":
                continue
            estimate = inference_full_length_legacy_v1(
                model,
                mixture,
                transform,
                PART_TO_TIMBRE[part_id],
                record["position_ids"][part_id],
                device,
                num_frames,
                hop_length,
                overlap_ratio,
                weight_floor=weight_floor,
                pad_time_to_multiple_of_8=pad_time_to_multiple_of_8,
            )
            reference_array, _ = sf.read(
                str(Path(data_root) / record["stems"][part_id]),
                dtype="float32",
                always_2d=True,
            )
            reference = torch.from_numpy(reference_array.T.copy())
            min_len = min(estimate.shape[-1], reference.shape[-1])
            wavesdr = compute_wavesdr(
                estimate[..., :min_len], reference[..., :min_len]
            ).item()
            row = {"item_id": record["item_id"]}
            if include_composition_id:
                row["composition_id"] = record["composition_id"]
            row.update(
                {
                    "part_id": part_id,
                    "layout_id": record["layout_id"],
                    "sdr": wavesdr,
                }
            )
            results.append(row)
    model.train()
    return results


def summarize_wavesdr(results):
    """Preserve the historical aggregate field names used by result JSON files."""

    values = [row["sdr"] for row in results]
    by_layout = defaultdict(list)
    for row in results:
        by_layout[row["layout_id"]].append(row["sdr"])
    return {
        "mean_sdr": float(np.mean(values)),
        "median_sdr": float(np.median(values)),
        "type1_median": float(
            np.median(by_layout.get("type1", [float("nan")]))
        ),
        "type2_median": float(
            np.median(by_layout.get("type2", [float("nan")]))
        ),
        "n": len(values),
    }
