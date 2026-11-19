from __future__ import annotations

from copy import deepcopy

import torch

from .models.separator import Separator


def build_paired_initialization(config: dict, device: str = "cpu") -> dict:
    """Build C0 and C0-SP with identical shared parameters and strict key accounting."""
    score_config = deepcopy(config.get("score") or {})
    if not score_config.get("enabled", False):
        raise ValueError("Paired initialization requires score.enabled=true in the supplied C0-SP config")
    base_config = deepcopy(config)
    base_config.pop("score", None)
    score_config["enabled"] = True
    score_config.setdefault("num_pitches", 128)
    score_config.setdefault("hidden_channels", 32)
    score_config.setdefault("injection", "pre_middle_add")
    enabled_config = deepcopy(config)
    enabled_config["score"] = score_config

    base = Separator(base_config).to(device)
    enabled = Separator(enabled_config).to(device)
    base_state = base.state_dict()
    enabled_state = enabled.state_dict()
    shared = sorted(set(base_state) & set(enabled_state))
    missing_shared = sorted(set(base_state) - set(shared))
    unexpected_shared = sorted(set(enabled_state) - set(shared))
    shape_mismatches = [key for key in shared if base_state[key].shape != enabled_state[key].shape]
    if missing_shared or shape_mismatches:
        raise ValueError(
            f"Paired initialization shared-key mismatch: missing={missing_shared}, shape_mismatches={shape_mismatches}"
        )
    if not unexpected_shared or not all(key.startswith("score_encoder.") for key in unexpected_shared):
        raise ValueError(f"Unexpected non-score keys in paired initialization: {unexpected_shared}")
    enabled.load_state_dict(base_state, strict=False)
    if not enabled.score_encoder.projection_is_zero():
        raise ValueError("Score projection must remain zero-initialized")
    for key in shared:
        if not torch.equal(base.state_dict()[key], enabled.state_dict()[key]):
            raise ValueError(f"Shared parameter mismatch after paired initialization: {key}")
    return {
        "base_config": base_config,
        "score_config": enabled_config,
        "base_model": base,
        "score_model": enabled,
        "shared_keys": shared,
        "score_only_keys": unexpected_shared,
    }


def verify_paired_step_zero(pair: dict, spec, timbre_id, position_id, score_roll, atol: float = 0.0) -> dict:
    base = pair["base_model"].eval()
    enabled = pair["score_model"].eval()
    with torch.inference_mode():
        base_output = base(spec, timbre_id, position_id)
        score_output = enabled(spec, timbre_id, position_id, score_roll=score_roll)
    difference = (base_output - score_output).abs()
    max_abs = float(difference.max().cpu())
    if max_abs > atol:
        raise AssertionError(f"Paired step-zero outputs differ: max_abs={max_abs}, tolerance={atol}")
    return {
        "output_shape": list(base_output.shape),
        "max_abs_difference": max_abs,
        "mean_abs_difference": float(difference.mean().cpu()),
        "score_projection_zero": pair["score_model"].score_encoder.projection_is_zero(),
    }
