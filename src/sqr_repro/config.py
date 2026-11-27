from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "paths",
    "data",
    "audio",
    "model",
    "training",
    "inference",
    "profiling",
    "toy",
    "score",
    "score_s",
    "score_coverage",
}

SECTION_KEYS = {
    "experiment": {
        "name",
        "status",
        "seed",
        "data_seed",
        "sampler_seed",
        "model_seed",
        "test_only",
        "run_mode",
        "architecture_status",
        "architecture_evidence_path",
    },
    "paths": {"run_root"},
    "data": {
        "root",
        "manifest",
        "train_manifest",
        "id_probe_manifest",
        "compositional_ood_manifest",
        "manifest_bundle",
        "frozen_cache_root",
        "target_sampling",
        "pair_cycle",
        "augmentation",
        "num_workers",
        "pin_memory",
        "drop_last",
    },
    "audio": {
        "sample_rate",
        "channels",
        "n_fft",
        "win_length",
        "hop_length",
        "num_frames",
        "center",
        "pad_mode",
        "window",
        "window_periodic",
        "normalized",
        "pack_order",
        "stft_dtype",
        "segment_samples",
    },
    "model": {
        "n_blocks",
        "base_channels",
        "stage_channels",
        "tfc_growth_rate",
        "internal_channels",
        "tfc_layers",
        "kernel_size",
        "tdf_bottleneck_factor",
        "tdf_min_units",
        "tdf_bias",
        "condition_route",
        "embedding_dim",
        "control_hidden_dims",
        "control_dropout",
        "control_hidden_norm",
        "control_output_norm",
        "modulation",
        "modulation_position",
        "output_mode",
        "activation_checkpointing",
        "architecture_version",
        "num_timbres",
        "num_positions",
    },
    "training": {
        "batch_size",
        "gradient_accumulation_steps",
        "precision",
        "loss",
        "optimizer",
        "learning_rate",
        "adam_betas",
        "adam_eps",
        "weight_decay",
        "foreach",
        "gradient_clip_norm",
        "scheduler",
        "max_attempted_updates",
        "validate_every_updates",
        "checkpoint_every_updates",
        "monitor",
        "monitor_mode",
        "compile",
    },
    "inference": {
        "evaluation_protocol",
        "batch_size",
        "overlap_ratio_requested",
        "align_stride_to_stft_hop",
        "blend_window",
        "blend_weight_floor",
        "keep_original_length",
        "save_subtype",
    },
    "profiling": {
        "warmup_updates",
        "measured_updates",
        "reserved_memory_budget_gib",
        "budget_is_advisory",
    },
    "score": {
        "enabled",
        "num_pitches",
        "hidden_channels",
        "injection",
    },
    "score_s": {
        "enabled",
        "train_mixtures",
        "probe_mixtures",
        "num_patterns",
    },
    "score_coverage": {
        "enabled",
        "train_pairs",
        "compositional_pairs",
        "train_mixtures_per_pair",
        "id_probe_mixtures_per_pair",
        "compositional_mixtures_per_pair",
        "max_updates",
        "checkpoint_updates",
    },
    "toy": {
        "enabled",
        "tasks",
        "train_mixtures_per_task",
        "probe_mixtures_per_task",
        "tp_train_mixtures",
        "tp_probe_mixtures",
        "position_degrees",
        "min_note_events",
        "max_note_events",
        "sanity_min_target_energy",
        "sanity_max_abs_correlation",
    },
}


class ConfigError(ValueError):
    pass


GROWTH24_PROJECTED_ARCHITECTURES = {
    "growth24_projected_a_v1": {
        "stage_channels": (124, 124, 124, 124),
        "control_hidden_dims": (64, 128, 384),
        "condition_route": "split_stage_paper",
    },
    "growth24_projected_b_v1": {
        "stage_channels": (96, 144, 144, 240),
        "control_hidden_dims": (64, 128, 256),
        "condition_route": "split_stage_paper",
    },
    "growth24_projected_c_v1": {
        "stage_channels": (96, 192, 336, 456),
        "control_hidden_dims": (64, 128, 64),
        "condition_route": "split_stage_paper",
    },
    "growth24_projected_a_v1_mdp": {
        "stage_channels": (124, 124, 124, 124),
        "control_hidden_dims": (64, 128, 384),
        "condition_route": "mid_dual_paper",
    },
    "growth24_projected_b_v1_mdp": {
        "stage_channels": (96, 144, 144, 240),
        "control_hidden_dims": (64, 128, 256),
        "condition_route": "mid_dual_paper",
    },
    "growth24_projected_c_v1_mdp": {
        "stage_channels": (96, 192, 336, 456),
        "control_hidden_dims": (64, 128, 64),
        "condition_route": "mid_dual_paper",
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {config_path}")
    resolved = deepcopy(data)
    normalize_config(resolved)
    validate_config(resolved)
    resolved["audio"]["segment_samples"] = segment_samples(resolved)
    return resolved


def write_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )


def normalize_config(config: dict[str, Any]) -> None:
    model = config.get("model")
    if not isinstance(model, dict):
        return
    if "base_channels" not in model and "internal_channels" in model:
        model["base_channels"] = model["internal_channels"]
    if "tfc_growth_rate" not in model and "internal_channels" in model:
        model["tfc_growth_rate"] = model["internal_channels"]
    if "internal_channels" not in model and "base_channels" in model:
        model["internal_channels"] = model["base_channels"]


def validate_config(config: dict[str, Any]) -> None:
    _reject_unknown("top level", config, TOP_LEVEL_KEYS)
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    for section, allowed in SECTION_KEYS.items():
        value = config.get(section)
        if section in {"toy", "score", "score_s", "score_coverage"} and value is None:
            continue
        if not isinstance(value, dict):
            raise ConfigError(f"{section} must be a mapping")
        _reject_unknown(section, value, allowed)

    audio = config["audio"]
    model = config["model"]
    training = config["training"]
    inference = config["inference"]
    toy = config.get("toy")
    score = config.get("score")
    score_s = config.get("score_s")
    score_coverage = config.get("score_coverage")

    for key in ("seed", "data_seed", "sampler_seed", "model_seed"):
        if key in config["experiment"] and not isinstance(
            config["experiment"][key], int
        ):
            raise ConfigError(f"experiment.{key} must be an integer")

    if audio["channels"] != 2:
        raise ConfigError("Only stereo audio is supported")
    legacy_inference_values = {
        "evaluation_protocol": "legacy_v1",
        "overlap_ratio_requested": 0.5,
        "align_stride_to_stft_hop": False,
        "blend_window": "hann",
        "blend_weight_floor": 1.0e-8,
        "keep_original_length": True,
    }
    legacy_mismatches = {
        key: {"expected": expected, "actual": inference.get(key)}
        for key, expected in legacy_inference_values.items()
        if inference.get(key) != expected
    }
    if legacy_mismatches:
        raise ConfigError(
            "legacy_v1 fixes the historical evaluation settings: "
            f"{legacy_mismatches}"
        )
    if audio["num_frames"] % 8 != 0:
        raise ConfigError(
            "num_frames must be divisible by 8 for three downsampling stages"
        )
    if (audio["n_fft"] // 2 + 1 - 1) % 8 != 0:
        raise ConfigError("n_fft must yield F-1 divisible by 8")
    if model["n_blocks"] != 7:
        raise ConfigError("n_blocks must be 7: 3 encoder + 1 middle + 3 decoder")
    if int(model["base_channels"]) <= 0 or int(model["tfc_growth_rate"]) <= 0:
        raise ConfigError("base_channels and tfc_growth_rate must be positive")
    architecture_version = str(model.get("architecture_version", ""))
    stage_channels = model.get("stage_channels")
    projected_spec = GROWTH24_PROJECTED_ARCHITECTURES.get(architecture_version)
    candidate_architecture = (
        architecture_version == "paper_route_w24_v1" or projected_spec is not None
    )
    if candidate_architecture:
        fixed_model_values = {
            "n_blocks": 7,
            "tfc_layers": 5,
            "kernel_size": [3, 3],
            "num_timbres": 3,
            "num_positions": 4,
            "embedding_dim": 32,
            "condition_route": projected_spec.get(
                "condition_route", "split_stage_paper"
            )
            if projected_spec is not None
            else "split_stage_paper",
            "tdf_bottleneck_factor": 16,
            "tdf_min_units": 16,
            "tdf_bias": False,
            "control_hidden_dims": (
                list(projected_spec["control_hidden_dims"])
                if projected_spec is not None
                else [64, 128, 256]
            ),
            "control_dropout": 0.1,
            "control_hidden_norm": "layer_norm",
            "control_output_norm": "none",
        }
        mismatches = {
            key: {"expected": expected, "actual": model.get(key)}
            for key, expected in fixed_model_values.items()
            if model.get(key) != expected
        }
        if audio["n_fft"] != 2048:
            mismatches["audio.n_fft"] = {"expected": 2048, "actual": audio["n_fft"]}
        if mismatches:
            raise ConfigError(
                f"{architecture_version} fixes its audited model settings: {mismatches}"
            )
    if projected_spec is not None:
        if not isinstance(stage_channels, list) or len(stage_channels) != 4:
            raise ConfigError(f"{architecture_version} requires four stage_channels")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in stage_channels
        ):
            raise ConfigError("stage_channels must contain integer widths")
        try:
            stage_channels = [int(value) for value in stage_channels]
        except (TypeError, ValueError) as exc:
            raise ConfigError("stage_channels must contain integers") from exc
        if any(value <= 0 for value in stage_channels):
            raise ConfigError("stage_channels must be positive")
        expected_channels = list(projected_spec["stage_channels"])
        if stage_channels != expected_channels:
            raise ConfigError(
                f"{architecture_version} fixes stage_channels to {expected_channels}"
            )
        if (
            int(model["base_channels"]) != stage_channels[0]
            or int(model["tfc_growth_rate"]) != 24
        ):
            raise ConfigError(
                f"{architecture_version} requires base_channels={stage_channels[0]} "
                "and tfc_growth_rate=24"
            )
    elif stage_channels is not None:
        raise ConfigError(
            "stage_channels are only supported by growth24 projected candidates"
        )
    elif int(model["base_channels"]) != int(model["tfc_growth_rate"]):
        raise ConfigError(
            "base_channels and tfc_growth_rate are separate config fields, but this implementation keeps them equal "
            "until a LaSAFT-compatible transition/projection rule is confirmed"
        )
    if architecture_version == "paper_route_w24_v1":
        if int(model["base_channels"]) != 24 or int(model["tfc_growth_rate"]) != 24:
            raise ConfigError(
                "paper_route_w24_v1 requires base_channels=tfc_growth_rate=24"
            )
    if model["condition_route"] not in {
        "split_stage",
        "split_stage_paper",
        "mean_decoder",
        "mid_dual_paper",
    }:
        raise ConfigError(
            "condition_route must be split_stage, split_stage_paper, mean_decoder, or mid_dual_paper"
        )
    if model["modulation"] != "gpocm":
        raise ConfigError("Only gpocm modulation is implemented in M0-M2")
    if model["output_mode"] != "complex_mapping":
        raise ConfigError("Only direct complex_mapping output is implemented")
    for key, default in (("num_timbres", 3), ("num_positions", 4)):
        if key in model and int(model[key]) <= 0:
            raise ConfigError(f"model.{key} must be positive")
    if score is not None:
        if not isinstance(score["enabled"], bool):
            raise ConfigError("score.enabled must be a boolean")
        if int(score["num_pitches"]) <= 0 or int(score["hidden_channels"]) <= 0:
            raise ConfigError(
                "score.num_pitches and score.hidden_channels must be positive"
            )
        if score["injection"] != "pre_middle_add":
            raise ConfigError("score.injection must be pre_middle_add")
    if score_s is not None:
        if not isinstance(score_s["enabled"], bool):
            raise ConfigError("score_s.enabled must be a boolean")
        for key in ("train_mixtures", "probe_mixtures", "num_patterns"):
            if int(score_s[key]) < 1:
                raise ConfigError(f"score_s.{key} must be positive")
    if score_coverage is not None:
        if not isinstance(score_coverage["enabled"], bool):
            raise ConfigError("score_coverage.enabled must be a boolean")
        train_pairs = {
            tuple(sorted(int(value) for value in pair))
            for pair in score_coverage["train_pairs"]
        }
        compositional_pairs = {
            tuple(sorted(int(value) for value in pair))
            for pair in score_coverage["compositional_pairs"]
        }
        if any(
            len(pair) != 2
            or pair[0] == pair[1]
            or any(value < 0 or value > 5 for value in pair)
            for pair in train_pairs | compositional_pairs
        ):
            raise ConfigError(
                "score_coverage pairs must contain two distinct primitive ids in [0, 5]"
            )
        if train_pairs & compositional_pairs:
            raise ConfigError(
                "score_coverage train and compositional pairs must be disjoint"
            )
        for key in (
            "train_mixtures_per_pair",
            "id_probe_mixtures_per_pair",
            "compositional_mixtures_per_pair",
            "max_updates",
        ):
            if int(score_coverage[key]) < 1:
                raise ConfigError(f"score_coverage.{key} must be positive")
        updates = {int(value) for value in score_coverage["checkpoint_updates"]}
        if (
            not updates
            or min(updates) < 1
            or max(updates) > int(score_coverage["max_updates"])
        ):
            raise ConfigError(
                "score_coverage.checkpoint_updates must be positive and within max_updates"
            )
    if (
        training["batch_size"] < 2
        and model["control_hidden_norm"] == "batch_norm"
        and not config["experiment"].get("test_only", False)
    ):
        raise ConfigError("BatchNorm1d controller training requires batch_size >= 2")
    if toy is not None:
        _validate_toy_config(toy)


def segment_samples(config: dict[str, Any]) -> int:
    audio = config["audio"]
    return (int(audio["num_frames"]) - 1) * int(audio["hop_length"])


def input_frequency_bins(config: dict[str, Any]) -> int:
    return int(config["audio"]["n_fft"]) // 2 + 1


def _reject_unknown(name: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {name}: {', '.join(unknown)}")


def _validate_toy_config(toy: dict[str, Any]) -> None:
    if not isinstance(toy["enabled"], bool):
        raise ConfigError("toy.enabled must be a boolean")
    tasks = toy["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ConfigError("toy.tasks must be a non-empty list")
    invalid = sorted(set(tasks) - {"T", "P", "TP"})
    if invalid:
        raise ConfigError(f"Unsupported toy task(s): {', '.join(invalid)}")
    for key in (
        "train_mixtures_per_task",
        "probe_mixtures_per_task",
        "tp_train_mixtures",
        "tp_probe_mixtures",
        "min_note_events",
        "max_note_events",
    ):
        if int(toy[key]) < 0:
            raise ConfigError(f"toy.{key} must be non-negative")
    if int(toy["min_note_events"]) < 1:
        raise ConfigError("toy.min_note_events must be positive")
    if int(toy["max_note_events"]) < int(toy["min_note_events"]):
        raise ConfigError("toy.max_note_events must be >= toy.min_note_events")
    position_degrees = toy["position_degrees"]
    if not isinstance(position_degrees, list) or len(position_degrees) != 4:
        raise ConfigError("toy.position_degrees must contain exactly four values")
    if float(toy["sanity_min_target_energy"]) <= 0:
        raise ConfigError("toy.sanity_min_target_energy must be positive")
    if not 0 < float(toy["sanity_max_abs_correlation"]) <= 1:
        raise ConfigError("toy.sanity_max_abs_correlation must be in (0, 1]")
