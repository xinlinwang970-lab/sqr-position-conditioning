from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .audio import AudioConfig
from .checkpoint import capture_rng_state, load_training_checkpoint, restore_rng_state, save_training_checkpoint
from .coverage import build_coverage_banks, validate_coverage_banks
from .config import write_resolved_config
from .toy import BalancedToyBatchSampler, ToyMixture, ToyQueryDataset, write_toy_manifest
from .toy_eval import (
    decompose_two_query_errors,
    evaluate_query_predictions,
    write_error_decomposition,
    write_query_errors,
)
from .trainer import (
    _runtime_info,
    _summary_rows,
    _validate_bounded_training_config,
    _write_rows,
    predict_toy_waveforms,
    train_toy_updates,
)


EXPECTED_PARENT_STAGE = "round2_data_coverage"
EXPECTED_PARENT_RUN = "s144"
EXPECTED_PARENT_UPDATES = 2000


def run_s144_resume_diagnostic(
    parent_checkpoint: str | Path,
    out_dir: str | Path,
    device: str = "cpu",
    target_updates: int = 18000,
    milestones: tuple[int, ...] = (6000, 12000, 18000),
    max_additional_updates: int = 16000,
) -> dict:
    import torch

    parent = Path(parent_checkpoint)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty for a controlled resume run: {out}")
    out.mkdir(parents=True, exist_ok=True)

    parent_payload = _load_parent_payload(parent, map_location=device)
    parent_config = parent_payload["config"]
    _validate_bounded_training_config(parent_config)
    _validate_resume_policy(parent_payload, target_updates, milestones, max_additional_updates)

    policy = {
        "parent_checkpoint": str(parent),
        "parent_stage": parent_payload["metadata"].get("stage"),
        "parent_run_name": parent_payload["metadata"].get("run_name"),
        "start_updates": int(parent_payload["completed_updates"]),
        "target_updates": int(target_updates),
        "milestones": list(milestones),
        "max_additional_updates": int(max_additional_updates),
        "allowed_runtime_differences": ["out_dir", "target_updates", "milestones", "max_additional_updates"],
        "config_mutation_allowed": False,
        "strict_config_check": True,
        "strict_model_load": True,
    }
    (out / "runtime_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    write_resolved_config(parent_config, out / "resolved_config.yaml")
    runtime = _runtime_info(device)
    (out / "environment.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    _write_output_contract(parent_config, out / "output_target_contract.json")

    banks = build_coverage_banks(parent_config)
    coverage_checks = validate_coverage_banks(banks, parent_config)
    if not coverage_checks["ok"]:
        (out / "coverage_data_checks.json").write_text(json.dumps(coverage_checks, indent=2), encoding="utf-8")
        raise ValueError("Coverage data validation failed: " + "; ".join(coverage_checks["errors"]))
    (out / "coverage_data_checks.json").write_text(json.dumps(coverage_checks, indent=2), encoding="utf-8")
    data_checks = _write_and_check_resume_manifests(parent, out, banks)
    if not data_checks["ok"]:
        (out / "resume_data_identity_checks.json").write_text(json.dumps(data_checks, indent=2), encoding="utf-8")
        raise ValueError("Resume data identity checks failed: " + "; ".join(data_checks["errors"]))
    (out / "resume_data_identity_checks.json").write_text(json.dumps(data_checks, indent=2), encoding="utf-8")

    smoke = run_exact_resume_smoke(parent, out / "smoke_exact_resume", device=device)
    if not smoke["ok"]:
        raise ValueError("Exact resume smoke failed")

    components = build_s144_resume_components(parent_config, device=device, train_mixtures=banks["s144_train"])
    checkpoint = load_training_checkpoint(
        parent,
        components["model"],
        optimizer=components["optimizer"],
        expected_config=parent_config,
        map_location=device,
    )
    _validate_parent_checkpoint(checkpoint, parent)
    restore_rng_state(checkpoint["rng_state"])
    components["sampler"].load_state_dict(checkpoint["sampler_state"])
    components["completed_updates"] = int(checkpoint["completed_updates"])
    preflight = _preflight_parent_eval(
        components,
        parent_config,
        parent,
        out,
        {"train": banks["s144_train"], "d72": banks["d72"]},
        device,
    )
    if not preflight["ok"]:
        raise ValueError("Preflight parent metric comparison failed")

    checkpoint_paths: list[str] = []
    train_curve_rows: list[dict] = []
    train_validation_rows: list[dict] = []
    checkpoint_eval_rows: list[dict] = []
    decomposition_rows: list[dict] = []
    query_error_files: list[str] = []
    decomposition_files: list[str] = []
    gate_rows: list[dict] = []
    started = perf_counter()
    memory: dict[str, float] = {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    current = int(components["completed_updates"])
    for milestone in milestones:
        if milestone <= current:
            continue
        curves = train_toy_updates(
            components,
            milestone - current,
            device=device,
            after_update=lambda updated, row: _resume_after_update(
                updated,
                row,
                parent_config,
                out,
                device,
                banks["s144_train"],
                checkpoint_paths,
                train_validation_rows,
            ),
        )
        train_curve_rows.extend(curves)
        current = int(components["completed_updates"])
        eval_result = _evaluate_checkpoint(
            components,
            parent_config,
            out,
            device,
            current,
            {
                "train": banks["s144_train"],
                "d72": banks["d72"],
                "legacy_probe": banks["legacy_probe"],
            },
        )
        checkpoint_eval_rows.extend(eval_result["metric_rows"])
        decomposition_rows.extend(eval_result["decomposition_rows"])
        query_error_files.extend(eval_result["query_error_files"])
        decomposition_files.extend(eval_result["decomposition_files"])
        if eval_result["checkpoint_path"] not in checkpoint_paths:
            checkpoint_paths.append(eval_result["checkpoint_path"])
        gate = _gate_result(eval_result["summaries"]["train"], eval_result["summaries"]["d72"], current)
        gate_rows.append(gate)
        if gate["pass_train"] and gate["pass_d72"]:
            break

    if device == "cuda":
        torch.cuda.synchronize()
        memory = {
            "cuda_train_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
            "cuda_train_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 3),
        }
    seconds = perf_counter() - started
    _write_rows(train_curve_rows, out / "train_curves.csv")
    _write_rows(train_validation_rows, out / "train_validation_curves.csv")
    _write_rows(checkpoint_eval_rows, out / "checkpoint_eval_summary.csv")
    _write_rows(decomposition_rows, out / "error_decomposition_summary.csv")
    _write_rows(gate_rows, out / "gate_summary.csv")
    _write_rows(components["sampler"].access_count_rows(), out / "sampler_access_counts.csv")

    final_updates = int(components["completed_updates"])
    passed = bool(gate_rows and gate_rows[-1]["pass_train"] and gate_rows[-1]["pass_d72"])
    status = "ROUND2_S144_EXPOSURE_PASS_TOY_T_P_NOT_TOY_TP" if passed else "ROUND2_S144_EXPOSURE_COMPLETE_NOT_PASS_TOY_BEHAVIOR"
    result = {
        "status": status,
        "parent_checkpoint": str(parent),
        "start_updates": int(parent_payload["completed_updates"]),
        "target_updates": int(target_updates),
        "completed_updates": final_updates,
        "attempted_additional_updates": final_updates - int(parent_payload["completed_updates"]),
        "milestones": list(milestones),
        "device": device,
        "runtime": runtime,
        "train_seconds": round(seconds, 4),
        **memory,
        "policy": policy,
        "coverage_data_checks": coverage_checks,
        "resume_data_identity_checks": data_checks,
        "preflight_parent_eval": preflight,
        "exact_resume_smoke": smoke,
        "checkpoint_paths": checkpoint_paths,
        "query_error_files": query_error_files,
        "decomposition_files": decomposition_files,
        "gate_rows": gate_rows,
    }
    (out / "resume_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_exact_resume_smoke(parent_checkpoint: str | Path, out_dir: str | Path, device: str = "cpu") -> dict:
    parent = Path(parent_checkpoint)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = _load_parent_payload(parent, map_location=device)
    config = payload["config"]
    _validate_parent_checkpoint(payload, parent)
    banks = build_coverage_banks(config)

    with _deterministic_smoke_guard(device):
        continuous = build_s144_resume_components(config, device=device, train_mixtures=banks["s144_train"])
        _load_checkpoint_state(parent, continuous, config, device=device)
        train_toy_updates(continuous, 10, device=device)

        segmented = build_s144_resume_components(config, device=device, train_mixtures=banks["s144_train"])
        _load_checkpoint_state(parent, segmented, config, device=device)
        train_toy_updates(segmented, 5, device=device)
        split_path = out / "update_002005.pt"
        save_training_checkpoint(
            split_path,
            segmented["model"],
            segmented["optimizer"],
            config,
            completed_updates=int(segmented["completed_updates"]),
            rng_state=capture_rng_state(),
            sampler_state=segmented["sampler"].state_dict(),
            metadata={
                "stage": "round2_s144_resume_smoke",
                "run_name": "s144",
                "completed_updates": segmented["completed_updates"],
            },
        )

        resumed = build_s144_resume_components(config, device=device, train_mixtures=banks["s144_train"])
        checkpoint = load_training_checkpoint(
            split_path,
            resumed["model"],
            optimizer=resumed["optimizer"],
            expected_config=config,
            map_location=device,
        )
        restore_rng_state(checkpoint["rng_state"])
        resumed["sampler"].load_state_dict(checkpoint["sampler_state"])
        resumed["completed_updates"] = int(checkpoint["completed_updates"])
        train_toy_updates(resumed, 5, device=device)

    comparison = {
        "model_and_buffers_equal": _state_dicts_close(continuous["model"].state_dict(), resumed["model"].state_dict()),
        "optimizer_equal": _optimizer_states_close(continuous["optimizer"].state_dict(), resumed["optimizer"].state_dict()),
        "completed_updates_equal": int(continuous["completed_updates"]) == int(resumed["completed_updates"]),
        "sampler_state_equal": continuous["sampler"].state_dict() == resumed["sampler"].state_dict(),
        "next_batch_queries_equal": _peek_next_batch_queries(continuous) == _peek_next_batch_queries(resumed),
        "continuous_completed_updates": int(continuous["completed_updates"]),
        "resumed_completed_updates": int(resumed["completed_updates"]),
        "deterministic_smoke_guard": device == "cuda",
    }
    comparison["ok"] = all(
        bool(comparison[key])
        for key in (
            "model_and_buffers_equal",
            "optimizer_equal",
            "completed_updates_equal",
            "sampler_state_equal",
            "next_batch_queries_equal",
        )
    )
    (out / "smoke_resume_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


class _deterministic_smoke_guard:
    def __init__(self, device: str):
        self.enabled = device == "cuda"
        self.previous_deterministic = None
        self.previous_benchmark = None
        self.previous_cudnn_deterministic = None

    def __enter__(self):
        if not self.enabled:
            return self
        import torch

        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_benchmark = torch.backends.cudnn.benchmark
        self.previous_cudnn_deterministic = torch.backends.cudnn.deterministic
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        import torch

        torch.use_deterministic_algorithms(bool(self.previous_deterministic))
        torch.backends.cudnn.benchmark = bool(self.previous_benchmark)
        torch.backends.cudnn.deterministic = bool(self.previous_cudnn_deterministic)
        return False


def build_s144_resume_components(config: dict, device: str, train_mixtures: list[ToyMixture] | None = None) -> dict:
    import torch

    from .models.separator import Separator

    _validate_bounded_training_config(config)
    mixtures = train_mixtures if train_mixtures is not None else build_coverage_banks(config)["s144_train"]
    dataset = ToyQueryDataset(mixtures, AudioConfig.from_dict(config["audio"]))
    sampler = BalancedToyBatchSampler(dataset, batch_size=int(config["training"]["batch_size"]), tasks=("T", "P"))
    model = Separator(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["adam_betas"]),
        eps=float(config["training"]["adam_eps"]),
        weight_decay=float(config["training"]["weight_decay"]),
        foreach=bool(config["training"]["foreach"]),
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "dataset": dataset,
        "sampler": sampler,
        "completed_updates": 0,
    }


def _resume_after_update(
    components: dict,
    row: dict,
    config: dict,
    out: Path,
    device: str,
    train_mixtures: list[ToyMixture],
    checkpoint_paths: list[str],
    train_validation_rows: list[dict],
) -> None:
    loss = float(row["loss"])
    if not np.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at update {row['completed_updates']}: {loss}")
    completed = int(components["completed_updates"])
    if completed % 1000 != 0:
        return
    checkpoint_path = _save_resume_checkpoint(out, components, config, completed)
    checkpoint_paths.append(str(checkpoint_path))
    train_eval = _evaluate_without_training_state_change(
        components,
        config,
        {"train": train_mixtures},
        device,
        out_dir=None,
        completed_updates=completed,
    )
    train_validation_rows.extend(train_eval["metric_rows"])


def _evaluate_checkpoint(
    components: dict,
    config: dict,
    out: Path,
    device: str,
    completed_updates: int,
    eval_sets: dict[str, list[ToyMixture]],
) -> dict:
    checkpoint_path = _save_resume_checkpoint(out, components, config, completed_updates)
    result = _evaluate_without_training_state_change(
        components,
        config,
        eval_sets,
        device,
        out_dir=out,
        completed_updates=completed_updates,
        checkpoint_path=str(checkpoint_path),
    )
    result["checkpoint_path"] = str(checkpoint_path)
    return result


def _evaluate_without_training_state_change(
    components: dict,
    config: dict,
    eval_sets: dict[str, list[ToyMixture]],
    device: str,
    out_dir: Path | None,
    completed_updates: int,
    checkpoint_path: str | None = None,
) -> dict:
    before_sampler = components["sampler"].state_dict()
    before_rng = capture_rng_state()
    before_model_hash = _state_dict_hash(components["model"].state_dict())
    metric_rows: list[dict] = []
    decomposition_rows: list[dict] = []
    query_error_files: list[str] = []
    decomposition_files: list[str] = []
    summaries: dict[str, dict] = {}
    decompositions: dict[str, dict] = {}
    for eval_name, mixtures in eval_sets.items():
        predictions = predict_toy_waveforms(components["model"], mixtures, config, device=device)
        summary = evaluate_query_predictions(mixtures, predictions)
        decomposition = decompose_two_query_errors(mixtures, predictions)
        summaries[eval_name] = summary
        decompositions[eval_name] = decomposition
        metric_rows.extend(_tag_rows(_summary_rows(summary, completed_updates, checkpoint_path=checkpoint_path), eval_name))
        decomposition_rows.extend(_decomposition_summary_rows(decomposition, completed_updates, eval_name, checkpoint_path))
        if out_dir is not None:
            query_path = out_dir / f"query_errors_{eval_name}_{completed_updates:06d}.csv"
            decomp_path = out_dir / f"error_decomposition_{eval_name}_{completed_updates:06d}.csv"
            write_query_errors(summary, query_path)
            write_error_decomposition(decomposition, decomp_path)
            query_error_files.append(str(query_path))
            decomposition_files.append(str(decomp_path))
    after_model_hash = _state_dict_hash(components["model"].state_dict())
    after_rng = capture_rng_state()
    if components["sampler"].state_dict() != before_sampler:
        raise RuntimeError("Evaluation mutated sampler state")
    if before_model_hash != after_model_hash:
        raise RuntimeError("Evaluation mutated model parameters or buffers")
    if not _rng_states_equal(before_rng, after_rng):
        raise RuntimeError("Evaluation mutated RNG state")
    return {
        "metric_rows": metric_rows,
        "decomposition_rows": decomposition_rows,
        "summaries": summaries,
        "decompositions": decompositions,
        "query_error_files": query_error_files,
        "decomposition_files": decomposition_files,
    }


def _preflight_parent_eval(
    components: dict,
    config: dict,
    parent: Path,
    out: Path,
    eval_sets: dict[str, list[ToyMixture]],
    device: str,
) -> dict:
    result = _evaluate_without_training_state_change(
        components,
        config,
        eval_sets,
        device,
        out_dir=out / "preflight",
        completed_updates=int(components["completed_updates"]),
    )
    saved_path = parent.parent.parent / "behavior_summary.json"
    if not saved_path.exists():
        raise ValueError(f"Missing parent behavior summary for full-precision comparison: {saved_path}")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    rows = _compare_saved_metrics(saved["model_behavior_by_task_split"], result["summaries"])
    ok = all(row["ok"] for row in rows)
    comparison = {
        "ok": ok,
        "parent_behavior_summary": str(saved_path),
        "comparison_tolerance": 1.0e-6,
        "rows": rows,
    }
    (out / "preflight_parent_metric_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_rows(result["metric_rows"], out / "preflight_metric_summary.csv")
    _write_rows(result["decomposition_rows"], out / "preflight_error_decomposition_summary.csv")
    return comparison


def _compare_saved_metrics(saved_by_task_split: dict, summaries: dict[str, dict]) -> list[dict]:
    rows = []
    actual: dict[str, dict] = {}
    for summary in summaries.values():
        actual.update(summary["aggregate_by_task_split"])
    metrics = [
        "median_r_model",
        "median_r_blind",
        "median_r_zero",
        "mean_nearest_source_accuracy",
        "mean_unique_selection_accuracy",
        "mean_tie_ratio",
        "total_mixtures",
        "total_queries",
    ]
    for key in ("P_train", "T_train", "P_d72", "T_d72"):
        if key not in saved_by_task_split or key not in actual:
            rows.append({"task_split": key, "metric": "presence", "ok": False, "error": "missing"})
            continue
        for metric in metrics:
            saved_value = saved_by_task_split[key][metric]
            actual_value = actual[key][metric]
            diff = abs(float(saved_value) - float(actual_value))
            rows.append(
                {
                    "task_split": key,
                    "metric": metric,
                    "saved": saved_value,
                    "actual": actual_value,
                    "abs_diff": diff,
                    "ok": diff <= 1.0e-6,
                }
            )
    return rows


def _save_resume_checkpoint(out: Path, components: dict, config: dict, completed: int) -> Path:
    checkpoint_path = out / "checkpoints" / f"update_{completed:06d}.pt"
    save_training_checkpoint(
        checkpoint_path,
        components["model"],
        components["optimizer"],
        config,
        completed_updates=completed,
        rng_state=capture_rng_state(),
        sampler_state=components["sampler"].state_dict(),
        metadata={"stage": "round2_s144_resume", "run_name": "s144", "completed_updates": completed},
    )
    return checkpoint_path


def _gate_result(train_summary: dict, d72_summary: dict, completed_updates: int) -> dict:
    train_by = train_summary["aggregate_by_task_split"]
    d72_by = d72_summary["aggregate_by_task_split"]
    train_passes = [_train_gate(train_by[key]) for key in ("P_train", "T_train")]
    d72_passes = [_heldout_gate(d72_by[key]) for key in ("P_d72", "T_d72")]
    return {
        "completed_updates": int(completed_updates),
        "pass_train": all(train_passes),
        "pass_d72": all(d72_passes),
        "pass_p_train": train_passes[0],
        "pass_t_train": train_passes[1],
        "pass_p_d72": d72_passes[0],
        "pass_t_d72": d72_passes[1],
    }


def _train_gate(values: dict) -> bool:
    return bool(
        values["median_r_model"] <= 0.5 * values["median_r_blind"]
        and values["mean_unique_selection_accuracy"] >= 0.90
    )


def _heldout_gate(values: dict) -> bool:
    return bool(
        values["median_r_model"] < values["median_r_blind"]
        and values["mean_unique_selection_accuracy"] >= 0.75
    )


def _load_parent_payload(parent: Path, map_location: str):
    import torch

    payload = torch.load(parent, map_location=map_location, weights_only=False)
    _validate_parent_checkpoint(payload, parent)
    return payload


def _load_checkpoint_state(path: Path, components: dict, config: dict, device: str) -> dict:
    checkpoint = load_training_checkpoint(
        path,
        components["model"],
        optimizer=components["optimizer"],
        expected_config=config,
        map_location=device,
    )
    restore_rng_state(checkpoint["rng_state"])
    components["sampler"].load_state_dict(checkpoint["sampler_state"])
    components["completed_updates"] = int(checkpoint["completed_updates"])
    return checkpoint


def _validate_parent_checkpoint(checkpoint: dict, parent: Path) -> None:
    metadata = checkpoint.get("metadata", {})
    if checkpoint.get("format_version") != 2:
        raise ValueError(f"{parent}: expected training checkpoint format_version=2")
    if metadata.get("stage") != EXPECTED_PARENT_STAGE:
        raise ValueError(f"{parent}: parent checkpoint must be from {EXPECTED_PARENT_STAGE}")
    if metadata.get("run_name") != EXPECTED_PARENT_RUN:
        raise ValueError(f"{parent}: parent checkpoint must be S144")
    if int(checkpoint.get("completed_updates", -1)) != EXPECTED_PARENT_UPDATES:
        raise ValueError(f"{parent}: parent checkpoint must be update {EXPECTED_PARENT_UPDATES}")


def _validate_resume_policy(
    checkpoint: dict,
    target_updates: int,
    milestones: tuple[int, ...],
    max_additional_updates: int,
) -> None:
    start = int(checkpoint["completed_updates"])
    if target_updates <= start:
        raise ValueError("target_updates must be greater than parent completed_updates")
    if target_updates - start > max_additional_updates:
        raise ValueError("target_updates exceeds the max_additional_updates policy")
    if tuple(sorted(milestones)) != tuple(milestones):
        raise ValueError("milestones must be sorted")
    if milestones[-1] != target_updates:
        raise ValueError("last milestone must equal target_updates")
    if any(item <= start or item > target_updates for item in milestones):
        raise ValueError("all milestones must be within (start_updates, target_updates]")


def _write_and_check_resume_manifests(parent: Path, out: Path, banks: dict[str, list[ToyMixture]]) -> dict:
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "s144_train": banks["s144_train"],
        "legacy_probe": banks["legacy_probe"],
        "d72": banks["d72"],
    }
    rows = []
    errors = []
    for name, mixtures in targets.items():
        generated = manifest_dir / f"{name}.jsonl"
        write_toy_manifest(mixtures, generated)
        generated_hash = _file_sha256(generated)
        parent_manifest = _parent_manifest_path(parent, name)
        parent_hash = _file_sha256(parent_manifest) if parent_manifest.exists() else None
        query_order_match = None
        if parent_manifest.exists():
            query_order_match = _query_order_from_manifest(generated) == _query_order_from_manifest(parent_manifest)
            if generated_hash != parent_hash:
                errors.append(f"{name}: manifest hash mismatch")
            if not query_order_match:
                errors.append(f"{name}: query order mismatch")
        else:
            errors.append(f"{name}: missing parent manifest {parent_manifest}")
        rows.append(
            {
                "manifest": name,
                "generated": str(generated),
                "generated_sha256": generated_hash,
                "parent": str(parent_manifest),
                "parent_sha256": parent_hash,
                "query_order_match": query_order_match,
            }
        )
    _write_rows(rows, out / "resume_manifest_hashes.csv")
    return {"ok": not errors, "errors": errors, "rows": rows}


def _parent_manifest_path(parent: Path, name: str) -> Path:
    run_dir = parent.parent.parent
    if name == "s144_train":
        return run_dir / "train_manifest.jsonl"
    return run_dir / f"{name}_manifest.jsonl"


def _query_order_from_manifest(path: Path) -> list[str]:
    order: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            order.extend(query["query_id"] for query in record["queries"])
    return order


def _write_output_contract(config: dict, path: Path) -> None:
    contract = {
        "head_output": "direct packed-complex spectrogram estimate",
        "loss_expression": "mean((prediction.float() - target_spec.float()) ** 2)",
        "training_target": "STFT(target waveform) packed as [real_left, real_right, imag_left, imag_right]",
        "inference_path": "predicted packed-complex spectrogram -> ISTFT",
        "mask_multiply_path_present": False,
        "config_output_mode": config["model"].get("output_mode"),
        "loss_config": config["training"].get("loss"),
        "input_shape_convention": "[B, 4, T, F]",
    }
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _decomposition_summary_rows(
    decomposition: dict,
    completed_updates: int,
    eval_set: str,
    checkpoint_path: str | None,
) -> list[dict]:
    rows = []
    for key, values in decomposition["aggregate_by_task_split"].items():
        task, split = key.split("_", 1)
        row = {
            "eval_set": eval_set,
            "completed_updates": int(completed_updates),
            "task": task,
            "split": split,
            **values,
        }
        if checkpoint_path is not None:
            row["checkpoint_path"] = checkpoint_path
        rows.append(row)
    return rows


def _tag_rows(rows: list[dict], eval_set: str) -> list[dict]:
    return [{"eval_set": eval_set, **row} for row in rows]


def _peek_next_batch_queries(components: dict) -> list[str]:
    state = components["sampler"].state_dict()
    indices = components["sampler"].next_batch_indices()
    query_ids = [components["dataset"].records[index][1].query_id for index in indices]
    components["sampler"].load_state_dict(state)
    return query_ids


def _state_dict_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_dicts_close(left: dict, right: dict) -> bool:
    import torch

    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key].detach().cpu()
        right_value = right[key].detach().cpu()
        if torch.is_floating_point(left_value) or torch.is_complex(left_value):
            if not torch.allclose(left_value, right_value, atol=1.0e-6):
                return False
        elif not torch.equal(left_value, right_value):
            return False
    return True


def _optimizer_states_close(left: dict, right: dict) -> bool:
    import torch

    if left.keys() != right.keys():
        return False
    if left["param_groups"] != right["param_groups"]:
        return False
    if left["state"].keys() != right["state"].keys():
        return False
    for param_id, left_state in left["state"].items():
        right_state = right["state"][param_id]
        if left_state.keys() != right_state.keys():
            return False
        for key, left_value in left_state.items():
            right_value = right_state[key]
            if hasattr(left_value, "detach"):
                if not torch.allclose(left_value.detach().cpu(), right_value.detach().cpu(), atol=1.0e-6):
                    return False
            elif left_value != right_value:
                return False
    return True


def _rng_states_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    import torch

    if left["python"] != right["python"]:
        return False
    if left["numpy"][0] != right["numpy"][0]:
        return False
    if not np.array_equal(left["numpy"][1], right["numpy"][1]):
        return False
    if tuple(left["numpy"][2:]) != tuple(right["numpy"][2:]):
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda = left.get("torch_cuda")
    right_cuda = right.get("torch_cuda")
    if left_cuda is None or right_cuda is None:
        return left_cuda is right_cuda
    if len(left_cuda) != len(right_cuda):
        return False
    return all(torch.equal(a, b) for a, b in zip(left_cuda, right_cuda))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
