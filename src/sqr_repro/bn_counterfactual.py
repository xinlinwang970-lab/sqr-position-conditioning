from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import load_training_checkpoint
from .config import load_config, write_resolved_config
from .coverage import build_coverage_banks, validate_coverage_banks
from .resume import build_s144_resume_components
from .toy import BalancedToyBatchSampler, ToyQueryDataset, collate_toy_batch
from .toy_eval import decompose_two_query_errors, evaluate_query_predictions, write_error_decomposition, write_query_errors
from .trainer import predict_toy_waveforms


def run_bn_counterfactual(
    config_path: str | Path,
    source_runs: dict[str, str | Path],
    out_dir: str | Path,
    device: str = "cuda",
    calibration_batch_size: int = 4,
) -> dict:
    import torch

    config = load_config(config_path)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if calibration_batch_size != 4:
        raise ValueError("BN counterfactual requires the pre-registered calibration batch size B=4")
    banks = build_coverage_banks(config)
    checks = validate_coverage_banks(banks, config)
    if not checks["ok"]:
        raise ValueError("Coverage data validation failed: " + "; ".join(checks["errors"]))
    inventory = _bn_inventory(config, device)
    (out / "bn_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    if not inventory["applicable"]:
        raise RuntimeError("No BatchNorm running statistics available for this counterfactual")

    calibration = _build_calibration_batches(config, banks["s144_train"], calibration_batch_size, device)
    _write_rows(calibration["rows"], out / "calibration_query_order.csv")
    (out / "calibration_policy.json").write_text(json.dumps(calibration["policy"], indent=2), encoding="utf-8")
    if not calibration["valid"]:
        raise RuntimeError("Calibration query coverage is invalid")

    source_results = {}
    for source_name, source_dir in source_runs.items():
        source_root = Path(source_dir)
        source_checkpoint = source_root / "checkpoints" / f"update_{_source_update(source_name):06d}.pt"
        if source_name == "parent18000":
            source_checkpoint = source_root / "checkpoints" / "update_018000.pt"
        elif source_name == "B0_20000":
            source_checkpoint = source_root / "checkpoints" / "update_020000.pt"
        elif source_name == "B1_20000":
            source_checkpoint = source_root / "checkpoints" / "update_020000.pt"
        if not source_checkpoint.exists():
            raise FileNotFoundError(source_checkpoint)
        source_results[source_name] = _run_source(
            source_name, source_checkpoint, config, banks, calibration, out / source_name, device
        )

    result = {
        "status": "ROUND2_BN_STATE_COUNTERFACTUAL_COMPLETE_NO_PARAMETER_UPDATES",
        "parameter_updates": 0,
        "device": device,
        "source_runs": {key: str(value) for key, value in source_runs.items()},
        "coverage_checks": checks,
        "bn_inventory": inventory,
        "calibration": calibration["policy"],
        "sources": source_results,
        "formal_score": "raw only",
        "projected_score": "not used",
    }
    write_resolved_config(config, out / "resolved_config.yaml")
    (out / "counterfactual_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _run_source(source_name, checkpoint_path, config, banks, calibration, out, device):
    import torch

    out.mkdir(parents=True, exist_ok=True)
    components = _load_components(checkpoint_path, config, banks["s144_train"], device)
    model = components["model"]
    c0_before = _state_snapshot(model)
    c0 = _evaluate_variant(model, config, banks, out / "C0", device, source_name, "C0")
    c0_after = _state_snapshot(model)
    if c0_before["all_hash"] != c0_after["all_hash"]:
        raise RuntimeError(f"{source_name} C0 evaluation changed model state")
    comparison = _compare_source_metric_if_available(source_name, c0, checkpoint_path.parent.parent)
    if not comparison["ok"]:
        raise RuntimeError(f"{source_name} C0 metric comparison failed")
    c0["source_metric_comparison"] = comparison

    c1_model = _load_components(checkpoint_path, config, banks["s144_train"], device)["model"]
    before = _state_snapshot(c1_model)
    bn_modules = _bn_modules(c1_model)
    original_modes = {name: module.training for name, module in c1_model.named_modules()}
    original_momentum = {name: module.momentum for name, module in bn_modules.items()}
    c1_model.eval()
    for name, module in bn_modules.items():
        module.reset_running_stats()
        module.momentum = None
        module.train()
    with torch.inference_mode():
        for batch in calibration["batches"]:
            c1_model(batch["spec"].to(device), batch["timbre_id"].to(device), batch["position_id"].to(device))
    for name, module in bn_modules.items():
        module.momentum = original_momentum[name]
    c1_model.eval()
    after_calibration = _state_snapshot(c1_model)
    changed = _changed_state_names(before, after_calibration)
    allowed = {f"{name}.running_mean" for name in bn_modules}
    allowed.update(f"{name}.running_var" for name in bn_modules)
    allowed.update(f"{name}.num_batches_tracked" for name in bn_modules)
    if not set(changed).issubset(allowed):
        raise RuntimeError(f"{source_name} C1 changed non-BN state: {sorted(set(changed) - allowed)}")
    if not all(not module.training for module in c1_model.modules()):
        raise RuntimeError(f"{source_name} C1 model was not restored to eval mode")
    c1 = _evaluate_variant(c1_model, config, banks, out / "C1", device, source_name, "C1")
    after_eval = _state_snapshot(c1_model)
    if _changed_state_names(after_calibration, after_eval):
        raise RuntimeError(f"{source_name} C1 evaluation changed state")
    _write_rows(
        [{"name": name, "changed": name in changed, "allowed": name in allowed} for name in sorted(set(before["names"]) | set(after_calibration["names"]))],
        out / "state_change_audit.csv",
    )
    return {
        "source": source_name,
        "checkpoint": str(checkpoint_path),
        "C0": c0,
        "C1": c1,
        "changed_state_names": changed,
        "allowed_changed_state_names": sorted(allowed),
        "parameter_hash_unchanged": before["parameter_hash"] == after_calibration["parameter_hash"],
        "buffer_audit": str(out / "state_change_audit.csv"),
    }


def _load_components(checkpoint_path, config, train_mixtures, device):
    from .checkpoint import load_training_checkpoint

    components = build_s144_resume_components(config, device=device, train_mixtures=train_mixtures)
    checkpoint = load_training_checkpoint(
        checkpoint_path,
        components["model"],
        optimizer=components["optimizer"],
        expected_config=config,
        map_location=device,
    )
    if int(checkpoint["completed_updates"]) not in {18000, 20000}:
        raise ValueError(f"Unexpected source checkpoint update: {checkpoint_path}")
    return components


def _evaluate_variant(model, config, banks, out, device, source_name, variant):
    out.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for eval_name, mixtures in (("s144_train", banks["s144_train"]), ("d72", banks["d72"])):
        predictions = predict_toy_waveforms(model, mixtures, config, device=device)
        summary = evaluate_query_predictions(mixtures, predictions)
        decomp = decompose_two_query_errors(mixtures, predictions)
        write_query_errors(summary, out / f"query_errors_{eval_name}.csv")
        write_error_decomposition(decomp, out / f"error_decomposition_{eval_name}.csv")
        for key, values in summary["aggregate_by_task_split"].items():
            task, split = key.split("_", 1)
            metric_rows.append({"source": source_name, "variant": variant, "eval_set": eval_name, "task": task, "split": split, **values})
    _write_rows(metric_rows, out / "metric_summary.csv")
    return {"metric_summary": str(out / "metric_summary.csv"), "query_errors": str(out)}


def _bn_inventory(config, device):
    import torch
    from .models.separator import Separator

    model = Separator(config).to(device)
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            rows.append({"name": name, "type": type(module).__name__, "track_running_stats": bool(module.track_running_stats), "momentum": module.momentum, "num_features": module.num_features})
    return {"applicable": bool(rows) and all(row["track_running_stats"] for row in rows), "count": len(rows), "modules": rows}


def _bn_modules(model):
    import torch

    return {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats}


def _build_calibration_batches(config, mixtures, batch_size, device):
    dataset = ToyQueryDataset(mixtures, __import__("sqr_repro.audio", fromlist=["AudioConfig"]).AudioConfig.from_dict(config["audio"]))
    sampler = BalancedToyBatchSampler(dataset, batch_size=batch_size, tasks=("T", "P"))
    rows = []
    batches = []
    expected = {query.query_id for mixture in mixtures for query in mixture.queries}
    seen = []
    for batch_index in range(144):
        indices = sampler.next_batch_indices()
        batch = collate_toy_batch([dataset[index] for index in indices])
        batches.append(batch)
        seen.extend(batch["query_id"])
        rows.append({"batch_index": batch_index, "mixture_ids": ",".join(batch["mixture_id"]), "query_ids": ",".join(batch["query_id"]), "tasks": ",".join(batch["task"]), "dataset_indices": ",".join(str(index) for index in indices)})
    policy = {"batch_size": batch_size, "num_batches": 144, "query_count": len(seen), "unique_query_count": len(set(seen)), "uses_d72": False, "uses_targets_for_stats": False, "sampler_tasks": ["T", "P"]}
    return {"valid": len(seen) == len(expected) == len(set(seen)) and set(seen) == expected, "rows": rows, "batches": batches, "policy": policy}


def _state_snapshot(model):
    names = []
    parameters = hashlib.sha256()
    buffers = hashlib.sha256()
    all_state = hashlib.sha256()
    for name, tensor in sorted(model.named_parameters()):
        names.append(name)
        raw = np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes()
        parameters.update(name.encode()); parameters.update(raw); all_state.update(name.encode()); all_state.update(raw)
    for name, tensor in sorted(model.named_buffers()):
        names.append(name)
        raw = np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes()
        buffers.update(name.encode()); buffers.update(raw); all_state.update(name.encode()); all_state.update(raw)
    return {"names": names, "parameter_hash": parameters.hexdigest(), "buffer_hash": buffers.hexdigest(), "all_hash": all_state.hexdigest(), "state_hashes": {name: _tensor_hash(tensor) for name, tensor in model.state_dict().items()}}


def _changed_state_names(before, after):
    return sorted(name for name in before["state_hashes"] if before["state_hashes"].get(name) != after["state_hashes"].get(name))


def _tensor_hash(tensor):
    return hashlib.sha256(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes()).hexdigest()


def _compare_source_metric_if_available(source_name, result, run_root):
    source_file = run_root / ("checkpoint_eval_summary.csv" if source_name == "parent18000" else "metric_summary.csv")
    actual_file = Path(result["metric_summary"])
    if not source_file.exists() or not actual_file.exists():
        return {"ok": False, "reason": "missing source or C0 metric file", "source_file": str(source_file), "actual_file": str(actual_file)}
    expected_rows = list(csv.DictReader(source_file.open("r", newline="", encoding="utf-8")))
    actual_rows = list(csv.DictReader(actual_file.open("r", newline="", encoding="utf-8")))
    if source_name == "parent18000":
        expected_rows = [row for row in expected_rows if row.get("completed_updates") == "18000" and row.get("eval_set") in {"train", "d72"}]
        expected = {("s144_train" if row["eval_set"] == "train" else row["eval_set"], row["task"]): row for row in expected_rows}
        actual = {(row["eval_set"], row["task"]): row for row in actual_rows}
    else:
        expected_rows = [row for row in expected_rows if row.get("update") == "20000"]
        expected = {("s144_train" if row["eval_set"] == "train" else row["eval_set"], row["task"]): row for row in expected_rows}
        actual = {(row["eval_set"], row["task"]): row for row in actual_rows}
    metrics = ["median_r_model", "median_r_blind", "median_r_zero", "mean_unique_selection_accuracy", "mean_tie_ratio"]
    rows = []
    for key in sorted(set(expected) | set(actual)):
        for metric in metrics:
            if key not in expected or key not in actual:
                rows.append({"eval_task": str(key), "metric": metric, "ok": False, "reason": "missing row"})
                continue
            diff = abs(float(expected[key][metric]) - float(actual[key][metric]))
            rows.append({"eval_task": str(key), "metric": metric, "source": expected[key][metric], "C0": actual[key][metric], "abs_diff": diff, "ok": diff <= 1.0e-5})
    return {"ok": all(row["ok"] for row in rows), "tolerance": 1.0e-5, "source_file": str(source_file), "actual_file": str(actual_file), "rows": rows}


def _source_update(source_name):
    return 18000 if source_name == "parent18000" else 20000


def _write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
