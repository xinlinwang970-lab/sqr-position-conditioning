from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .checkpoint import load_training_checkpoint
from .config import load_config, write_resolved_config
from .coverage import build_coverage_banks, validate_coverage_banks
from .resume import build_s144_resume_components
from .toy_eval import decompose_two_query_errors, evaluate_query_predictions, write_error_decomposition, write_query_errors
from .trainer import predict_toy_waveforms


def run_projection_evaluator(
    config_path: str | Path,
    run_dir: str | Path,
    out_dir: str | Path,
    checkpoint_update: int = 18000,
    device: str = "cuda",
) -> dict:
    config = load_config(config_path)
    root = Path(run_dir)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    banks = build_coverage_banks(config)
    checks = validate_coverage_banks(banks, config)
    if not checks["ok"]:
        raise ValueError("Coverage data validation failed: " + "; ".join(checks["errors"]))
    checkpoint_path = root / "checkpoints" / f"update_{int(checkpoint_update):06d}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    components = build_s144_resume_components(config, device=device, train_mixtures=banks["s144_train"])
    checkpoint = load_training_checkpoint(
        checkpoint_path,
        components["model"],
        optimizer=components["optimizer"],
        expected_config=config,
        map_location=device,
    )
    if int(checkpoint["completed_updates"]) != int(checkpoint_update):
        raise ValueError("Checkpoint completed_updates does not match requested update")
    model = components["model"]
    model.eval()
    before_hash = _state_hash(model)
    metric_rows = []
    decomposition_rows = []
    raw_query_files = []
    projected_query_files = []
    eval_sets = (("s144_train", banks["s144_train"]), ("d72", banks["d72"]))
    for eval_name, mixtures in eval_sets:
        predictions = predict_toy_waveforms(model, mixtures, config, device=device)
        projected = project_same_mixture_predictions(mixtures, predictions)
        raw_summary = evaluate_query_predictions(mixtures, predictions)
        projected_summary = evaluate_query_predictions(mixtures, projected)
        raw_decomp = decompose_two_query_errors(mixtures, predictions)
        projected_decomp = decompose_two_query_errors(mixtures, projected)
        raw_query_path = out / f"raw_query_errors_{eval_name}_{checkpoint_update:06d}.csv"
        projected_query_path = out / f"projected_query_errors_{eval_name}_{checkpoint_update:06d}.csv"
        write_query_errors(raw_summary, raw_query_path)
        write_query_errors(projected_summary, projected_query_path)
        write_error_decomposition(raw_decomp, out / f"raw_error_decomposition_{eval_name}_{checkpoint_update:06d}.csv")
        write_error_decomposition(projected_decomp, out / f"projected_error_decomposition_{eval_name}_{checkpoint_update:06d}.csv")
        raw_query_files.append(str(raw_query_path))
        projected_query_files.append(str(projected_query_path))
        metric_rows.extend(_metric_rows(checkpoint_update, eval_name, "raw", raw_summary))
        metric_rows.extend(_metric_rows(checkpoint_update, eval_name, "projected", projected_summary))
        decomposition_rows.extend(_decomp_rows(checkpoint_update, eval_name, "raw", raw_decomp))
        decomposition_rows.extend(_decomp_rows(checkpoint_update, eval_name, "projected", projected_decomp))
    after_hash = _state_hash(model)
    if before_hash != after_hash:
        raise RuntimeError("Projection evaluation changed model parameters or buffers")
    _write_rows(metric_rows, out / "projection_metric_summary.csv")
    _write_rows(decomposition_rows, out / "projection_decomposition_summary.csv")
    write_resolved_config(config, out / "resolved_config.yaml")
    result = {
        "status": "PROJECTION_EVALUATION_COMPLETE_NO_PARAMETER_UPDATES",
        "parameter_updates": 0,
        "checkpoint_update": int(checkpoint_update),
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "coverage_checks": checks,
        "raw_query_files": raw_query_files,
        "projected_query_files": projected_query_files,
        "formal_score": "raw only",
        "projected_score": "diagnostic only; not a replacement for raw acceptance",
    }
    (out / "projection_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def project_same_mixture_predictions(mixtures, predictions):
    projected = dict(predictions)
    for mixture in mixtures:
        if len(mixture.queries) != 2:
            raise ValueError(f"{mixture.mixture_id}: expected exactly two legal queries")
        q1, q2 = mixture.queries
        h1 = np.asarray(predictions[q1.query_id])
        h2 = np.asarray(predictions[q2.query_id])
        x = np.asarray(mixture.mixture)
        if h1.shape != h2.shape or h1.shape != x.shape:
            raise ValueError(f"{mixture.mixture_id}: projection shape mismatch")
        correction = (x - h1 - h2) / 2.0
        projected[q1.query_id] = h1 + correction
        projected[q2.query_id] = h2 + correction
    return projected


def _metric_rows(update, eval_name, variant, summary):
    rows = []
    for key, values in summary["aggregate_by_task_split"].items():
        task, split = key.split("_", 1)
        rows.append({"update": int(update), "eval_set": eval_name, "variant": variant, "task": task, "split": split, **values})
    return rows


def _decomp_rows(update, eval_name, variant, decomposition):
    rows = []
    for key, values in decomposition["aggregate_by_task_split"].items():
        task, split = key.split("_", 1)
        rows.append({"update": int(update), "eval_set": eval_name, "variant": variant, "task": task, "split": split, **values})
    return rows


def _write_rows(rows, path):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _state_hash(model):
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()
