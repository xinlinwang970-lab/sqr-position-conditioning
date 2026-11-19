from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import capture_rng_state, load_training_checkpoint, restore_rng_state, save_training_checkpoint
from .config import write_resolved_config
from .coverage import build_coverage_banks, validate_coverage_banks
from .resume import build_s144_resume_components, _evaluate_without_training_state_change
from .toy_eval import decompose_two_query_errors, evaluate_query_predictions
from .trainer import predict_toy_waveforms, train_toy_updates


def run_common_loss_ablation(
    parent_checkpoint: str | Path,
    out_dir: str | Path,
    device: str = "cuda",
    attempted_updates: int = 2000,
    milestones: tuple[int, ...] = (0, 500, 1000, 1500, 2000),
) -> dict:
    import torch

    parent = Path(parent_checkpoint)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(parent, map_location=device, weights_only=False)
    config = payload["config"]
    if int(payload["completed_updates"]) != 18000:
        raise ValueError("Common-loss ablation requires the 18000 parent checkpoint")
    banks = build_coverage_banks(config)
    checks = validate_coverage_banks(banks, config)
    if not checks["ok"]:
        raise ValueError("Coverage data validation failed: " + "; ".join(checks["errors"]))
    policy = {
        "status": "ROUND2_COMMON_LOSS_ABLATION",
        "parent_checkpoint": str(parent),
        "parent_completed_updates": 18000,
        "attempted_updates_per_branch": int(attempted_updates),
        "milestones": list(milestones),
        "branches": {"B0": {"common_loss_lambda": 0.0}, "B1": {"common_loss_lambda": 1.0}},
        "fixed": ["model", "loss base", "normalization", "data", "sampling", "learning_rate", "batch", "fp32"],
        "formal_acceptance_score": "raw",
        "projected_score": "diagnostic_only",
        "strict_parent_config": True,
    }
    (out / "runtime_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    write_resolved_config(config, out / "resolved_config.yaml")
    branch_results = {}
    for branch, common_lambda in (("B0", 0.0), ("B1", 1.0)):
        branch_results[branch] = _run_branch(
            branch,
            common_lambda,
            parent,
            config,
            banks,
            out / branch,
            device,
            attempted_updates,
            milestones,
        )
    result = {
        "status": "ROUND2_COMMON_LOSS_ABLATION_COMPLETE",
        "parent_checkpoint": str(parent),
        "device": device,
        "attempted_updates_per_branch": int(attempted_updates),
        "branches": branch_results,
        "coverage_checks": checks,
    }
    (out / "ablation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _run_branch(branch, common_lambda, parent, config, banks, out, device, attempted_updates, milestones):
    out.mkdir(parents=True, exist_ok=True)
    components = build_s144_resume_components(config, device=device, train_mixtures=banks["s144_train"])
    checkpoint = load_training_checkpoint(
        parent,
        components["model"],
        optimizer=components["optimizer"],
        expected_config=config,
        map_location=device,
    )
    restore_rng_state(checkpoint["rng_state"])
    components["sampler"].load_state_dict(checkpoint["sampler_state"])
    components["completed_updates"] = int(checkpoint["completed_updates"])
    initial_hash = _state_hash(components["model"])
    eval_sets = {"train": banks["s144_train"], "d72": banks["d72"]}
    curves = []
    metrics = []
    access_rows = []
    previous = 18000
    for milestone in milestones:
        if milestone == 0:
            continue
        target = 18000 + int(milestone)
        if target > 18000 + int(attempted_updates):
            raise ValueError(f"{branch}: milestone exceeds attempted update budget")
        chunk = target - int(components["completed_updates"])
        curves.extend(
            train_toy_updates(
                components,
                chunk,
                device=device,
                common_loss_lambda=common_lambda,
            )
        )
        previous = target
        checkpoint_path = _save_branch_checkpoint(out, components, config, branch, common_lambda)
        evaluation = _evaluate_without_training_state_change(
            components,
            config,
            eval_sets,
            device,
            out_dir=out,
            completed_updates=target,
            checkpoint_path=str(checkpoint_path),
        )
        for eval_name, summary in evaluation["summaries"].items():
            for key, values in summary["aggregate_by_task_split"].items():
                task, split = key.split("_", 1)
                metrics.append({"branch": branch, "lambda": common_lambda, "update": target, "eval_set": eval_name, "task": task, "split": split, **values})
        access_rows.extend({"branch": branch, "lambda": common_lambda, "update": target, **row} for row in components["sampler"].access_count_rows())
    _write_rows(curves, out / "train_curves.csv")
    _write_rows(metrics, out / "metric_summary.csv")
    _write_rows(access_rows, out / "sampler_access_counts.csv")
    (out / "runtime_policy.json").write_text(json.dumps({"branch": branch, "common_loss_lambda": common_lambda, "parent_checkpoint": str(parent), "base_config_unchanged": True}, indent=2), encoding="utf-8")
    return {
        "branch": branch,
        "common_loss_lambda": common_lambda,
        "completed_updates": int(components["completed_updates"]),
        "checkpoint_paths": sorted(str(path) for path in (out / "checkpoints").glob("*.pt")),
        "initial_state_hash": initial_hash,
        "final_state_hash": _state_hash(components["model"]),
        "sampler_access_rows": len(access_rows),
    }


def _save_branch_checkpoint(out, components, config, branch, common_lambda):
    completed = int(components["completed_updates"])
    path = out / "checkpoints" / f"update_{completed:06d}.pt"
    save_training_checkpoint(
        path,
        components["model"],
        components["optimizer"],
        config,
        completed_updates=completed,
        rng_state=capture_rng_state(),
        sampler_state=components["sampler"].state_dict(),
        metadata={"stage": "round2_common_loss_ablation", "branch": branch, "common_loss_lambda": common_lambda, "completed_updates": completed},
    )
    return path


def _state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
