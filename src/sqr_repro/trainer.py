from __future__ import annotations

import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

from .losses import packed_complex_mse, packed_complex_mse_with_common_consistency


def run_smoke(config: dict, device: str = "cpu", updates: int = 2) -> dict:
    import torch

    from .models.separator import Separator

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested, but torch.cuda.is_available() is false")
    if updates < 1 or updates > 20:
        raise ValueError("M0-M2 smoke measured updates must be between 1 and 20")

    torch.manual_seed(int(config["experiment"]["seed"]))
    model = Separator(config).to(device)
    model.train()
    first_param = next(model.parameters()).detach().cpu().clone()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["adam_betas"]),
        eps=float(config["training"]["adam_eps"]),
        weight_decay=float(config["training"]["weight_decay"]),
        foreach=bool(config["training"]["foreach"]),
    )
    batch = int(config["training"]["batch_size"])
    frames = int(config["audio"]["num_frames"])
    freq = int(config["audio"]["n_fft"]) // 2 + 1
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    precision = str(config["training"].get("precision", "fp32"))
    use_amp = device == "cuda" and precision == "amp_float16"
    score_config = config.get("score") or {}
    score_enabled = bool(score_config.get("enabled", False))
    score_pitches = int(score_config.get("num_pitches", 128))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    warmup_updates = int(config.get("profiling", {}).get("warmup_updates", 0))

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for micro in range(accumulation):
            spec = torch.randn(batch, 4, frames, freq, device=device)
            target = torch.randn_like(spec)
            timbre = torch.tensor([(micro + i) % int(config["model"].get("num_timbres", 3)) for i in range(batch)], device=device)
            position = torch.tensor([(micro + i) % int(config["model"].get("num_positions", 4)) for i in range(batch)], device=device)
            score_roll = torch.randn(batch, score_pitches, frames, device=device) if score_enabled else None
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with context:
                prediction = model(spec, timbre.long(), position.long(), score_roll=score_roll)
                loss = packed_complex_mse(prediction, target) / accumulation
            scaler.scale(loss).backward()
            update_loss += float(loss.detach().cpu()) * accumulation
        scaler.step(optimizer)
        scaler.update()
        return update_loss / accumulation

    for _ in range(warmup_updates):
        train_step()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = perf_counter()
    losses = []
    for _ in range(updates):
        losses.append(train_step())
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = perf_counter() - started
    changed = not torch.allclose(first_param, next(model.parameters()).detach().cpu())
    result = {
        "device": device,
        "precision": "amp_float16" if use_amp else "fp32",
        "warmup_updates": warmup_updates,
        "attempted_updates": updates,
        "completed_updates": len(losses),
        "losses": losses,
        "losses_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "first_parameter_changed": bool(changed),
        "score_enabled": score_enabled,
        "score_input_shape": [batch, score_pitches, frames] if score_enabled else None,
        "seconds": round(seconds, 4),
    }
    if device == "cuda":
        result["cuda_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / (1024**2), 3)
        result["cuda_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / (1024**2), 3)
    return result


def build_toy_training_components(config: dict, device: str = "cpu", split: str = "train") -> dict:
    import numpy as np
    import torch

    from .audio import AudioConfig
    from .models.separator import Separator
    from .toy import BalancedToyBatchSampler, ToyQueryDataset, generate_toy_corpus, validate_toy_corpus

    _validate_bounded_training_config(config)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mixtures = generate_toy_corpus(config)
    checks = validate_toy_corpus(mixtures, config)
    if not checks["ok"]:
        raise ValueError("Toy data validation failed: " + "; ".join(checks["errors"]))
    selected = [mixture for mixture in mixtures if mixture.split == split and mixture.task in {"T", "P"}]
    dataset = ToyQueryDataset(selected, AudioConfig.from_dict(config["audio"]))
    sampler = BalancedToyBatchSampler(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        tasks=tuple(task for task in config["toy"]["tasks"] if task in {"T", "P"}),
    )
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
        "mixtures": mixtures,
        "checks": checks,
        "completed_updates": 0,
    }


def train_toy_updates(
    components: dict,
    updates: int,
    device: str = "cpu",
    after_update=None,
    common_loss_lambda: float = 0.0,
) -> list[dict]:
    import torch

    from .toy import collate_toy_batch

    model = components["model"]
    optimizer = components["optimizer"]
    dataset = components["dataset"]
    sampler = components["sampler"]
    model.train()
    curves = []
    for _ in range(updates):
        indices = sampler.next_batch_indices()
        batch = collate_toy_batch([dataset[index] for index in indices])
        spec = batch["spec"].to(device)
        target = batch["target"].to(device)
        timbre = batch["timbre_id"].to(device)
        position = batch["position_id"].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(spec, timbre, position)
        loss, common_loss = packed_complex_mse_with_common_consistency(
            prediction,
            target,
            batch["spec"].to(device),
            batch["mixture_id"],
            batch["query_id"],
            common_loss_lambda,
        )
        loss.backward()
        optimizer.step()
        components["completed_updates"] = int(components.get("completed_updates", 0)) + 1
        curves.append(
            {
                "completed_updates": components["completed_updates"],
                "loss": float(loss.detach().cpu()),
                "common_loss": float(common_loss.detach().cpu()),
                "common_loss_lambda": float(common_loss_lambda),
                "batch_tasks": ",".join(batch["task"]),
                "batch_mixture_ids": ",".join(batch["mixture_id"]),
                "batch_query_ids": ",".join(batch["query_id"]),
                "unique_timbres": int(torch.unique(timbre.detach().cpu()).numel()),
                "unique_positions": int(torch.unique(position.detach().cpu()).numel()),
            }
        )
        if after_update is not None:
            after_update(components, curves[-1])
    return curves


def predict_toy_waveforms(model, mixtures, config: dict, device: str = "cpu") -> dict:
    import torch

    from .audio import AudioConfig, AudioTransform
    from .toy import iter_query_records

    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    predictions = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for mixture, query in iter_query_records(mixtures):
                wave = torch.from_numpy(mixture.mixture).unsqueeze(0).to(device)
                spec = transform.stft(wave)
                timbre = torch.tensor([query.timbre_id], dtype=torch.long, device=device)
                position = torch.tensor([query.position_id], dtype=torch.long, device=device)
                predicted_spec = model(spec, timbre, position)
                predicted_wave = transform.istft(predicted_spec, length=mixture.mixture.shape[-1])
                predictions[query.query_id] = predicted_wave[0].detach().cpu().numpy().astype("float32")
    finally:
        model.train(was_training)
    return predictions


def run_bounded_toy_training(
    config: dict,
    out_dir: str | Path,
    device: str = "cpu",
    attempted_updates: int | None = None,
) -> dict:
    from .config import write_resolved_config
    from .toy import write_toy_manifest
    from .toy_eval import baseline_predictions, evaluate_query_predictions, write_query_errors

    attempted = int(attempted_updates or config["training"]["max_attempted_updates"])
    max_attempted = int(config["training"]["max_attempted_updates"])
    if attempted < 1 or attempted > max_attempted:
        raise ValueError(f"Round 2 toy training requires attempted updates in [1, {max_attempted}]")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, out / "resolved_config.yaml")
    runtime = _runtime_info(device)
    (out / "environment.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    components = build_toy_training_components(config, device=device, split="train")
    write_toy_manifest(components["mixtures"], out / "toy_manifest.jsonl")
    input_shape = _toy_input_shape(components, config)
    parameter_count = components["model"].count_parameters()
    train_mixtures = [mixture for mixture in components["mixtures"] if mixture.split == "train"]
    probe_mixtures = [mixture for mixture in components["mixtures"] if mixture.split == "probe"]
    validation_rows: list[dict] = []
    checkpoint_probe_rows: list[dict] = []
    checkpoint_paths: list[str] = []
    saved_checkpoint_updates: set[int] = set()
    validate_every = int(config["training"]["validate_every_updates"])
    checkpoint_every = int(config["training"]["checkpoint_every_updates"])
    required_checkpoint_updates = {attempted}
    if attempted >= 1000:
        required_checkpoint_updates.add(1000)

    def after_update(updated_components: dict, _row: dict) -> None:
        completed = int(updated_components["completed_updates"])
        should_checkpoint = (
            completed in required_checkpoint_updates
            or (checkpoint_every > 0 and completed % checkpoint_every == 0)
        )
        if should_checkpoint and completed not in saved_checkpoint_updates:
            checkpoint_path = _save_toy_checkpoint(updated_components, config, out, completed)
            checkpoint_paths.append(str(checkpoint_path))
            saved_checkpoint_updates.add(completed)
        if validate_every > 0 and (completed % validate_every == 0 or completed == attempted):
            train_summary = _evaluate_model_on_mixtures(
                updated_components["model"],
                train_mixtures,
                config,
                device=device,
            )
            validation_rows.extend(_summary_rows(train_summary, completed))
        if completed in required_checkpoint_updates:
            probe_summary = _evaluate_model_on_mixtures(
                updated_components["model"],
                probe_mixtures,
                config,
                device=device,
            )
            checkpoint_path = out / "checkpoints" / f"update_{completed:06d}.pt"
            checkpoint_probe_rows.extend(_summary_rows(probe_summary, completed, checkpoint_path=str(checkpoint_path)))

    memory = {}
    started = perf_counter()
    if device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    curves = train_toy_updates(components, attempted, device=device, after_update=after_update)
    if device == "cuda":
        torch.cuda.synchronize()
        memory = {
            "cuda_train_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
            "cuda_train_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 3),
        }
    train_seconds = perf_counter() - started
    _write_curves(curves, out / "train_curves.csv")
    _write_rows(validation_rows, out / "validation_curves.csv")
    _write_rows(checkpoint_probe_rows, out / "checkpoint_probe_summary.csv")
    _write_rows(components["sampler"].access_count_rows(), out / "sampler_access_counts.csv")
    model_predictions = predict_toy_waveforms(components["model"], train_mixtures + probe_mixtures, config, device=device)
    model_summary = evaluate_query_predictions(train_mixtures + probe_mixtures, model_predictions)
    blind_summary = evaluate_query_predictions(
        train_mixtures + probe_mixtures,
        baseline_predictions(train_mixtures + probe_mixtures, "blind"),
    )
    zero_summary = evaluate_query_predictions(
        train_mixtures + probe_mixtures,
        baseline_predictions(train_mixtures + probe_mixtures, "zero"),
    )
    write_query_errors(model_summary, out / "query_errors_model.csv")
    _write_group_summary(model_summary, out / "behavior_by_task_split_model.csv")
    result = {
        "status": _toy_run_status(attempted),
        "device": device,
        "precision": str(config["training"]["precision"]),
        "input_shape": input_shape,
        "parameter_count": parameter_count,
        "runtime": runtime,
        "attempted_updates": attempted,
        "completed_updates": int(components["completed_updates"]),
        "train_seconds": round(train_seconds, 4),
        **memory,
        "data_checks": components["checks"],
        "model_behavior_summary": model_summary["aggregate"],
        "model_behavior_by_task_split": model_summary["aggregate_by_task_split"],
        "blind_behavior_summary": blind_summary["aggregate"],
        "blind_behavior_by_task_split": blind_summary["aggregate_by_task_split"],
        "zero_behavior_summary": zero_summary["aggregate"],
        "zero_behavior_by_task_split": zero_summary["aggregate_by_task_split"],
        "checkpoint_paths": checkpoint_paths,
        "validation_curve_path": str(out / "validation_curves.csv"),
        "checkpoint_probe_summary_path": str(out / "checkpoint_probe_summary.csv"),
        "sampler_access_counts_path": str(out / "sampler_access_counts.csv"),
    }
    (out / "behavior_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _validate_bounded_training_config(config: dict) -> None:
    training = config["training"]
    if str(training["precision"]) != "fp32":
        raise ValueError("Round 2 bounded trainer only supports FP32")
    if int(training["gradient_accumulation_steps"]) != 1:
        raise ValueError("Round 2 bounded trainer requires gradient_accumulation_steps=1")
    if int(config["data"]["num_workers"]) != 0:
        raise ValueError("Round 2 bounded trainer requires num_workers=0")
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("Round 2 bounded trainer only supports Adam")
    if "toy" not in config or not config["toy"].get("enabled", False):
        raise ValueError("Round 2 bounded trainer requires toy.enabled=true")


def _toy_run_status(attempted_updates: int) -> str:
    if attempted_updates == 1000:
        return "ROUND2_TOY_JOB1_COMPLETE_NOT_PASS_TOY_BEHAVIOR"
    if attempted_updates > 1000:
        return "ROUND2_DIAGNOSTIC_COMPLETE_NOT_PASS_TOY_BEHAVIOR"
    return "ROUND2_SHORT_RUN_COMPLETE_NOT_TOY_BEHAVIOR"


def _write_curves(curves: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not curves:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)


def _write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _toy_input_shape(components: dict, config: dict) -> list[int]:
    sample = components["dataset"][0]["spec"]
    return [int(config["training"]["batch_size"]), *[int(dim) for dim in sample.shape]]


def _write_group_summary(summary: dict, path: Path) -> None:
    rows = []
    for key, values in summary["aggregate_by_task_split"].items():
        task, split = key.split("_", 1)
        row = {"task": task, "split": split}
        row.update(values)
        rows.append(row)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_model_on_mixtures(model, mixtures: list, config: dict, device: str = "cpu") -> dict:
    from .toy_eval import evaluate_query_predictions

    predictions = predict_toy_waveforms(model, mixtures, config, device=device)
    return evaluate_query_predictions(mixtures, predictions)


def _summary_rows(summary: dict, update: int, checkpoint_path: str | None = None) -> list[dict]:
    rows = []
    for key, values in summary["aggregate_by_task_split"].items():
        task, split = key.split("_", 1)
        row = {
            "completed_updates": int(update),
            "task": task,
            "split": split,
            "median_r_model": values["median_r_model"],
            "median_r_blind": values["median_r_blind"],
            "median_r_zero": values["median_r_zero"],
            "mean_nearest_source_accuracy": values["mean_nearest_source_accuracy"],
            "mean_unique_selection_accuracy": values["mean_unique_selection_accuracy"],
            "mean_tie_ratio": values["mean_tie_ratio"],
            "total_mixtures": values["total_mixtures"],
            "total_queries": values["total_queries"],
        }
        if checkpoint_path is not None:
            row["checkpoint_path"] = checkpoint_path
        rows.append(row)
    return rows


def _save_toy_checkpoint(components: dict, config: dict, out: Path, completed: int) -> Path:
    from .checkpoint import capture_rng_state, save_training_checkpoint

    checkpoint_path = out / "checkpoints" / f"update_{completed:06d}.pt"
    save_training_checkpoint(
        checkpoint_path,
        components["model"],
        components["optimizer"],
        config,
        completed_updates=completed,
        rng_state=capture_rng_state(),
        sampler_state=components["sampler"].state_dict(),
        metadata={
            "stage": "round2_bounded_toy",
            "completed_updates": completed,
        },
    )
    return checkpoint_path


def _runtime_info(device: str) -> dict:
    import sys

    try:
        import torch
    except ModuleNotFoundError as exc:
        return {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch_available": False,
            "torch_error": str(exc),
            "requested_device": device,
        }
    info = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_available": True,
        "torch_version": torch.__version__,
        "torch_file": getattr(torch, "__file__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "requested_device": device,
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info["cuda_device_name"] = prop.name
        info["cuda_device_memory_gib"] = round(prop.total_memory / (1024**3), 3)
    return info

