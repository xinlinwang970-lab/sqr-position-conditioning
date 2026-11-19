from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from .audio import AudioConfig, AudioTransform
from .checkpoint import capture_rng_state, save_training_checkpoint
from .config import write_resolved_config
from .losses import packed_complex_mse
from .paired import build_paired_initialization
from .score import pitch_shift_piano_roll, shift_piano_roll, swap_piano_rolls, zero_piano_roll_like
from .score_toy import (
    BalancedScoreBatchSampler,
    ScoreQueryDataset,
    collate_score_batch,
    generate_score_corpus,
    validate_score_corpus,
    write_score_manifest,
)


VARIANTS = ("correct", "swapped", "time_shifted", "wrong_pitch", "zero")


def run_score_route(config: dict, out_dir: str | Path, device: str = "cpu", attempted_updates: int | None = None) -> dict:
    import torch

    if not (config.get("score_s") or {}).get("enabled", False):
        raise ValueError("score_s.enabled=true is required")
    if not (config.get("score") or {}).get("enabled", False):
        raise ValueError("score.enabled=true is required for the D1 side of P2")
    updates = int(attempted_updates or config["training"]["max_attempted_updates"])
    if updates < 1 or updates > int(config["training"]["max_attempted_updates"]):
        raise ValueError("P2 updates must be within the configured bounded budget")
    if str(config["training"]["precision"]) != "fp32":
        raise ValueError("P2 bounded Score-S trainer currently requires FP32")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, out / "resolved_config.yaml")
    runtime = _runtime_info(device)
    (out / "environment.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    random.seed(int(config["experiment"]["seed"]))
    np.random.seed(int(config["experiment"]["seed"]))
    torch.manual_seed(int(config["experiment"]["seed"]))

    mixtures = generate_score_corpus(config)
    data_checks = validate_score_corpus(mixtures, config)
    if not data_checks["ok"]:
        raise ValueError("Score-S data validation failed: " + "; ".join(data_checks["errors"]))
    write_score_manifest(mixtures, out / "manifests" / "score_s_manifest.jsonl")
    train_mixtures = [mixture for mixture in mixtures if mixture.split == "train"]
    probe_mixtures = [mixture for mixture in mixtures if mixture.split == "probe"]
    audio_config = AudioConfig.from_dict(config["audio"])
    train_dataset = ScoreQueryDataset(train_mixtures, audio_config)
    sampler_d0 = BalancedScoreBatchSampler(train_dataset, batch_size=int(config["training"]["batch_size"]))
    sampler_d1 = BalancedScoreBatchSampler(train_dataset, batch_size=int(config["training"]["batch_size"]))
    pair = build_paired_initialization(config, device=device)
    model_d0 = pair["base_model"]
    model_d1 = pair["score_model"]
    optimizer_d0 = _make_optimizer(model_d0, config)
    optimizer_d1 = _make_optimizer(model_d1, config)
    curves: list[dict] = []
    checkpoint_paths: list[str] = []
    checkpoint_updates = {100, 500, 1000, 2000}
    started = perf_counter()
    for update in range(1, updates + 1):
        indices_d0 = sampler_d0.next_batch_indices()
        indices_d1 = sampler_d1.next_batch_indices()
        if indices_d0 != indices_d1:
            raise RuntimeError(f"D0/D1 sampler trajectory diverged at update {update}")
        batch = collate_score_batch([train_dataset[index] for index in indices_d0])
        row = {"completed_updates": update, "batch_mixture_ids": ",".join(batch["mixture_id"]), "batch_query_ids": ",".join(batch["query_id"])}
        for label, model, optimizer, use_score in (("D0", model_d0, optimizer_d0, False), ("D1", model_d1, optimizer_d1, True)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            spec = batch["spec"].to(device)
            target = batch["target"].to(device)
            prediction = (
                model(spec, batch["timbre_id"].to(device), batch["position_id"].to(device), score_roll=batch["score_roll"].to(device))
                if use_score
                else model(spec, batch["timbre_id"].to(device), batch["position_id"].to(device))
            )
            loss = packed_complex_mse(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {label} loss at update {update}")
            loss.backward()
            optimizer.step()
            row[f"{label.lower()}_loss"] = float(loss.detach().cpu())
        curves.append(row)
        if update in checkpoint_updates:
            checkpoint_paths.extend(_save_pair_checkpoints(out, update, pair, optimizer_d0, optimizer_d1, sampler_d0, sampler_d1, config))

    validation_rows = []
    for label, model, use_score in (("D0", model_d0, False), ("D1", model_d1, True)):
        for split_name, split_mixtures in (("train", train_mixtures), ("probe", probe_mixtures)):
            summary = evaluate_score_route_model(model, split_mixtures, config, device=device, use_score=use_score, variant="correct")
            row = {"completed_updates": updates, "model": label, "split": split_name}
            row.update(summary["aggregate"])
            validation_rows.append(row)

    intervention_rows = []
    for label, model, use_score in (("D0", model_d0, False), ("D1", model_d1, True)):
        summary = evaluate_score_route_interventions(model, probe_mixtures, config, device=device, use_score=use_score)
        for row in summary["rows"]:
            row["model"] = label
            row["completed_updates"] = updates
            intervention_rows.append(row)
    _write_rows(curves, out / "train_curves.csv")
    _write_rows(validation_rows, out / "validation_curves.csv")
    _write_rows(intervention_rows, out / "score_intervention_detail.csv")
    _write_rows(sampler_d0.access_count_rows(), out / "sampler_access_counts_d0.csv")
    _write_rows(sampler_d1.access_count_rows(), out / "sampler_access_counts_d1.csv")
    summary = {
        "status": "P2_SCORE_ROUTE_SHORT_RUN_COMPLETE_NOT_BEHAVIOR_PASS" if updates < 2000 else "P2_SCORE_ROUTE_COMPLETE_REQUIRES_GATE_REVIEW",
        "device": device,
        "attempted_updates": updates,
        "completed_updates": updates,
        "train_seconds": round(perf_counter() - started, 4),
        "runtime": runtime,
        "data_checks": data_checks,
        "paired_shared_key_count": len(pair["shared_keys"]),
        "paired_score_only_key_count": len(pair["score_only_keys"]),
        "checkpoint_paths": checkpoint_paths,
        "validation_rows": validation_rows,
        "intervention_summary": _aggregate_interventions(intervention_rows),
    }
    (out / "score_route_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_score_route_model(model, mixtures, config, device="cpu", use_score=True, variant="correct") -> dict:
    predictions = _predict_variant(model, mixtures, config, device, use_score, variant)
    rows = []
    for mixture in mixtures:
        for query in mixture.queries:
            prediction = predictions[query.query_id]
            target = mixture.target_for_query(query)
            errors = [float(np.sum((prediction.astype(np.float64) - source.astype(np.float64)) ** 2)) for source in mixture.sources]
            target_energy = float(np.sum(target.astype(np.float64) ** 2) + 1e-12)
            blind = mixture.mixture / 2.0
            blind_error = float(np.sum((blind.astype(np.float64) - target.astype(np.float64)) ** 2))
            nearest = int(np.argmin(errors))
            minimum = min(errors)
            tie = sum(abs(value - minimum) <= 1e-8 + 1e-6 * max(1.0, abs(minimum)) for value in errors) > 1
            rows.append({"mixture_id": mixture.mixture_id, "query_id": query.query_id, "r_model": errors[query.target_source_index] / target_energy, "r_blind": blind_error / target_energy, "nearest_source_correct": nearest == query.target_source_index, "unique_selected_correct": nearest == query.target_source_index and not tie, "tie": tie})
    return {"rows": rows, "aggregate": _aggregate_score_rows(rows)}


def evaluate_score_route_interventions(model, mixtures, config, device="cpu", use_score=True) -> dict:
    rows = []
    for variant in VARIANTS:
        predictions = _predict_variant(model, mixtures, config, device, use_score, variant)
        for mixture in mixtures:
            query_predictions = [predictions[query.query_id] for query in mixture.queries]
            target_difference = mixture.sources[0].astype(np.float64) - mixture.sources[1].astype(np.float64)
            predicted_difference = query_predictions[0].astype(np.float64) - query_predictions[1].astype(np.float64)
            target_norm = float(np.linalg.norm(target_difference) + 1e-12)
            pred_norm = float(np.linalg.norm(predicted_difference))
            cosine = float(np.sum(predicted_difference * target_difference) / (np.linalg.norm(predicted_difference) * target_norm + 1e-12))
            for query, prediction in zip(mixture.queries, query_predictions):
                target = mixture.target_for_query(query).astype(np.float64)
                errors = [float(np.sum((prediction.astype(np.float64) - source.astype(np.float64)) ** 2)) for source in mixture.sources]
                error = float(errors[query.target_source_index] / (np.sum(target * target) + 1e-12))
                nearest = int(np.argmin(errors))
                minimum = min(errors)
                tie = sum(abs(value - minimum) <= 1e-8 + 1e-6 * max(1.0, abs(minimum)) for value in errors) > 1
                rows.append({"variant": variant, "mixture_id": mixture.mixture_id, "query_id": query.query_id, "target_source_index": query.target_source_index, "normalized_target_error": error, "nearest_source_correct": nearest == query.target_source_index, "unique_selected_correct": nearest == query.target_source_index and not tie, "tie": tie, "predicted_target_difference_amplitude_ratio": pred_norm / target_norm, "difference_direction_cosine": cosine, "output_difference_norm": pred_norm})
    return {"rows": rows}


def _predict_variant(model, mixtures, config, device, use_score, variant):
    import torch

    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    was_training = model.training
    model.eval()
    predictions = {}
    try:
        with torch.inference_mode():
            for mixture in mixtures:
                scores = [query.score for query in mixture.queries]
                if variant == "swapped":
                    scores = list(swap_piano_rolls(scores[0], scores[1]))
                elif variant == "time_shifted":
                    scores = [shift_piano_roll(score, 2) for score in scores]
                elif variant == "wrong_pitch":
                    scores = [pitch_shift_piano_roll(score, 5) for score in scores]
                elif variant == "zero":
                    scores = [zero_piano_roll_like(score) for score in scores]
                wave = torch.from_numpy(mixture.mixture).unsqueeze(0).to(device)
                spec = transform.stft(wave)
                for query, score in zip(mixture.queries, scores):
                    timbre = torch.tensor([query.timbre_id], dtype=torch.long, device=device)
                    position = torch.tensor([query.position_id], dtype=torch.long, device=device)
                    if use_score:
                        output = model(spec, timbre, position, score_roll=torch.from_numpy(score).unsqueeze(0).to(device))
                    else:
                        output = model(spec, timbre, position)
                    predicted = transform.istft(output, length=mixture.mixture.shape[-1])[0]
                    predictions[query.query_id] = predicted.detach().cpu().numpy().astype(np.float32)
    finally:
        model.train(was_training)
    return predictions


def _save_pair_checkpoints(out, update, pair, optimizer_d0, optimizer_d1, sampler_d0, sampler_d1, config):
    paths = []
    for label, model, optimizer, sampler, model_config in (("D0", pair["base_model"], optimizer_d0, sampler_d0, pair["base_config"]), ("D1", pair["score_model"], optimizer_d1, sampler_d1, pair["score_config"])):
        path = out / "checkpoints" / f"{label.lower()}_update_{update:06d}.pt"
        save_training_checkpoint(path, model, optimizer, model_config, update, capture_rng_state(), sampler.state_dict(), metadata={"stage": "paper1_p2_score_route", "model": label, "completed_updates": update})
        paths.append(str(path))
    return paths


def _make_optimizer(model, config):
    import torch

    training = config["training"]
    return torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]), betas=tuple(training["adam_betas"]), eps=float(training["adam_eps"]), weight_decay=float(training["weight_decay"]), foreach=bool(training["foreach"]))


def _aggregate_score_rows(rows):
    if not rows:
        return {"median_r_model": None, "median_r_blind": None, "nearest_accuracy": None, "unique_accuracy": None, "tie_ratio": None, "num_queries": 0}
    return {"median_r_model": float(np.median([row["r_model"] for row in rows])), "median_r_blind": float(np.median([row["r_blind"] for row in rows])), "nearest_accuracy": float(np.mean([row["nearest_source_correct"] for row in rows])), "unique_accuracy": float(np.mean([row["unique_selected_correct"] for row in rows])), "tie_ratio": float(np.mean([row["tie"] for row in rows])), "num_queries": len(rows)}


def _aggregate_interventions(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["variant"]), []).append(row)
    return {f"{model}_{variant}": {"mean_normalized_target_error": float(np.mean([item["normalized_target_error"] for item in values])), "mean_nearest_accuracy": float(np.mean([item["nearest_source_correct"] for item in values])), "mean_unique_accuracy": float(np.mean([item["unique_selected_correct"] for item in values])), "mean_tie_ratio": float(np.mean([item["tie"] for item in values])), "mean_output_difference_norm": float(np.mean([item["output_difference_norm"] for item in values])), "mean_difference_direction_cosine": float(np.mean([item["difference_direction_cosine"] for item in values]))} for (model, variant), values in sorted(grouped.items())}


def _write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _runtime_info(device):
    import sys
    import torch

    info = {"python_executable": sys.executable, "torch_version": torch.__version__, "torch_file": torch.__file__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "requested_device": device}
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info["cuda_device_name"] = prop.name
        info["cuda_device_memory_gib"] = round(prop.total_memory / (1024**3), 3)
    return info
