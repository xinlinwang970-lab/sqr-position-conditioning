from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
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
from .score_coverage import generate_p2b_corpus, manifest_hash, validate_p2b_corpus, write_p2b_manifests, write_rows
from .score_route import _make_optimizer
from .score_toy import BalancedScoreBatchSampler, ScoreQueryDataset, collate_score_batch


VARIANTS = ("correct", "swapped", "time_shifted", "wrong_pitch", "zero")


def run_p2b(config: dict, out_dir: str | Path, device: str = "cpu", attempted_updates: int | None = None) -> dict:
    import torch

    coverage = config["score_coverage"]
    if not coverage.get("enabled", False) or not config.get("score", {}).get("enabled", False):
        raise ValueError("score_coverage.enabled and score.enabled must be true")
    max_updates = int(coverage["max_updates"])
    updates = int(attempted_updates if attempted_updates is not None else max_updates)
    if updates < 1 or updates > max_updates:
        raise ValueError("P2B updates must be within score_coverage.max_updates")
    if config["training"]["precision"] != "fp32" or int(config["training"]["gradient_accumulation_steps"]) != 1:
        raise ValueError("P2B requires FP32 and gradient_accumulation_steps=1")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, out / "resolved_config.yaml")
    splits = generate_p2b_corpus(config)
    checks = validate_p2b_corpus(splits, config)
    (out / "data_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not checks["ok"]:
        raise ValueError("P2B data validation failed: " + "; ".join(checks["errors"]))
    write_p2b_manifests(splits, out / "manifests")
    (out / "manifest_hash.txt").write_text(manifest_hash(splits) + "\n", encoding="utf-8")
    _write_json(out / "environment.json", _runtime_info(device))
    _write_json(out / "exposure_plan.json", _exposure_plan(config, splits))

    train = splits["train"]
    dataset = ScoreQueryDataset(train, AudioConfig.from_dict(config["audio"]))
    sampler_d0 = BalancedScoreBatchSampler(dataset, batch_size=int(config["training"]["batch_size"]))
    sampler_d1 = BalancedScoreBatchSampler(dataset, batch_size=int(config["training"]["batch_size"]))
    pair = build_paired_initialization(config, device=device)
    opt_d0 = _make_optimizer(pair["base_model"], config)
    opt_d1 = _make_optimizer(pair["score_model"], config)
    _write_json(out / "paired_initialization_evidence.json", _paired_evidence(pair, dataset, sampler_d0, sampler_d1, config, device))

    curves: list[dict] = []
    trajectory: list[dict] = []
    checkpoint_updates = {int(value) for value in coverage["checkpoint_updates"]}
    started = perf_counter()
    for update in range(1, updates + 1):
        idx0 = sampler_d0.next_batch_indices()
        idx1 = sampler_d1.next_batch_indices()
        if idx0 != idx1:
            raise RuntimeError(f"D0/D1 sampler trajectory diverged at update {update}")
        batch = collate_score_batch([dataset[index] for index in idx0])
        trajectory.append({"update": update, "query_ids": list(batch["query_id"])})
        row = {"completed_updates": update, "batch_query_ids": ",".join(batch["query_id"])}
        for label, model, optimizer, use_score in (("D0", pair["base_model"], opt_d0, False), ("D1", pair["score_model"], opt_d1, True)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            spec = batch["spec"].to(device)
            target = batch["target"].to(device)
            if use_score:
                pred = model(spec, batch["timbre_id"].to(device), batch["position_id"].to(device), score_roll=batch["score_roll"].to(device))
            else:
                pred = model(spec, batch["timbre_id"].to(device), batch["position_id"].to(device))
            loss = packed_complex_mse(pred, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {label} loss at update {update}")
            loss.backward()
            optimizer.step()
            row[f"{label.lower()}_loss"] = float(loss.detach().cpu())
        curves.append(row)
        if update in checkpoint_updates:
            for label, model, optimizer, sampler, model_config in (("D0", pair["base_model"], opt_d0, sampler_d0, pair["base_config"]), ("D1", pair["score_model"], opt_d1, sampler_d1, pair["score_config"])):
                save_training_checkpoint(out / "checkpoints" / f"{label.lower()}_update_{update:06d}.pt", model, optimizer, model_config, update, capture_rng_state(), sampler.state_dict(), metadata={"stage": "paper1_p2b_score_coverage", "model": label, "manifest_hash": manifest_hash(splits)})
    write_rows(curves, out / "train_curves.csv")
    write_rows(sampler_d0.access_count_rows(), out / "sampler_access_counts_d0.csv")
    write_rows(sampler_d1.access_count_rows(), out / "sampler_access_counts_d1.csv")
    (out / "sampler_trajectory_hash.txt").write_text(_hash_json(trajectory) + "\n", encoding="utf-8")
    if updates == max_updates and all((out / "checkpoints" / f"{label.lower()}_update_{updates:06d}.pt").exists() for label in ("D0", "D1")):
        _evaluate_checkpoints(config, out, splits, device)
    result = {"status": "P2B_SMOKE_COMPLETE" if updates < max_updates else "P2B_FORMAL_TRAIN_AND_EVAL_COMPLETE", "attempted_updates": updates, "completed_updates": updates, "train_seconds": round(perf_counter() - started, 4), "manifest_hash": manifest_hash(splits), "data_checks": checks, "checkpoint_updates_written": sorted(checkpoint_updates & set(range(1, updates + 1)))}
    _write_json(out / "p2b_run_summary.json", result)
    return result


def _evaluate_checkpoints(config, out, splits, device):
    import torch
    from .models.separator import Separator

    eval_rows, detail_rows, decomp_rows, intervention_rows, shift_rows = [], [], [], [], []
    pair_rows = []
    checkpoints = sorted((out / "checkpoints").glob("*_update_*.pt"))
    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    for path in checkpoints:
        label = "D1" if path.name.startswith("d1_") else "D0"
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = Separator(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        step = int(checkpoint["completed_updates"])
        for split, mixtures in splits.items():
            for variant in VARIANTS:
                predictions = _predict_coverage(model, mixtures, transform, device, label == "D1", variant)
                rows = _score_rows(mixtures, predictions, variant)
                detail_rows.extend({"model": label, "step": step, "split": split, **row} for row in rows)
                eval_rows.append(_aggregate_eval_row(label, step, split, variant, rows))
                if variant == "correct":
                    decomp_rows.extend(_decompose_rows(label, step, split, mixtures, predictions))
            if label == "D1":
                for cells in (0, -1, 1, -2, 2, -4, 4):
                    predictions = _predict_coverage(model, mixtures, transform, device, True, "score_shift", shift_cells=cells)
                    rows = _score_rows(mixtures, predictions, f"cell_{cells:+d}")
                    shift_rows.append(_aggregate_eval_row(label, step, split, f"cell_{cells:+d}", rows) | {"bottleneck_cells": cells, "input_frames": cells * 8, "seconds": cells * 8 * int(config["audio"]["hop_length"]) / int(config["audio"]["sample_rate"])})
        model.cpu()
    write_rows(eval_rows, out / "checkpoint_eval_all.csv")
    write_rows(detail_rows, out / "query_mixture_detail.csv")
    write_rows(decomp_rows, out / "common_query_decomposition.csv")
    write_rows(shift_rows, out / "time_shift_sweep.csv")
    for row in eval_rows:
        if row["variant"] in ("correct", "swapped", "wrong_pitch", "zero"):
            pair_rows.append(row)
    write_rows(pair_rows, out / "intervention_summary.csv")
    _write_json(out / "pattern_pair_summary.json", _pattern_summary(detail_rows))
    _write_json(out / "p2b_evaluation_info.json", {"score_shift_rule": "bottleneck cell = 8 input frames; shift_piano_roll zero-pads and uses half-open frame indexing", "checkpoint_count": len(checkpoints), "splits": {key: len(value) for key, value in splits.items()}})


def _predict_coverage(model, mixtures, transform, device, use_score, variant, shift_cells=0):
    import torch
    predictions = {}
    with torch.inference_mode():
        for mixture in mixtures:
            wave = torch.from_numpy(mixture.mixture).unsqueeze(0).to(device)
            spec = transform.stft(wave)
            for query in mixture.queries:
                score = query.score
                if variant == "swapped":
                    score = swap_piano_rolls(mixture.queries[0].score, mixture.queries[1].score)[query.target_source_index]
                elif variant == "time_shifted":
                    score = shift_piano_roll(score, 2)
                elif variant == "score_shift":
                    score = shift_piano_roll(score, shift_cells * 8)
                elif variant == "wrong_pitch":
                    score = pitch_shift_piano_roll(score, 5)
                elif variant == "zero":
                    score = zero_piano_roll_like(score)
                args = (spec, torch.tensor([query.timbre_id], device=device), torch.tensor([query.position_id], device=device))
                output = model(*args, score_roll=torch.from_numpy(score).unsqueeze(0).to(device)) if use_score else model(*args)
                predictions[query.query_id] = transform.istft(output, length=mixture.mixture.shape[-1])[0].detach().cpu().numpy().astype(np.float32)
    return predictions


def _score_rows(mixtures, predictions, variant):
    rows = []
    for mixture in mixtures:
        query_predictions = [predictions[query.query_id].astype(np.float64) for query in mixture.queries]
        target_difference = mixture.sources[0].astype(np.float64) - mixture.sources[1].astype(np.float64)
        predicted_difference = query_predictions[0] - query_predictions[1]
        target_difference_norm = float(np.linalg.norm(target_difference))
        predicted_difference_norm = float(np.linalg.norm(predicted_difference))
        mixture_amplitude_ratio = predicted_difference_norm / (target_difference_norm + 1e-12)
        mixture_direction_cosine = float(np.sum(predicted_difference * target_difference) / (predicted_difference_norm * target_difference_norm + 1e-12))
        for query in mixture.queries:
            prediction = predictions[query.query_id].astype(np.float64)
            target = mixture.target_for_query(query).astype(np.float64)
            errors = [float(np.sum((prediction - source.astype(np.float64)) ** 2)) for source in mixture.sources]
            denom = float(np.sum(target * target) + 1e-12)
            blind = mixture.mixture.astype(np.float64) / 2.0
            blind_error = float(np.sum((blind - target) ** 2) / denom)
            zero_error = float(np.sum(target * target) / denom)
            minimum = min(errors)
            ties = [i for i, value in enumerate(errors) if abs(value - minimum) <= 1e-8 + 1e-6 * max(1.0, abs(minimum))]
            nearest = int(np.argmin(errors))
            rows.append({"mixture_id": mixture.mixture_id, "query_id": query.query_id, "variant": variant, "pattern_family": f"pair_{min(mixture.pattern_pair)}_{max(mixture.pattern_pair)}", "target_source_index": query.target_source_index, "r_model": errors[query.target_source_index] / denom, "r_blind": blind_error, "r_zero": zero_error, "nearest_accuracy": int(nearest == query.target_source_index), "unique_accuracy": int(nearest == query.target_source_index and len(ties) == 1), "tie": int(len(ties) > 1), "target_energy": denom, "output_norm": float(np.linalg.norm(prediction)), "target_norm": float(np.linalg.norm(target)), "all_source_errors": json.dumps(errors), "nearest_source_index": nearest, "tie_count": len(ties), "direction_cosine": mixture_direction_cosine, "amplitude_ratio": mixture_amplitude_ratio})
    return rows


def _aggregate_eval_row(model, step, split, variant, rows):
    if not rows:
        return {"model": model, "step": step, "split": split, "variant": variant, "query_count": 0}
    query = np.asarray([row["r_model"] for row in rows], dtype=float)
    mixture_values = {}
    for row in rows:
        mixture_values.setdefault(row["mixture_id"], []).append(row["r_model"])
    mix = np.asarray([np.mean(value) for value in mixture_values.values()], dtype=float)
    return {"model": model, "step": step, "split": split, "variant": variant, "query_count": len(rows), "mixture_count": len(mix), "query_mean_R": float(np.mean(query)), "query_median_R": float(np.median(query)), "mixture_mean_R": float(np.mean(mix)), "mixture_median_R": float(np.median(mix)), "mixture_p90_R": float(np.percentile(mix, 90)), "blind_R": float(np.mean([row["r_blind"] for row in rows])), "nearest_accuracy": float(np.mean([row["nearest_accuracy"] for row in rows])), "unique_accuracy": float(np.mean([row["unique_accuracy"] for row in rows])), "tie_ratio": float(np.mean([row["tie"] for row in rows])), "mean_direction_cosine": float(np.mean([row["direction_cosine"] for row in rows])), "median_direction_cosine": float(np.median([row["direction_cosine"] for row in rows])), "mean_amplitude_ratio": float(np.mean([row["amplitude_ratio"] for row in rows])), "median_amplitude_ratio": float(np.median([row["amplitude_ratio"] for row in rows]))}


def _decompose_rows(model, step, split, mixtures, predictions):
    rows = []
    for mixture in mixtures:
        h1, h2 = (predictions[q.query_id].astype(np.float64) for q in mixture.queries)
        s1, s2 = (mixture.sources[i].astype(np.float64) for i in (0, 1))
        denom = float(np.sum(s1 * s1) + np.sum(s2 * s2) + 1e-12)
        mh, dh, m, d = (h1 + h2) / 2, (h1 - h2) / 2, (s1 + s2) / 2, (s1 - s2) / 2
        raw = (np.sum((h1 - s1) ** 2) + np.sum((h2 - s2) ** 2)) / denom
        common = 2 * np.sum((mh - m) ** 2) / denom
        query = 2 * np.sum((dh - d) ** 2) / denom
        pdiff, tdiff = h1 - h2, s1 - s2
        pn, tn = np.linalg.norm(pdiff), np.linalg.norm(tdiff)
        rows.append({"model": model, "step": step, "split": split, "mixture_id": mixture.mixture_id, "pattern_family": f"pair_{min(mixture.pattern_pair)}_{max(mixture.pattern_pair)}", "R_raw": float(raw), "R_common": float(common), "R_query": float(query), "identity_error": float(raw - common - query), "predicted_difference_norm": float(pn), "true_difference_norm": float(tn), "amplitude_ratio": float(pn / (tn + 1e-12)), "direction_cosine": float(np.sum(pdiff * tdiff) / (pn * tn + 1e-12))})
    return rows


def _pattern_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["split"], row["pattern_family"], row["variant"]), []).append(row["r_model"])
    return [{"split": split, "pattern_family": family, "variant": variant, "query_count": len(values), "mean_R": float(np.mean(values)), "median_R": float(np.median(values)), "p90_R": float(np.percentile(values, 90))} for (split, family, variant), values in sorted(grouped.items())]


def _exposure_plan(config: dict, splits: dict) -> dict:
    old = Path(__file__).resolve().parents[2] / "artifacts" / "paper1" / "p2_score_route" / "full_2000" / "sampler_access_counts_d1.csv"
    visits = []
    if old.exists():
        with old.open(newline="", encoding="utf-8") as handle:
            visits = [int(row["visits"]) for row in csv.DictReader(handle)]
    unique = sorted(set(visits))
    if not visits:
        return {"status": "NOT_AVAILABLE", "source": str(old), "reason": "historical P2 sampler counts not found"}
    return {"status": "OK" if len(unique) == 1 else "NON_UNIFORM_REQUIRES_REVIEW", "source": str(old), "historical_unique_visits": unique, "historical_min": min(visits), "historical_max": max(visits), "p2b_train_queries": len(splits["train"]) * 2, "p2b_target_updates": int(np.ceil(max(visits) * len(splits["train"]) * 2 / int(config["training"]["batch_size"]))) }


def _paired_evidence(pair, dataset, sampler_d0, sampler_d1, config, device):
    import torch

    def state_hash(model, keys=None):
        keys = sorted(keys or model.state_dict())
        h = hashlib.sha256()
        for key in keys:
            tensor = model.state_dict()[key].detach().cpu().contiguous().numpy()
            h.update(key.encode()); h.update(tensor.tobytes())
        return h.hexdigest()

    first = collate_score_batch([dataset[i] for i in range(min(4, len(dataset)))])
    spec = first["spec"].to(device)
    timbre = first["timbre_id"].to(device)
    position = first["position_id"].to(device)
    score = first["score_roll"].to(device)
    with torch.inference_mode():
        a = pair["base_model"].eval()(spec, timbre, position)
        b = pair["score_model"].eval()(spec, timbre, position, score_roll=score)
    shared = sorted(pair["shared_keys"])
    score_only = sorted(pair["score_only_keys"])
    return {"status": "PAIRED_INITIALIZATION_USED", "device": device, "base_initial_state_hash": state_hash(pair["base_model"]), "score_initial_state_hash": state_hash(pair["score_model"]), "shared_parameter_hashes_equal": all(torch.equal(pair["base_model"].state_dict()[k], pair["score_model"].state_dict()[k]) for k in shared), "shared_key_count": len(shared), "shared_keys_hash": state_hash(pair["base_model"], shared), "score_only_keys": score_only, "score_projection_zero": bool(pair["score_model"].score_encoder.projection_is_zero()), "step0_max_output_difference": float((a - b).abs().max().cpu()), "step0_mean_output_difference": float((a - b).abs().mean().cpu()), "optimizer": {"type": "Adam", "learning_rate": config["training"]["learning_rate"], "betas": config["training"]["adam_betas"], "eps": config["training"]["adam_eps"], "weight_decay": config["training"]["weight_decay"]}, "first_batch_query_ids": first["query_id"], "sampler_initial_state": {"d0": sampler_d0.state_dict(), "d1": sampler_d1.state_dict()}}


def _runtime_info(device):
    import torch
    return {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda, "device_requested": device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
