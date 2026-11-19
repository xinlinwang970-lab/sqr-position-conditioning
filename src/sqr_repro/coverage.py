from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from .audio import AudioConfig
from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .config import write_resolved_config
from .toy import (
    BalancedToyBatchSampler,
    ToyMixture,
    ToyQueryDataset,
    all_toy_scenes,
    generate_toy_corpus,
    generate_toy_mixture_for_scene,
    scene_key_from_spec,
    scene_label,
    stable_toy_seed,
    toy_scene_key,
    validate_toy_corpus,
    write_toy_manifest,
)
from .toy_eval import baseline_predictions, evaluate_query_predictions, write_query_errors
from .trainer import (
    _evaluate_model_on_mixtures,
    _runtime_info,
    _summary_rows,
    _toy_input_shape,
    _validate_bounded_training_config,
    _write_group_summary,
    _write_rows,
    train_toy_updates,
)


S16 = "s16"
S144 = "s144"
D72 = "d72"


def run_data_coverage_comparison(
    config: dict,
    out_dir: str | Path,
    device: str = "cpu",
    attempted_updates: int | None = None,
) -> dict:
    import torch

    _validate_bounded_training_config(config)
    attempted = int(attempted_updates or config["training"]["max_attempted_updates"])
    max_attempted = int(config["training"]["max_attempted_updates"])
    if attempted < 1 or attempted > max_attempted:
        raise ValueError(f"Coverage comparison requires attempted updates in [1, {max_attempted}]")
    if attempted > 2000:
        raise ValueError("Coverage comparison is capped at 2000 updates")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, out / "resolved_config.yaml")
    runtime = _runtime_info(device)
    (out / "environment.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    banks = build_coverage_banks(config)
    checks = validate_coverage_banks(banks, config)
    if not checks["ok"]:
        (out / "coverage_data_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        raise ValueError("Coverage data validation failed: " + "; ".join(checks["errors"]))
    (out / "coverage_data_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    write_coverage_artifacts(banks, out)

    initial_state, initial_hash = _initial_state(config, device=device)
    torch.save({"model_state_dict": initial_state, "sha256": initial_hash}, out / "initial_state.pt")

    run_results = {}
    for run_name, train_mixtures in ((S16, banks["s16_train"]), (S144, banks["s144_train"])):
        run_results[run_name] = _run_one_coverage_training(
            config=config,
            run_name=run_name,
            out_dir=out / run_name,
            device=device,
            attempted_updates=attempted,
            train_mixtures=train_mixtures,
            legacy_probe=banks["legacy_probe"],
            d72=banks["d72"],
            initial_state=initial_state,
            initial_hash=initial_hash,
        )

    result = {
        "status": "ROUND2_DATA_COVERAGE_COMPLETE_NOT_TOY_TP",
        "attempted_updates_per_run": attempted,
        "runs": run_results,
        "initial_state_sha256": initial_hash,
        "runtime": runtime,
        "data_checks": checks,
    }
    (out / "coverage_comparison_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_coverage_banks(config: dict) -> dict[str, list[ToyMixture]]:
    original = generate_toy_corpus(config, tasks=("T", "P"))
    original_train = [mixture for mixture in original if mixture.split == "train"]
    legacy_probe = [mixture for mixture in original if mixture.split == "probe"]
    s144_train = _build_s144_train(config, original_train)
    d72 = _build_d72(config)
    return {
        "original": original,
        "original_train": original_train,
        "legacy_probe": legacy_probe,
        "s16_train": list(original_train),
        "s144_train": s144_train,
        "d72": d72,
    }


def validate_coverage_banks(banks: dict[str, list[ToyMixture]], config: dict) -> dict:
    errors: list[str] = []
    original_by_id = {mixture.mixture_id: mixture.mixture_hash for mixture in banks["original_train"]}
    s144_by_id = {mixture.mixture_id: mixture.mixture_hash for mixture in banks["s144_train"]}
    for mixture_id, hash_value in original_by_id.items():
        if s144_by_id.get(mixture_id) != hash_value:
            errors.append(f"{mixture_id}: original train hash changed in S144")

    for bank_name in ("s16_train", "s144_train", "legacy_probe", "d72"):
        ids = [mixture.mixture_id for mixture in banks[bank_name]]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            errors.append(f"{bank_name}: duplicate mixture id(s): {', '.join(duplicates)}")

    combined_hashes: dict[str, set[str]] = {}
    for bank_name in ("s144_train", "legacy_probe", "d72"):
        for mixture in banks[bank_name]:
            combined_hashes.setdefault(mixture.mixture_hash, set()).add(bank_name)
    leaked = {hash_value: names for hash_value, names in combined_hashes.items() if len(names) > 1}
    if leaked:
        errors.append(f"mixture hash leakage across S144/probe/D72: {len(leaked)}")

    source_hashes: dict[str, set[str]] = {}
    for bank_name in ("s144_train", "legacy_probe", "d72"):
        for mixture in banks[bank_name]:
            for source_hash in mixture.manifest_record()["source_hashes"]:
                source_hashes.setdefault(source_hash, set()).add(bank_name)
    source_leakage = {hash_value: names for hash_value, names in source_hashes.items() if len(names) > 1}
    if source_leakage:
        errors.append(f"source hash leakage across S144/probe/D72: {len(source_leakage)}")

    for bank_name, mixtures in (
        ("s16_all", banks["s16_train"] + banks["legacy_probe"] + banks["d72"]),
        ("s144_all", banks["s144_train"] + banks["legacy_probe"] + banks["d72"]),
    ):
        check = validate_toy_corpus(mixtures, config)
        if not check["ok"]:
            errors.extend(f"{bank_name}: {error}" for error in check["errors"])

    _check_scene_counts(banks["s144_train"], task="T", expected_per_scene=12, label="S144")
    _check_scene_counts(banks["s144_train"], task="P", expected_per_scene=8, label="S144")
    _check_scene_counts(banks["d72"], task="T", expected_per_scene=6, label="D72")
    _check_scene_counts(banks["d72"], task="P", expected_per_scene=4, label="D72")
    for task, expected, label, mixtures in (
        ("T", 12, "S144", banks["s144_train"]),
        ("P", 8, "S144", banks["s144_train"]),
        ("T", 6, "D72", banks["d72"]),
        ("P", 4, "D72", banks["d72"]),
    ):
        counts = _scene_counter([mixture for mixture in mixtures if mixture.task == task])
        for scene in all_toy_scenes(task):
            key = scene_key_from_spec(task, scene)
            if counts[key] != expected:
                errors.append(f"{label} {key}: expected {expected}, got {counts[key]}")

    return {
        "ok": not errors,
        "errors": errors,
        "counts": {
            name: len(mixtures)
            for name, mixtures in banks.items()
            if isinstance(mixtures, list)
        },
        "original_train_hashes_preserved_in_s144": all(
            s144_by_id.get(mixture_id) == hash_value for mixture_id, hash_value in original_by_id.items()
        ),
    }


def write_coverage_artifacts(banks: dict[str, list[ToyMixture]], out: Path) -> None:
    manifest_dir = out / "manifests"
    for name in ("original_train", "legacy_probe", "s16_train", "s144_train", "d72"):
        write_toy_manifest(banks[name], manifest_dir / f"{name}.jsonl")
    _write_rows(_manifest_hash_rows(manifest_dir), out / "manifest_hashes.csv")
    _write_rows(_coverage_rows("original_train", banks["original_train"]), out / "coverage_original_train.csv")
    _write_rows(_coverage_rows("legacy_probe", banks["legacy_probe"]), out / "coverage_legacy_probe.csv")
    _write_rows(_coverage_rows("s144_train", banks["s144_train"]), out / "coverage_s144_train.csv")
    _write_rows(_coverage_rows("d72", banks["d72"]), out / "coverage_d72.csv")


def _build_s144_train(config: dict, original_train: list[ToyMixture]) -> list[ToyMixture]:
    mixtures = list(original_train)
    base_seed = int(config["experiment"]["seed"])
    global_index = 0
    for task, target_per_scene in (("T", 12), ("P", 8)):
        task_original = [mixture for mixture in original_train if mixture.task == task]
        counts = _scene_counter(task_original)
        for scene in all_toy_scenes(task):
            key = scene_key_from_spec(task, scene)
            if counts[key] > target_per_scene:
                raise ValueError(f"Original {key} already has {counts[key]} mixtures, above target {target_per_scene}")
            deficit = target_per_scene - counts[key]
            label = scene_label(task, scene)
            for local_index in range(deficit):
                seed = stable_toy_seed(base_seed, S144, "train", task, label, local_index)
                mixture_id = f"{S144}_train_{task}_{label}_{global_index:04d}"
                mixtures.append(generate_toy_mixture_for_scene(config, task, "train", mixture_id, seed, scene))
                global_index += 1
    return mixtures


def _build_d72(config: dict) -> list[ToyMixture]:
    mixtures = []
    base_seed = int(config["experiment"]["seed"])
    global_index = 0
    for task, per_scene in (("T", 6), ("P", 4)):
        for scene in all_toy_scenes(task):
            label = scene_label(task, scene)
            for local_index in range(per_scene):
                seed = stable_toy_seed(base_seed, D72, "validation", task, label, local_index)
                mixture_id = f"{D72}_{task}_{label}_{global_index:04d}"
                mixtures.append(generate_toy_mixture_for_scene(config, task, D72, mixture_id, seed, scene))
                global_index += 1
    return mixtures


def _run_one_coverage_training(
    config: dict,
    run_name: str,
    out_dir: Path,
    device: str,
    attempted_updates: int,
    train_mixtures: list[ToyMixture],
    legacy_probe: list[ToyMixture],
    d72: list[ToyMixture],
    initial_state: dict,
    initial_hash: str,
) -> dict:
    import torch

    from .models.separator import Separator

    out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, out_dir / "resolved_config.yaml")
    write_toy_manifest(train_mixtures, out_dir / "train_manifest.jsonl")
    write_toy_manifest(legacy_probe, out_dir / "legacy_probe_manifest.jsonl")
    write_toy_manifest(d72, out_dir / "d72_manifest.jsonl")

    random.seed(stable_toy_seed(config["experiment"]["seed"], run_name, "training-random"))
    np.random.seed(stable_toy_seed(config["experiment"]["seed"], run_name, "numpy") % (2**32))
    torch.manual_seed(stable_toy_seed(config["experiment"]["seed"], run_name, "torch"))

    model = Separator(config).to(device)
    model.load_state_dict({name: value.to(device) for name, value in initial_state.items()}, strict=True)
    loaded_initial_hash = _state_dict_sha256(model.state_dict())
    if loaded_initial_hash != initial_hash:
        raise ValueError(f"{run_name}: loaded initial state hash mismatch")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["adam_betas"]),
        eps=float(config["training"]["adam_eps"]),
        weight_decay=float(config["training"]["weight_decay"]),
        foreach=bool(config["training"]["foreach"]),
    )
    dataset = ToyQueryDataset(train_mixtures, AudioConfig.from_dict(config["audio"]))
    sampler = BalancedToyBatchSampler(dataset, batch_size=int(config["training"]["batch_size"]), tasks=("T", "P"))
    components = {
        "model": model,
        "optimizer": optimizer,
        "dataset": dataset,
        "sampler": sampler,
        "mixtures": train_mixtures + legacy_probe + d72,
        "checks": validate_toy_corpus(train_mixtures + legacy_probe + d72, config),
        "completed_updates": 0,
    }
    if not components["checks"]["ok"]:
        raise ValueError(f"{run_name}: toy data validation failed: " + "; ".join(components["checks"]["errors"]))

    validation_rows: list[dict] = []
    checkpoint_eval_rows: list[dict] = []
    checkpoint_paths: list[str] = []
    validate_every = int(config["training"]["validate_every_updates"])
    checkpoint_every = int(config["training"]["checkpoint_every_updates"])
    required_checkpoint_updates = {attempted_updates}
    if attempted_updates >= 1000:
        required_checkpoint_updates.add(1000)
    saved_checkpoint_updates: set[int] = set()

    def after_update(updated_components: dict, _row: dict) -> None:
        completed = int(updated_components["completed_updates"])
        should_checkpoint = (
            completed in required_checkpoint_updates
            or (checkpoint_every > 0 and completed % checkpoint_every == 0)
        )
        if should_checkpoint and completed not in saved_checkpoint_updates:
            checkpoint_path = _save_coverage_checkpoint(out_dir, updated_components, config, completed, run_name)
            checkpoint_paths.append(str(checkpoint_path))
            saved_checkpoint_updates.add(completed)
        if validate_every > 0 and (completed % validate_every == 0 or completed == attempted_updates):
            train_summary = _evaluate_model_on_mixtures(model, train_mixtures, config, device=device)
            validation_rows.extend(_tagged_rows(_summary_rows(train_summary, completed), eval_set="train"))
        if completed in required_checkpoint_updates:
            for eval_name, mixtures in (("legacy_probe", legacy_probe), ("d72", d72)):
                summary = _evaluate_model_on_mixtures(model, mixtures, config, device=device)
                checkpoint_path = out_dir / "checkpoints" / f"update_{completed:06d}.pt"
                checkpoint_eval_rows.extend(
                    _tagged_rows(_summary_rows(summary, completed, checkpoint_path=str(checkpoint_path)), eval_set=eval_name)
                )

    memory = {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = perf_counter()
    curves = train_toy_updates(components, attempted_updates, device=device, after_update=after_update)
    if device == "cuda":
        torch.cuda.synchronize()
        memory = {
            "cuda_train_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
            "cuda_train_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 3),
        }
    seconds = perf_counter() - started

    _write_rows(curves, out_dir / "train_curves.csv")
    _write_rows(validation_rows, out_dir / "validation_curves.csv")
    _write_rows(checkpoint_eval_rows, out_dir / "checkpoint_eval_summary.csv")
    _write_rows(sampler.access_count_rows(), out_dir / "sampler_access_counts.csv")

    combined = train_mixtures + legacy_probe + d72
    final_summary = _evaluate_model_on_mixtures(model, combined, config, device=device)
    write_query_errors(final_summary, out_dir / "query_errors_model.csv")
    _write_group_summary(final_summary, out_dir / "behavior_by_task_split_model.csv")

    loaded_rows, loaded_query_files = _evaluate_loaded_checkpoints(
        config=config,
        run_name=run_name,
        out_dir=out_dir,
        checkpoint_paths=[Path(path) for path in checkpoint_paths],
        eval_sets={"legacy_probe": legacy_probe, "d72": d72},
        device=device,
    )
    _write_rows(loaded_rows, out_dir / "checkpoint_eval_summary_loaded.csv")

    result = {
        "run_name": run_name,
        "status": "ROUND2_DATA_COVERAGE_RUN_COMPLETE",
        "attempted_updates": attempted_updates,
        "completed_updates": int(components["completed_updates"]),
        "train_mixtures": len(train_mixtures),
        "train_queries": len(dataset),
        "legacy_probe_mixtures": len(legacy_probe),
        "d72_mixtures": len(d72),
        "input_shape": _toy_input_shape(components, config),
        "parameter_count": model.count_parameters(),
        "initial_state_sha256": initial_hash,
        "loaded_initial_state_sha256": loaded_initial_hash,
        "train_seconds": round(seconds, 4),
        **memory,
        "data_checks": components["checks"],
        "model_behavior_summary": final_summary["aggregate"],
        "model_behavior_by_task_split": final_summary["aggregate_by_task_split"],
        "checkpoint_paths": checkpoint_paths,
        "loaded_checkpoint_query_error_files": loaded_query_files,
    }
    (out_dir / "behavior_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _evaluate_loaded_checkpoints(
    config: dict,
    run_name: str,
    out_dir: Path,
    checkpoint_paths: list[Path],
    eval_sets: dict[str, list[ToyMixture]],
    device: str,
) -> tuple[list[dict], list[str]]:
    from .models.separator import Separator

    rows: list[dict] = []
    query_error_files: list[str] = []
    for checkpoint_path in checkpoint_paths:
        model = Separator(config).to(device)
        checkpoint = load_training_checkpoint(checkpoint_path, model, expected_config=config, map_location=device)
        completed = int(checkpoint["completed_updates"])
        for eval_name, mixtures in eval_sets.items():
            summary = _evaluate_model_on_mixtures(model, mixtures, config, device=device)
            rows.extend(_tagged_rows(_summary_rows(summary, completed, checkpoint_path=str(checkpoint_path)), eval_set=eval_name))
            query_path = out_dir / f"query_errors_loaded_{eval_name}_{completed:06d}.csv"
            write_query_errors(summary, query_path)
            query_error_files.append(str(query_path))
    return rows, query_error_files


def _save_coverage_checkpoint(out_dir: Path, components: dict, config: dict, completed: int, run_name: str) -> Path:
    from .checkpoint import capture_rng_state

    checkpoint_path = out_dir / "checkpoints" / f"update_{completed:06d}.pt"
    save_training_checkpoint(
        checkpoint_path,
        components["model"],
        components["optimizer"],
        config,
        completed_updates=completed,
        rng_state=capture_rng_state(),
        sampler_state=components["sampler"].state_dict(),
        metadata={"stage": "round2_data_coverage", "run_name": run_name, "completed_updates": completed},
    )
    return checkpoint_path


def _initial_state(config: dict, device: str) -> tuple[dict, str]:
    import torch

    from .models.separator import Separator

    init_seed = stable_toy_seed(config["experiment"]["seed"], "coverage", "initial-state")
    random.seed(init_seed)
    np.random.seed(init_seed % (2**32))
    torch.manual_seed(init_seed)
    model = Separator(config).to(device)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return state, _state_dict_sha256(state)


def _state_dict_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _scene_counter(mixtures: list[ToyMixture]) -> Counter:
    return Counter(toy_scene_key(mixture) for mixture in mixtures)


def _check_scene_counts(mixtures: list[ToyMixture], task: str, expected_per_scene: int, label: str) -> None:
    counts = _scene_counter([mixture for mixture in mixtures if mixture.task == task])
    for scene in all_toy_scenes(task):
        key = scene_key_from_spec(task, scene)
        if counts[key] != expected_per_scene:
            raise ValueError(f"{label} {key}: expected {expected_per_scene}, got {counts[key]}")


def _coverage_rows(bank_name: str, mixtures: list[ToyMixture]) -> list[dict]:
    rows = []
    for task in ("T", "P"):
        task_mixtures = [mixture for mixture in mixtures if mixture.task == task]
        counts = _scene_counter(task_mixtures)
        for scene in all_toy_scenes(task):
            key = scene_key_from_spec(task, scene)
            rows.append({"bank": bank_name, "task": task, "scene_key": key, "count": int(counts[key])})
    return rows


def _manifest_hash_rows(manifest_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(manifest_dir.glob("*.jsonl")):
        rows.append({"path": str(path), "sha256": _file_sha256(path)})
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tagged_rows(rows: list[dict], eval_set: str) -> list[dict]:
    return [{"eval_set": eval_set, **row} for row in rows]
