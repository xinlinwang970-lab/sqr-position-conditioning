from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .checkpoint import load_training_checkpoint
from .config import load_config
from .coverage import build_coverage_banks, validate_coverage_banks
from .resume import build_s144_resume_components
from .toy import ToyMixture, toy_scene_key
from .toy_eval import decompose_two_query_errors, evaluate_query_predictions
from .trainer import _write_rows, predict_toy_waveforms


DEFAULT_CHECKPOINTS = (6000, 12000, 18000)


def run_scene_level_diagnostic(
    config_path: str | Path,
    run_dir: str | Path,
    out_dir: str | Path,
    device: str = "cuda",
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
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
    manifest_checks = _check_manifest_identity(root, banks, out)
    if not manifest_checks["ok"]:
        raise ValueError("Manifest identity check failed: " + "; ".join(manifest_checks["errors"]))
    access_rows = _read_access_counts(root / "sampler_access_counts.csv")
    access_checks = _check_access_counts(access_rows, banks["s144_train"])
    (out / "diagnostic_policy.json").write_text(
        json.dumps(
            {
                "parameter_updates": 0,
                "training_started": False,
                "model_modified": False,
                "loss_modified": False,
                "sampling_modified": False,
                "data_modified": False,
                "checkpoints": list(checkpoints),
                "r_statistics": "per-mixture R; mean and median reported separately",
                "postprocess": "offline mixture-consistency projection only; never replaces raw score",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scene_rows: list[dict] = []
    mixture_rows: list[dict] = []
    consistency_rows: list[dict] = []
    direction_rows: list[dict] = []
    checkpoint_summaries: list[dict] = []

    for update in checkpoints:
        checkpoint_path = root / "checkpoints" / f"update_{update:06d}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        model = _load_eval_model(config, checkpoint_path, device)
        for eval_name, mixtures in (("s144_train", banks["s144_train"]), ("d72", banks["d72"])):
            before_state = _model_state_hash(model)
            predictions = predict_toy_waveforms(model, mixtures, config, device=device)
            after_state = _model_state_hash(model)
            if before_state != after_state:
                raise RuntimeError(f"Model state changed during read-only evaluation: update={update}, eval={eval_name}")
            summary = evaluate_query_predictions(mixtures, predictions)
            decomposition = decompose_two_query_errors(mixtures, predictions)
            mixture_detail = _build_mixture_detail(update, eval_name, mixtures, predictions, access_rows)
            mixture_rows.extend(mixture_detail)
            current_directions = _direction_rows(update, eval_name, mixtures, predictions)
            direction_rows.extend(current_directions)
            consistency_rows.extend(_consistency_rows(update, eval_name, mixture_detail))
            scene_rows.extend(_aggregate_scene_rows(update, eval_name, mixture_detail))
            checkpoint_summaries.extend(
                _checkpoint_summary_rows(
                    update, eval_name, summary, decomposition, mixture_detail, current_directions
                )
            )

    _write_rows(scene_rows, out / "scene_summary.csv")
    _write_rows(mixture_rows, out / "mixture_detail.csv")
    _write_rows(direction_rows, out / "query_direction_detail.csv")
    _write_rows(consistency_rows, out / "mixture_consistency_detail.csv")
    _write_rows(checkpoint_summaries, out / "checkpoint_summary.csv")
    _write_rows(_scene_access_rows(banks["s144_train"], access_rows), out / "scene_access_summary.csv")

    result = {
        "status": "ROUND2_SCENE_DIAGNOSTIC_COMPLETE_NO_PARAMETER_UPDATES",
        "parameter_updates": 0,
        "device": device,
        "config_path": str(config_path),
        "run_dir": str(root),
        "checkpoints": list(checkpoints),
        "coverage_checks": checks,
        "manifest_checks": manifest_checks,
        "access_checks": access_checks,
        "outputs": {
            "scene_summary": str(out / "scene_summary.csv"),
            "mixture_detail": str(out / "mixture_detail.csv"),
            "query_direction_detail": str(out / "query_direction_detail.csv"),
            "mixture_consistency_detail": str(out / "mixture_consistency_detail.csv"),
            "checkpoint_summary": str(out / "checkpoint_summary.csv"),
            "scene_access_summary": str(out / "scene_access_summary.csv"),
        },
    }
    (out / "diagnostic_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _load_eval_model(config: dict, checkpoint_path: Path, device: str):
    components = build_s144_resume_components(config, device=device)
    checkpoint = load_training_checkpoint(
        checkpoint_path,
        components["model"],
        optimizer=components["optimizer"],
        expected_config=config,
        map_location=device,
    )
    if int(checkpoint["completed_updates"]) != int(checkpoint_path.stem.split("_")[-1]):
        raise ValueError(f"Checkpoint update mismatch: {checkpoint_path}")
    components["model"].eval()
    return components["model"]


def _build_mixture_detail(
    update: int,
    eval_name: str,
    mixtures: list[ToyMixture],
    predictions: dict[str, np.ndarray],
    access_rows: dict[str, int],
) -> list[dict]:
    rows: list[dict] = []
    for mixture in mixtures:
        if len(mixture.queries) != 2:
            raise ValueError(f"{mixture.mixture_id}: expected exactly two queries")
        q1, q2 = mixture.queries
        s1 = mixture.target_for_query(q1).astype(np.float64)
        s2 = mixture.target_for_query(q2).astype(np.float64)
        h1 = np.asarray(predictions[q1.query_id], dtype=np.float64)
        h2 = np.asarray(predictions[q2.query_id], dtype=np.float64)
        x = mixture.mixture.astype(np.float64)
        denominator = float(np.sum(s1 * s1) + np.sum(s2 * s2) + 1.0e-12)
        raw_num = float(np.sum((h1 - s1) ** 2) + np.sum((h2 - s2) ** 2))
        common_hat = (h1 + h2) / 2.0
        common = x / 2.0
        diff_hat = (h1 - h2) / 2.0
        diff = (s1 - s2) / 2.0
        common_num = float(2.0 * np.sum((common_hat - common) ** 2))
        query_num = float(2.0 * np.sum((diff_hat - diff) ** 2))
        correction = (x - h1 - h2) / 2.0
        projected_h1 = h1 + correction
        projected_h2 = h2 + correction
        projected_num = float(np.sum((projected_h1 - s1) ** 2) + np.sum((projected_h2 - s2) ** 2))
        rows.append(
            {
                "update": int(update),
                "eval_set": eval_name,
                "mixture_id": mixture.mixture_id,
                "task": mixture.task,
                "split": mixture.split,
                "scene_key": toy_scene_key(mixture),
                "source_hash_1": _array_hash(s1),
                "source_hash_2": _array_hash(s2),
                "query_id_1": q1.query_id,
                "query_id_2": q2.query_id,
                "query_visits_1": int(access_rows.get(q1.query_id, 0)),
                "query_visits_2": int(access_rows.get(q2.query_id, 0)),
                "denominator": denominator,
                "raw_numerator": raw_num,
                "common_numerator": common_num,
                "query_numerator": query_num,
                "projected_numerator": projected_num,
                "mixture_residual_l2": float(np.sqrt(np.sum((h1 + h2 - x) ** 2))),
                "r_raw": raw_num / denominator,
                "r_common": common_num / denominator,
                "r_query": query_num / denominator,
                "r_projected": projected_num / denominator,
                "r_blind": float(
                    np.sum((x / 2.0 - s1) ** 2) + np.sum((x / 2.0 - s2) ** 2)
                )
                / denominator,
                "identity_abs_error": abs(raw_num / denominator - (common_num + query_num) / denominator),
                "projected_query_abs_error": abs(projected_num / denominator - query_num / denominator),
            }
        )
    return rows


def _direction_rows(update: int, eval_name: str, mixtures: list[ToyMixture], predictions: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for mixture in mixtures:
        q1, q2 = mixture.queries
        target_diff = (mixture.target_for_query(q1) - mixture.target_for_query(q2)).astype(np.float64).reshape(-1)
        predicted_diff = (predictions[q1.query_id] - predictions[q2.query_id]).astype(np.float64).reshape(-1)
        target_norm = float(np.linalg.norm(target_diff))
        predicted_norm = float(np.linalg.norm(predicted_diff))
        cosine = float(np.dot(predicted_diff, target_diff) / (predicted_norm * target_norm)) if target_norm and predicted_norm else 0.0
        rows.append(
            {
                "update": int(update),
                "eval_set": eval_name,
                "mixture_id": mixture.mixture_id,
                "task": mixture.task,
                "scene_key": toy_scene_key(mixture),
                "target_difference_norm": target_norm,
                "predicted_difference_norm": predicted_norm,
                "difference_amplitude_ratio": predicted_norm / target_norm if target_norm else 0.0,
                "difference_direction_cosine": cosine,
                "difference_relative_error": float(np.linalg.norm(predicted_diff - target_diff) / target_norm) if target_norm else 0.0,
            }
        )
    return rows


def _consistency_rows(update: int, eval_name: str, mixture_rows: list[dict]) -> list[dict]:
    return [
        {
            "update": row["update"],
            "eval_set": eval_name,
            "mixture_id": row["mixture_id"],
            "scene_key": row["scene_key"],
            "raw_r": row["r_raw"],
            "projected_r": row["r_projected"],
            "raw_mixture_residual_l2": row["mixture_residual_l2"],
            "postprocessed_mixture_residual_l2": 0.0,
            "raw_score_is_formal": True,
            "projected_score_is_formal": False,
        }
        for row in mixture_rows
    ]


def _aggregate_scene_rows(update: int, eval_name: str, rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["scene_key"]].append(row)
    out = []
    for scene_key, scene_rows in sorted(grouped.items()):
        raw_values = [row["r_raw"] for row in scene_rows]
        common_values = [row["r_common"] for row in scene_rows]
        query_values = [row["r_query"] for row in scene_rows]
        total_raw = float(sum(raw_values))
        top_count = max(1, int(np.ceil(len(raw_values) * 0.10)))
        top_share = float(sum(sorted(raw_values, reverse=True)[:top_count]) / total_raw) if total_raw else 0.0
        out.append(
            {
                "update": int(update),
                "eval_set": eval_name,
                "scene_key": scene_key,
                "task": scene_rows[0]["task"],
                "mixture_count": len(scene_rows),
                "mean_r_raw": float(np.mean(raw_values)),
                "median_r_raw": float(np.median(raw_values)),
                "mean_r_common": float(np.mean(common_values)),
                "median_r_common": float(np.median(common_values)),
                "mean_r_query": float(np.mean(query_values)),
                "median_r_query": float(np.median(query_values)),
                "sum_r_raw": total_raw,
                "top_10pct_r_raw_sum_share": top_share,
                "mean_r_raw_minus_median": float(np.mean(raw_values) - np.median(raw_values)),
                "p90_r_raw": float(np.percentile(raw_values, 90)),
                "max_r_raw": float(np.max(raw_values)),
                "min_query_visits": min(min(row["query_visits_1"], row["query_visits_2"]) for row in scene_rows),
                "max_query_visits": max(max(row["query_visits_1"], row["query_visits_2"]) for row in scene_rows),
                "mean_query_visits": float(np.mean([row["query_visits_1"] + row["query_visits_2"] for row in scene_rows]) / 2.0),
            }
        )
    return out


def _checkpoint_summary_rows(
    update: int,
    eval_name: str,
    summary: dict,
    decomposition: dict,
    rows: list[dict],
    directions: list[dict],
) -> list[dict]:
    aggregate = summary["aggregate"]
    decomp = decomposition["aggregate"]
    return [
        {
            "update": int(update),
            "eval_set": eval_name,
            "median_r_model_formal": aggregate["median_r_model"],
            "mean_r_raw_diagnostic": float(np.mean([row["r_raw"] for row in rows])),
            "median_r_raw_diagnostic": float(np.median([row["r_raw"] for row in rows])),
            "mean_r_common": decomp["mean_r_common"],
            "median_r_common": decomp["median_r_common"],
            "mean_r_query": decomp["mean_r_query"],
            "median_r_query": decomp["median_r_query"],
            "mean_r_projected_offline": float(np.mean([row["r_projected"] for row in rows])),
            "median_r_projected_offline": float(np.median([row["r_projected"] for row in rows])),
            "mean_difference_amplitude_ratio": float(
                np.mean([row["difference_amplitude_ratio"] for row in directions])
            ),
            "median_difference_amplitude_ratio": float(
                np.median([row["difference_amplitude_ratio"] for row in directions])
            ),
            "mean_direction_cosine": float(np.mean([row["difference_direction_cosine"] for row in directions])),
            "median_direction_cosine": float(np.median([row["difference_direction_cosine"] for row in directions])),
            "max_identity_abs_error": max(row["identity_abs_error"] for row in rows),
        }
    ]


def _check_manifest_identity(root: Path, banks: dict[str, list[ToyMixture]], out: Path) -> dict:
    errors = []
    rows = []
    for name, mixtures, filename in (
        ("s144_train", banks["s144_train"], "s144_train.jsonl"),
        ("d72", banks["d72"], "d72.jsonl"),
    ):
        parent = root / "manifests" / filename.replace("_manifest", "")
        generated = out / f"{name}_manifest.jsonl"
        generated.write_text("\n".join(json.dumps(m.manifest_record(), sort_keys=True) for m in mixtures) + "\n", encoding="utf-8")
        parent_hash = _file_hash(parent)
        generated_hash = _file_hash(generated)
        if parent_hash != generated_hash:
            errors.append(f"{name}: manifest hash mismatch")
        rows.append({"name": name, "parent_sha256": parent_hash, "generated_sha256": generated_hash, "match": parent_hash == generated_hash})
    (out / "manifest_identity.json").write_text(json.dumps({"ok": not errors, "rows": rows, "errors": errors}, indent=2), encoding="utf-8")
    return {"ok": not errors, "rows": rows, "errors": errors}


def _read_access_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["query_id"]: int(row["visits"]) for row in csv.DictReader(handle)}


def _check_access_counts(access_rows: dict[str, int], mixtures: list[ToyMixture]) -> dict:
    expected = {query.query_id for mixture in mixtures for query in mixture.queries}
    missing = sorted(expected - set(access_rows))
    unexpected = sorted(set(access_rows) - expected)
    visits = [access_rows[query_id] for query_id in sorted(expected) if query_id in access_rows]
    return {
        "ok": not missing and not unexpected,
        "missing_query_ids": missing,
        "unexpected_query_ids": unexpected,
        "min_visits": min(visits) if visits else None,
        "max_visits": max(visits) if visits else None,
        "mean_visits": float(np.mean(visits)) if visits else None,
        "unique_visit_counts": sorted(set(visits)),
    }


def _scene_access_rows(mixtures: list[ToyMixture], access_rows: dict[str, int]) -> list[dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for mixture in mixtures:
        grouped[toy_scene_key(mixture)].extend(access_rows.get(query.query_id, 0) for query in mixture.queries)
    return [
        {
            "scene_key": scene,
            "query_rows": len(visits),
            "min_visits": min(visits),
            "max_visits": max(visits),
            "mean_visits": float(np.mean(visits)),
            "total_visits": int(sum(visits)),
        }
        for scene, visits in sorted(grouped.items())
    ]


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array.astype(np.float32)).tobytes()).hexdigest()


def _model_state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
