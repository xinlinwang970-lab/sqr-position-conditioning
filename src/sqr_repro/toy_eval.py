from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .toy import ToyMixture


DEFAULT_TIE_ABS_TOL = 1.0e-8
DEFAULT_TIE_REL_TOL = 1.0e-6


def baseline_predictions(mixtures: list[ToyMixture], kind: str) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for mixture in mixtures:
        legal_targets = [mixture.target_for_query(query) for query in mixture.queries]
        blind = mixture.mixture / float(len(legal_targets))
        for query_index, query in enumerate(mixture.queries):
            if kind == "correct":
                prediction = mixture.target_for_query(query)
            elif kind == "blind":
                prediction = blind
            elif kind == "swapped":
                prediction = legal_targets[(query_index + 1) % len(legal_targets)]
            elif kind == "zero":
                prediction = np.zeros_like(mixture.mixture)
            else:
                raise ValueError(f"Unknown baseline kind: {kind}")
            predictions[query.query_id] = prediction.astype(np.float32, copy=False)
    return predictions


def evaluate_query_predictions(
    mixtures: list[ToyMixture],
    predictions: dict[str, np.ndarray],
    eps: float = 1.0e-12,
    tie_abs_tol: float = DEFAULT_TIE_ABS_TOL,
    tie_rel_tol: float = DEFAULT_TIE_REL_TOL,
) -> dict:
    _validate_query_index(mixtures, predictions)
    rows: list[dict] = []
    mixture_summaries: list[dict] = []
    for mixture in mixtures:
        target_energy = 0.0
        model_error = 0.0
        blind_error = 0.0
        zero_error = 0.0
        nearest_correct = 0
        unique_correct = 0
        tied_queries = 0
        blind = mixture.mixture / float(len(mixture.queries))
        for query in mixture.queries:
            target = mixture.target_for_query(query)
            prediction = _validated_prediction(predictions[query.query_id], query.query_id, target)
            errors = [squared_l2(prediction, source) for source in mixture.sources]
            selection = classify_selection(
                errors,
                query.source_index,
                tie_abs_tol=tie_abs_tol,
                tie_rel_tol=tie_rel_tol,
            )
            nearest_correct += int(selection["nearest_source_correct"])
            unique_correct += int(selection["unique_selected_correct"])
            tied_queries += int(selection["tie_count"] > 1)
            query_error = float(errors[query.source_index])
            energy = squared_l2(target, np.zeros_like(target))
            target_energy += energy
            model_error += query_error
            blind_error += squared_l2(blind, target)
            zero_error += energy
            rows.append(
                {
                    "mixture_id": mixture.mixture_id,
                    "task": mixture.task,
                    "split": mixture.split,
                    "query_id": query.query_id,
                    "source_index": query.source_index,
                    "nearest_source_index": selection["nearest_source_index"],
                    "nearest_source_correct": selection["nearest_source_correct"],
                    "unique_selected": selection["unique_selected"],
                    "unique_selected_correct": selection["unique_selected_correct"],
                    "tie_count": selection["tie_count"],
                    "tied_source_indices": selection["tied_source_indices"],
                    "target_error": query_error,
                    "all_source_errors": errors,
                }
            )
        denom = target_energy + eps
        mixture_summaries.append(
            {
                "mixture_id": mixture.mixture_id,
                "task": mixture.task,
                "split": mixture.split,
                "r_model": model_error / denom,
                "r_blind": blind_error / denom,
                "r_zero": zero_error / denom,
                "nearest_source_accuracy": nearest_correct / float(len(mixture.queries)),
                "unique_selection_accuracy": unique_correct / float(len(mixture.queries)),
                "tie_ratio": tied_queries / float(len(mixture.queries)),
                "num_queries": len(mixture.queries),
            }
        )
    return {
        "mixtures": mixture_summaries,
        "queries": rows,
        "aggregate": _aggregate(mixture_summaries),
        "aggregate_by_task_split": _aggregate_by_task_split(mixture_summaries),
        "tie_abs_tol": tie_abs_tol,
        "tie_rel_tol": tie_rel_tol,
    }


def decompose_two_query_errors(
    mixtures: list[ToyMixture],
    predictions: dict[str, np.ndarray],
    eps: float = 1.0e-12,
) -> dict:
    _validate_query_index(mixtures, predictions)
    rows: list[dict] = []
    for mixture in mixtures:
        if len(mixture.queries) != 2 or len(mixture.sources) != 2:
            raise ValueError(f"{mixture.mixture_id}: common/query decomposition requires exactly two queries and sources")
        q1, q2 = mixture.queries
        s1 = mixture.target_for_query(q1).astype(np.float64)
        s2 = mixture.target_for_query(q2).astype(np.float64)
        h1 = _validated_prediction(predictions[q1.query_id], q1.query_id, s1).astype(np.float64)
        h2 = _validated_prediction(predictions[q2.query_id], q2.query_id, s2).astype(np.float64)
        x = mixture.mixture.astype(np.float64)
        m_hat = (h1 + h2) / 2.0
        d_hat = (h1 - h2) / 2.0
        m = x / 2.0
        d = (s1 - s2) / 2.0
        denominator = squared_l2(s1, np.zeros_like(s1)) + squared_l2(s2, np.zeros_like(s2)) + eps
        r_common = 2.0 * squared_l2(m_hat, m) / denominator
        r_query = 2.0 * squared_l2(d_hat, d) / denominator
        r_raw = (squared_l2(h1, s1) + squared_l2(h2, s2)) / denominator
        correction = (x - h1 - h2) / 2.0
        h1_projected = h1 + correction
        h2_projected = h2 + correction
        r_projected = (squared_l2(h1_projected, s1) + squared_l2(h2_projected, s2)) / denominator
        blind = x / 2.0
        r_blind = (squared_l2(blind, s1) + squared_l2(blind, s2)) / denominator
        rows.append(
            {
                "mixture_id": mixture.mixture_id,
                "task": mixture.task,
                "split": mixture.split,
                "query_id_1": q1.query_id,
                "query_id_2": q2.query_id,
                "source_index_1": q1.source_index,
                "source_index_2": q2.source_index,
                "r_raw": float(r_raw),
                "r_common": float(r_common),
                "r_query": float(r_query),
                "r_projected": float(r_projected),
                "r_blind": float(r_blind),
                "identity_abs_error": float(abs(r_raw - (r_common + r_query))),
                "projected_query_abs_error": float(abs(r_projected - r_query)),
            }
        )
    return {
        "mixtures": rows,
        "aggregate": _aggregate_decomposition(rows),
        "aggregate_by_task_split": _aggregate_decomposition_by_task_split(rows),
        "eps": eps,
    }


def write_error_decomposition(summary: dict, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mixture_id",
        "task",
        "split",
        "query_id_1",
        "query_id_2",
        "source_index_1",
        "source_index_2",
        "r_raw",
        "r_common",
        "r_query",
        "r_projected",
        "r_blind",
        "identity_abs_error",
        "projected_query_abs_error",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["mixtures"])


def write_query_errors(summary: dict, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mixture_id",
                "task",
                "split",
                "query_id",
                "source_index",
                "nearest_source_index",
                "nearest_source_correct",
                "unique_selected",
                "unique_selected_correct",
                "tie_count",
                "tied_source_indices",
                "target_error",
                "all_source_errors",
            ],
        )
        writer.writeheader()
        for row in summary["queries"]:
            out_row = dict(row)
            out_row["tied_source_indices"] = json.dumps(out_row["tied_source_indices"])
            out_row["all_source_errors"] = json.dumps(out_row["all_source_errors"])
            writer.writerow(out_row)


def squared_l2(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"squared_l2 shape mismatch: {left.shape} != {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("squared_l2 received NaN or Inf")
    diff = left.astype(np.float64) - right.astype(np.float64)
    return float(np.sum(diff * diff))


def classify_selection(
    errors: list[float],
    target_index: int,
    tie_abs_tol: float = DEFAULT_TIE_ABS_TOL,
    tie_rel_tol: float = DEFAULT_TIE_REL_TOL,
) -> dict:
    values = np.array(errors, dtype=np.float64)
    if target_index < 0 or target_index >= len(values):
        raise ValueError(f"target_index {target_index} is outside {len(values)} source errors")
    if not np.isfinite(values).all():
        raise ValueError("selection errors contain NaN or Inf")
    nearest = int(np.argmin(values))
    minimum = float(values[nearest])
    tolerance = float(tie_abs_tol + tie_rel_tol * max(1.0, abs(minimum)))
    tied = np.flatnonzero(np.abs(values - minimum) <= tolerance).astype(int).tolist()
    unique = len(tied) == 1
    return {
        "nearest_source_index": nearest,
        "nearest_source_correct": nearest == int(target_index),
        "unique_selected": unique,
        "unique_selected_correct": unique and nearest == int(target_index),
        "tie_count": len(tied),
        "tied_source_indices": tied,
        "tie_tolerance": tolerance,
    }


def _validate_query_index(mixtures: list[ToyMixture], predictions: dict[str, np.ndarray]) -> None:
    expected: list[str] = []
    for mixture in mixtures:
        num_sources = len(mixture.sources)
        for query in mixture.queries:
            if query.source_index < 0 or query.source_index >= num_sources:
                raise ValueError(
                    f"{query.query_id}: source_index {query.source_index} is outside {num_sources} sources"
                )
            expected.append(query.query_id)
    duplicates = sorted({query_id for query_id in expected if expected.count(query_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate query id(s): {', '.join(duplicates)}")
    expected_set = set(expected)
    prediction_set = set(predictions)
    missing = sorted(expected_set - prediction_set)
    unexpected = sorted(prediction_set - expected_set)
    if missing:
        raise ValueError(f"Missing prediction query id(s): {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected prediction query id(s): {', '.join(unexpected)}")


def _validated_prediction(prediction: np.ndarray, query_id: str, target: np.ndarray) -> np.ndarray:
    array = np.asarray(prediction)
    if array.shape != target.shape:
        raise ValueError(f"{query_id}: prediction shape {array.shape} does not match target {target.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{query_id}: prediction contains NaN or Inf")
    return array


def _aggregate(mixture_summaries: list[dict]) -> dict:
    if not mixture_summaries:
        return {
            "median_r_model": None,
            "median_r_blind": None,
            "median_r_zero": None,
            "mean_nearest_source_accuracy": None,
            "mean_unique_selection_accuracy": None,
            "mean_tie_ratio": None,
            "total_mixtures": 0,
            "total_queries": 0,
        }
    return {
        "median_r_model": float(np.median([item["r_model"] for item in mixture_summaries])),
        "median_r_blind": float(np.median([item["r_blind"] for item in mixture_summaries])),
        "median_r_zero": float(np.median([item["r_zero"] for item in mixture_summaries])),
        "mean_nearest_source_accuracy": float(np.mean([item["nearest_source_accuracy"] for item in mixture_summaries])),
        "mean_unique_selection_accuracy": float(
            np.mean([item["unique_selection_accuracy"] for item in mixture_summaries])
        ),
        "mean_tie_ratio": float(np.mean([item["tie_ratio"] for item in mixture_summaries])),
        "total_mixtures": int(len(mixture_summaries)),
        "total_queries": int(sum(item["num_queries"] for item in mixture_summaries)),
    }


def _aggregate_by_task_split(mixture_summaries: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for item in mixture_summaries:
        key = f"{item['task']}_{item['split']}"
        grouped.setdefault(key, []).append(item)
    return {key: _aggregate(items) for key, items in sorted(grouped.items())}


def _aggregate_decomposition(rows: list[dict]) -> dict:
    if not rows:
        return {
            "mean_r_raw": None,
            "mean_r_common": None,
            "mean_r_query": None,
            "mean_r_projected": None,
            "mean_r_blind": None,
            "median_r_raw": None,
            "median_r_common": None,
            "median_r_query": None,
            "median_r_projected": None,
            "median_r_blind": None,
            "max_identity_abs_error": None,
            "max_projected_query_abs_error": None,
            "total_mixtures": 0,
        }
    return {
        "mean_r_raw": float(np.mean([row["r_raw"] for row in rows])),
        "mean_r_common": float(np.mean([row["r_common"] for row in rows])),
        "mean_r_query": float(np.mean([row["r_query"] for row in rows])),
        "mean_r_projected": float(np.mean([row["r_projected"] for row in rows])),
        "mean_r_blind": float(np.mean([row["r_blind"] for row in rows])),
        "median_r_raw": float(np.median([row["r_raw"] for row in rows])),
        "median_r_common": float(np.median([row["r_common"] for row in rows])),
        "median_r_query": float(np.median([row["r_query"] for row in rows])),
        "median_r_projected": float(np.median([row["r_projected"] for row in rows])),
        "median_r_blind": float(np.median([row["r_blind"] for row in rows])),
        "max_identity_abs_error": float(max(row["identity_abs_error"] for row in rows)),
        "max_projected_query_abs_error": float(max(row["projected_query_abs_error"] for row in rows)),
        "total_mixtures": int(len(rows)),
    }


def _aggregate_decomposition_by_task_split(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['task']}_{row['split']}"
        grouped.setdefault(key, []).append(row)
    return {key: _aggregate_decomposition(items) for key, items in sorted(grouped.items())}
