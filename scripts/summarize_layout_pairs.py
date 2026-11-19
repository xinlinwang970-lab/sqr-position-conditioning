#!/usr/bin/env python3
"""Describe paired layout differences using saved query scores only.

No audio, weights, training or inference is used. Generate a deterministic
derived JSON, or use --check to compare it with the committed summary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT))
from verify_results import PARTS, recompute, rows_from

INPUTS = {
    "Baseline": "results/test_results/author_release_12ep_warmup_cosine_test_results.json",
    "P10": "results/checkpoint_evals/P10_test_ep424.json",
    "P15": "results/checkpoint_evals/P15_test_ep424.json",
    "P50": "results/checkpoint_evals/P50_test_ep424.json",
    "E4": "results/checkpoint_evals/E4_test_ep424.json",
}


def paired_statistics(type1, type2):
    """Aggregate already aligned pairs; the small mathematical seam allows hand checks."""
    first, second = list(type1), list(type2)
    if not first or len(first) != len(second):
        raise ValueError("Paired score sequences must have equal, nonzero length")
    values = first + second
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in values):
        raise ValueError("SDR scores must be finite numeric values")
    differences = [b - a for a, b in zip(first, second)]
    return {
        "overall_mean_sdr": mean(values),
        "type1_median": median(first),
        "type2_median": median(second),
        "delta_pooled_medians": median(second) - median(first),
        "median_paired_delta": median(differences),
        "median_paired_abs_delta": median(abs(value) for value in differences),
        "mean_paired_abs_delta": mean(abs(value) for value in differences),
        "mean_paired_delta": mean(differences),
        "positive_pairs": sum(value > 0 for value in differences),
        "negative_pairs": sum(value < 0 for value in differences),
        "zero_pairs": sum(value == 0 for value in differences),
    }


def summarize_rows(rows):
    """Require the complete frozen six-source, four-part, two-layout grid."""
    for row in rows:
        for key in ("composition_id", "part_id", "layout_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing or empty score identity: {key}")
        value = row.get("sdr")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("SDR scores must be finite numeric values")
        if "item_id" in row:
            prefix, separator, source = str(row["item_id"]).partition("-")
            if (not separator or prefix not in {"1", "2"} or "type" + prefix != row["layout_id"]
                    or source != row["composition_id"]):
                raise ValueError(f"item_id disagrees with source/layout identity: {row['item_id']}")
    # Reuse the existing duplicate, finite-score and full-grid validation.
    recompute(rows)
    grid = {(row["composition_id"], row["part_id"], row["layout_id"]): float(row["sdr"]) for row in rows}
    sources = sorted({key[0] for key in grid})
    pairs = []
    for source in sources:
        for part in PARTS:
            first, second = grid[source, part, "type1"], grid[source, part, "type2"]
            pairs.append({"source_id": source, "part_id": part, "type1_sdr": first, "type2_sdr": second,
                          "delta": second - first, "abs_delta": abs(second - first)})

    def aggregate(selected):
        return paired_statistics([pair["type1_sdr"] for pair in selected],
                                 [pair["type2_sdr"] for pair in selected])

    return {
        "source_ids": sources,
        "source_count": len(sources),
        "part_count": len(PARTS),
        "pair_count": len(pairs),
        "score_count": len(rows),
        "statistics": aggregate(pairs),
        "by_part": {part: {"source_count": len(sources), "pair_count": len(sources),
                           "score_count": 2 * len(sources),
                           "statistics": aggregate([pair for pair in pairs if pair["part_id"] == part])}
                    for part in PARTS},
        "pairs": pairs,
    }


def summarize_all(repo_root=REPO_ROOT):
    root = Path(repo_root)
    runs = {}
    cohort = None
    for label, relative in INPUTS.items():
        data = (root / relative).read_bytes()
        summary = summarize_rows(rows_from(json.loads(data)))
        if cohort is None:
            cohort = summary["source_ids"]
        elif summary["source_ids"] != cohort:
            raise ValueError("All methods must use the same source IDs and target parts")
        runs[label] = {"input": relative, "input_sha256": hashlib.sha256(data).hexdigest(), **summary}
    return {
        "schema_version": 1,
        "scope": "Post-hoc saved-score descriptive analysis; no new training or inference",
        "units": "dB",
        "metric": "Gain-dependent waveform SDR",
        "pair_key": ["source_id", "part_id"],
        "source_id_definition": "composition_id identifies a source recording in this frozen URMP selection; the legacy baseline adapter removes the first layout prefix from item_id",
        "delta_direction": "type2 minus type1",
        "definitions": {
            "overall_mean_sdr": "mean of all 48 query scores per run",
            "delta_pooled_medians": "median(type2) - median(type1)",
            "median_paired_delta": "median of source-part paired differences",
            "median_paired_abs_delta": "median of absolute source-part paired differences",
            "mean_paired_abs_delta": "mean of absolute source-part paired differences",
            "mean_paired_delta": "mean of signed source-part paired differences",
            "by_part": "The same statistics over 6 pairs / 12 scores for one part",
        },
        "statistical_unit_warning": "Each run has 24 correlated source-part pairs from 6 recordings; all runs reuse those recordings. No significance tests, confidence intervals or optimal-probability claims are made.",
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/layout_pair_summary.json")
    parser.add_argument("--check", action="store_true", help="Recompute and compare with the stored derived JSON without writing")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.check:
        protected = [REPO_ROOT / "results" / folder for folder in ("checkpoint_evals", "test_results", "provenance")]
        if any(output.is_relative_to(folder.resolve()) for folder in protected):
            parser.error("Output cannot replace raw results or historical evidence")
    report = summarize_all()
    if args.check:
        stored = json.loads(output.read_text(encoding="utf-8"))
        if stored != report:
            parser.error("Stored layout pair summary differs from deterministic recomputation")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": "LAYOUT_PAIRS_VERIFIED" if args.check else "LAYOUT_PAIRS_WRITTEN",
                      "runs": len(report["runs"]), "sources_per_run": 6, "pairs_per_run": 24,
                      "scores_per_run": 48, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
