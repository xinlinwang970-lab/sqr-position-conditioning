#!/usr/bin/env python3
"""Recompute saved aggregate scores and verify bundled evidence hashes.

No audio, checkpoint loading or inference is performed. Exit nonzero on any
changed raw evidence, duplicated query, incomplete grid, or inconsistent summary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTS = ("violin1", "violin2", "viola", "cello")


def rows_from(payload):
    if isinstance(payload, list):
        rows = payload
    else:
        rows = next((payload[key] for key in ("by_track", "results") if isinstance(payload.get(key), list)), None)
    if rows is None:
        raise ValueError("No per-query scores in payload")
    # The historical dense baseline stores item_id, without composition/layout.
    # Preserve bytes and adapt only the calculation view.
    adapted = []
    for source in rows:
        row = dict(source)
        if "composition_id" not in row:
            row["composition_id"] = row["item_id"].split("-", 1)[1]
        if "layout_id" not in row:
            row["layout_id"] = f"type{row['item_id'].split('-', 1)[0]}"
        adapted.append(row)
    return adapted


def recompute(rows):
    keys = [(r["composition_id"], r["layout_id"], r["part_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate composition/layout/part query")
    works = sorted({r["composition_id"] for r in rows})
    expected = {(work, layout, part) for work in works for layout in ("type1", "type2") for part in PARTS}
    if set(keys) != expected or len(works) != 6:
        raise ValueError("Expected complete 6-work x 2-layout x 4-part test grid")
    values = [float(r["sdr"]) for r in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite score")
    return {"mean_sdr": mean(values), "median_sdr": median(values),
            **{f"{layout}_median": median(float(r["sdr"]) for r in rows if r["layout_id"] == layout) for layout in ("type1", "type2")},
            "by_part_median": {part: median(float(r["sdr"]) for r in rows if r["part_id"] == part) for part in PARTS},
            "n": len(rows), "composition_ids": works}


def check_summary(payload, computed):
    if isinstance(payload, list):
        return
    saved = payload.get("summary", payload.get("final", payload))
    aliases = {"overall_mean_sdr": "mean_sdr", "overall_median_sdr": "median_sdr", "mean": "mean_sdr", "median": "median_sdr"}
    for key, value in saved.items():
        target = aliases.get(key, key)
        if target in computed and isinstance(value, (int, float)) and not math.isclose(value, computed[target], abs_tol=1e-6, rel_tol=0):
            raise ValueError(f"Saved summary differs for {key}: {value} vs {computed[target]}")


def verify(repo_root=REPO_ROOT):
    repo_root = Path(repo_root)
    files = sorted((repo_root / "results/checkpoint_evals").glob("*.json"))
    files += sorted((repo_root / "results/test_results").glob("*.json"))
    scores = {}
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        computed = recompute(rows_from(payload))
        check_summary(payload, computed)
        scores[file.relative_to(repo_root).as_posix()] = computed
    if len(files) != 11:
        raise ValueError(f"Expected 11 original result files, found {len(files)}")
    inventory = json.loads((repo_root / "results/provenance/imported_files.json").read_text(encoding="utf-8"))
    original = json.loads((repo_root / "results/provenance/release_origin_files.json").read_text(encoding="utf-8"))
    # Original result bytes must stay unchanged. Setup/docs changes are intentional.
    selected_original = [r for r in original["files"] if r["path"].startswith("results/")]
    for record in inventory + selected_original:
        file = repo_root / record["path"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {record['path']}")
    return {"status": "SAVED_RESULTS_VERIFIED", "result_files": len(files), "queries": sum(r["n"] for r in scores.values()),
            "evidence_hashes_checked": len(inventory) + len(selected_original), "scores": scores,
            "scope": "Arithmetic and preserved file bytes; no fresh checkpoint/audio inference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
