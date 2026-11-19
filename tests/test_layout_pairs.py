import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("layout_pairs", REPO / "scripts/summarize_layout_pairs.py")
layout_pairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout_pairs)


def grid_rows():
    rows = []
    for source_index in range(6):
        source = f"source-{source_index}"
        for part_index, part in enumerate(layout_pairs.PARTS):
            first = source_index * 10 + part_index
            delta = source_index + part_index + 1
            for prefix, layout, value in ((1, "type1", first), (2, "type2", first + delta)):
                rows.append({"composition_id": source, "item_id": f"{prefix}-{source}",
                             "part_id": part, "layout_id": layout, "sdr": value})
    return rows


def test_hand_calculation_distinguishes_pooled_and_paired_statistics():
    result = layout_pairs.paired_statistics([0, 10, 11], [9, 0, 12])
    assert result["delta_pooled_medians"] == -1
    assert result["median_paired_delta"] == 1
    assert result["median_paired_abs_delta"] == 9
    assert result["mean_paired_abs_delta"] == pytest.approx(20 / 3)
    assert result["mean_paired_delta"] == 0
    assert result["overall_mean_sdr"] == 7


def test_pairs_use_source_and_part_independent_of_input_order():
    rows = grid_rows()
    expected = layout_pairs.summarize_rows(rows)
    random.Random(31).shuffle(rows)
    assert layout_pairs.summarize_rows(rows) == expected
    assert expected["source_count"] == 6
    assert expected["pair_count"] == 24
    assert expected["score_count"] == 48
    for pair in expected["pairs"]:
        source_index = int(pair["source_id"].split("-")[1])
        part_index = layout_pairs.PARTS.index(pair["part_id"])
        assert pair["delta"] == source_index + part_index + 1


def test_duplicate_query_is_rejected():
    rows = grid_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        layout_pairs.summarize_rows(rows + [rows[0]])


def test_missing_layout_or_source_is_rejected():
    rows = grid_rows()
    with pytest.raises(ValueError, match="complete"):
        layout_pairs.summarize_rows(rows[1:])
    with pytest.raises(ValueError, match="complete"):
        layout_pairs.summarize_rows([row for row in rows if row["composition_id"] != "source-0"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "1.0", None, True])
def test_invalid_scores_fail_before_aggregation(value):
    rows = grid_rows()
    rows[0]["sdr"] = value
    with pytest.raises(ValueError, match="finite numeric"):
        layout_pairs.summarize_rows(rows)


def test_item_prefix_and_explicit_layout_must_agree():
    rows = grid_rows()
    rows[0]["item_id"] = "2-source-0"
    with pytest.raises(ValueError, match="item_id"):
        layout_pairs.summarize_rows(rows)


def test_legacy_baseline_identity_adapter_preserves_layout_pairing():
    rows = grid_rows()
    legacy = [{key: value for key, value in row.items() if key not in {"composition_id", "layout_id"}} for row in rows]
    adapted = layout_pairs.rows_from(legacy)
    assert layout_pairs.summarize_rows(adapted) == layout_pairs.summarize_rows(rows)


def test_different_source_cohorts_across_methods_are_rejected(tmp_path):
    for label, relative in layout_pairs.INPUTS.items():
        rows = grid_rows()
        if label == "P50":
            for row in rows:
                if row["composition_id"] == "source-0":
                    row["composition_id"] = "different-source"
                    row["item_id"] = row["item_id"].split("-", 1)[0] + "-different-source"
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps({"by_track": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="same source"):
        layout_pairs.summarize_all(tmp_path)


def test_saved_reference_values_are_recomputed_from_raw_files():
    report = layout_pairs.summarize_all(REPO)
    expected = {
        "Baseline": (1.965948, 0.955149, 1.346252, 2.428426),
        "P10": (0.370914, 0.495831, 0.789737, 1.084389),
        "P15": (0.035617, 0.630755, 0.751140, 0.826208),
        "P50": (-0.082794, 0.605020, 0.680916, 0.826974),
        "E4": (0.101539, 0.419219, 0.607307, 0.733515),
    }
    keys = ("delta_pooled_medians", "median_paired_delta", "median_paired_abs_delta", "mean_paired_abs_delta")
    for label, values in expected.items():
        run = report["runs"][label]
        assert (run["source_count"], run["part_count"], run["pair_count"], run["score_count"]) == (6, 4, 24, 48)
        assert tuple(run["statistics"][key] for key in keys) == pytest.approx(values, abs=1e-6, rel=0)
    p15, e4 = report["runs"]["P15"], report["runs"]["E4"]
    assert p15["by_part"]["viola"]["statistics"]["overall_mean_sdr"] == pytest.approx(0.9553594055275122)
    assert e4["by_part"]["viola"]["statistics"]["overall_mean_sdr"] == pytest.approx(1.1047311735649903)
    assert e4["by_part"]["viola"]["pair_count"] == 6
    assert report["runs"]["P50"]["statistics"]["mean_paired_delta"] == pytest.approx(0.6707648495212197)


def test_committed_summary_matches_deterministic_recomputation():
    generated = layout_pairs.summarize_all(REPO)
    committed = json.loads((REPO / "results/layout_pair_summary.json").read_text(encoding="utf-8"))
    assert committed == generated


def test_readme_and_report_numbers_match_the_derived_statistics():
    report = layout_pairs.summarize_all(REPO)
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for label in ("P10", "P15", "P50"):
        line = next(line for line in readme.splitlines() if line.startswith(f"| {label} |"))
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        stats = report["runs"][label]["statistics"]
        keys = ("overall_mean_sdr", "type1_median", "type2_median", "delta_pooled_medians", "median_paired_abs_delta")
        assert cells[2:] == [f"{stats[key]:.3f}" for key in keys]
    claim = readme.split("**3.", 1)[1].split("Metric:", 1)[0]
    for label in ("P15", "E4"):
        run = report["runs"][label]
        for value in (run["statistics"]["overall_mean_sdr"], run["statistics"]["median_paired_abs_delta"],
                      run["by_part"]["viola"]["statistics"]["overall_mean_sdr"]):
            assert f"{value:.3f}" in claim
    text = (REPO / "docs/EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    paired = text.split("## Paired layout analysis", 1)[1].split("## Per-part medians", 1)[0]
    keys = ("overall_mean_sdr", "delta_pooled_medians", "median_paired_delta", "median_paired_abs_delta", "mean_paired_abs_delta", "mean_paired_delta")
    for label, run in report["runs"].items():
        line = next(line for line in paired.splitlines() if line.startswith(f"| {label} |"))
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        assert cells[1:] == [f"{run['statistics'][key]:.6f}" for key in keys]


def test_cli_check_from_external_working_directory_and_tamper_detection(tmp_path):
    script = REPO / "scripts/summarize_layout_pairs.py"
    output = tmp_path / "summary.json"
    def command(*args):
        return subprocess.run([sys.executable, str(script), *map(str, args)], cwd=tmp_path,
                              capture_output=True, text=True, timeout=30)
    assert command("--output", output).returncode == 0
    initial_bytes = output.read_bytes()
    assert b"\r\n" not in initial_bytes
    assert command("--output", output).returncode == 0
    assert output.read_bytes() == initial_bytes
    assert command("--output", output, "--check").returncode == 0
    changed = json.loads(output.read_text())
    changed["runs"]["E4"]["statistics"]["median_paired_abs_delta"] += 0.1
    output.write_text(json.dumps(changed))
    result = command("--output", output, "--check")
    assert result.returncode != 0
    assert "differs" in result.stderr


def test_cli_refuses_to_overwrite_raw_evidence():
    script = REPO / "scripts/summarize_layout_pairs.py"
    source = REPO / layout_pairs.INPUTS["P15"]
    before = source.read_bytes()
    result = subprocess.run([sys.executable, str(script), "--output", str(source)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert source.read_bytes() == before
