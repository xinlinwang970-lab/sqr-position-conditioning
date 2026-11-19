import json
from pathlib import Path

import numpy as np
import pytest

from sqr_repro.data import DataContractError, crop_same_start, split_leakage_keys, target_ids, validate_manifest_record
from sqr_repro.fixtures import make_fixtures


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "test_artifacts"


def test_target_ids_for_viola_type1():
    timbre, position = target_ids("viola", {"violin1": 3, "violin2": 0, "viola": 1, "cello": 2})
    assert timbre == 1
    assert position == 1


def test_fixture_manifest_validates():
    out_dir = ARTIFACTS / "fixture_valid"
    manifest = make_fixtures(out_dir, seed=123, sample_rate=8000, seconds=0.2)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    validate_manifest_record(record, out_dir, require_audio=True)
    assert record["is_fixture"] is True


def test_manifest_rejects_missing_stem():
    out_dir = ARTIFACTS / "fixture_missing"
    manifest = make_fixtures(out_dir, seed=123, sample_rate=8000, seconds=0.2)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    del record["stems"]["viola"]
    with pytest.raises(DataContractError, match="stems"):
        validate_manifest_record(record, out_dir, require_audio=False)


def test_crop_same_start():
    arrays = [np.arange(10), np.arange(100, 110)]
    cropped = crop_same_start(arrays, start=2, length=4)
    assert cropped[0].tolist() == [2, 3, 4, 5]
    assert cropped[1].tolist() == [102, 103, 104, 105]


def test_split_leakage_detection():
    leakage = split_leakage_keys(
        [
            {"composition_id": "a", "performance_id": "p1", "split": "train"},
            {"composition_id": "a", "performance_id": "p1", "split": "valid"},
            {"composition_id": "b", "performance_id": "p2", "split": "train"},
        ]
    )
    assert leakage == {("a", "p1"): {"train", "valid"}}

