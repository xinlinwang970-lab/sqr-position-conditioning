import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wav_item(root, item, *, frames=2048):
    builder = script("build_data_split_manifests")
    directory = root / "extracted" / item
    directory.mkdir(parents=True)
    layout = int(item.split("-")[0])
    waves = []
    suffixes = {"violin1": "Violin_1", "violin2": "Violin_2", "viola": "Viola", "cello": "Cello"}
    for index, (part, seat) in enumerate(builder.POSITION_IDS[layout].items()):
        wave = np.full((frames, 2), 0.005 * (index + 1), dtype=np.float32)
        sf.write(directory / f"{seat + 1}_{suffixes[part]}.wav", wave, 44100, subtype="FLOAT")
        waves.append(wave)
    sf.write(directory / f"mix_{item[2:]}.wav", sum(waves), 44100, subtype="FLOAT")
    return directory


def selection_fixture(tmp_path):
    root, selection = tmp_path / "audio", tmp_path / "selection"
    selection.mkdir()
    builder = script("build_data_split_manifests")
    records = {}
    for split, item in {"train": "2-1-training", "val": "1-8-CaraSposa", "test": "2-24-Pirates"}.items():
        directory = wav_item(root, item, frames=179264 if split == "train" else 2048)
        record = builder.build_record(directory, "extracted", split, "fixture", "test audio")
        (selection / f"sqr_paper_split_{split}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        records[split] = record
    return root, selection, records


def test_rebuild_selects_val_from_shared_extraction_without_moving_audio(tmp_path):
    builder = script("build_data_split_manifests")
    root, selection, records = selection_fixture(tmp_path)
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.wav"))
    report = builder.rebuild(root, selection, tmp_path / "manifests")
    assert report["audio_matches_frozen_manifests"] is True
    assert report["records"] == {"train": 1, "val": 1, "test": 1}
    val = json.loads((tmp_path / "manifests/sqr_paper_split_val.jsonl").read_text())
    assert val["position_ids"] == {"violin1": 3, "violin2": 0, "viola": 1, "cello": 2}
    assert val["mixture"].startswith("extracted/")
    assert before == sorted(path.relative_to(root).as_posix() for path in root.rglob("*.wav"))


def test_rebuild_fails_on_changed_audio_and_never_overwrites(tmp_path):
    builder = script("build_data_split_manifests")
    root, selection, records = selection_fixture(tmp_path)
    record = records["train"]
    sf.write(root / record["mixture"], np.zeros((record["num_samples"], 2), np.float32), 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="differs from frozen"):
        builder.rebuild(root, selection, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    report = builder.rebuild(root, selection, tmp_path / "out", allow_hash_differences=True)
    assert report["audio_matches_frozen_manifests"] is False
    with pytest.raises(FileExistsError):
        builder.rebuild(root, selection, tmp_path / "out", allow_hash_differences=True)


def test_cache_and_real_loader_keep_waveforms_targets_and_alignment(tmp_path):
    torch = pytest.importorskip("torch")
    from sqr_repro.paper_c0.dataset import PaperC0WavDataset

    root, selection, records = selection_fixture(tmp_path)
    cache = script("build_dense_cache").build_cache(root, tmp_path / "cache",
        {split: selection / f"sqr_paper_split_{split}.jsonl" for split in records})
    assert cache["normalization"].startswith("none")
    dataset = PaperC0WavDataset(selection / "sqr_paper_split_train.jsonl", root, "train",
                               crop_samples=179200, all_targets=True, require_fixture=False)
    item = dataset[0]
    assert item["position_id"] == 3
    assert item["mixture"].shape == (2, 179200)
    assert torch.allclose(item["mixture"], sum(item["all_targets"].values()), atol=1e-7)
    meta = next(r for r in cache["records"] if r["split"] == "train")
    with np.load(tmp_path / "cache" / meta["npz_path"]) as stored:
        original = sf.read(root / records["train"]["stems"]["viola"], dtype="float32", always_2d=True)[0]
        assert np.array_equal(stored["viola"], original.T)
        assert int(stored["num_segments"]) == 1


def test_preflight_rejects_position_labels_inconsistent_with_declared_layout(tmp_path):
    from sqr_repro.release_paths import validate_manifest
    root, selection, records = selection_fixture(tmp_path)
    record = records["train"]
    record["position_ids"]["violin2"] = 0
    file = selection / "bad_positions.jsonl"
    file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Position labels disagree"):
        validate_manifest(file, root)


def test_legacy_validation_alias_is_accepted_only_for_validation(tmp_path):
    from sqr_repro.release_paths import validate_manifest
    root, selection, records = selection_fixture(tmp_path)
    record = records["val"]
    record["split"] = "validation"
    file = selection / "legacy_val.jsonl"
    file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert len(validate_manifest(file, root, split="val")) == 1
    with pytest.raises(ValueError, match="Unexpected split"):
        validate_manifest(file, root, split="test")


def test_saved_scores_are_recomputable_and_e3_checkpoint_is_unambiguous():
    verify = script("verify_results")
    report = verify.verify(REPO)
    assert report["result_files"] == 11
    assert report["queries"] == 528
    ep275 = report["scores"]["results/checkpoint_evals/E3_test_ep274.json"]
    final = report["scores"]["results/test_results/PureDrop_E3_300ep_p025_test_results.json"]
    assert ep275["type2_median"] == pytest.approx(1.631005526, abs=1e-8)
    assert final["type2_median"] == pytest.approx(1.686057448, abs=1e-8)


def test_duplicate_queries_are_rejected():
    verify = script("verify_results")
    row = {"composition_id": "a", "layout_id": "type1", "part_id": "viola", "sdr": 1.0}
    with pytest.raises(ValueError, match="Duplicate"):
        verify.recompute([row, row])
