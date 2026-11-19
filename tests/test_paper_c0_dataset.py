import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import soundfile as sf
import numpy as np

from sqr_repro.audio import AudioConfig, AudioTransform
from sqr_repro.fixtures import make_fixtures
from sqr_repro.paper_c0.dataset import PaperC0WavDataset, adapt_manifest_identity


def build_two_split_fixture(tmp_path: Path) -> Path:
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_manifest = make_fixtures(train_dir, seed=10, sample_rate=8000, seconds=0.6)
    valid_manifest = make_fixtures(valid_dir, seed=20, sample_rate=8000, seconds=0.6)
    train = json.loads(train_manifest.read_text(encoding="utf-8"))
    valid = json.loads(valid_manifest.read_text(encoding="utf-8"))
    train.update({"item_id": "fixture-train", "composition_id": "fixture-work-train", "performance_id": "take-train", "split": "train"})
    valid.update({"item_id": "fixture-valid", "composition_id": "fixture-work-valid", "performance_id": "take-valid", "split": "validation", "layout_id": "type2"})
    # The fixture's type-2 record uses the explicitly separate fixture layout.
    valid["position_ids"] = {"violin1": 3, "violin2": 2, "viola": 1, "cello": 0}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(train) + "\n" + json.dumps(valid) + "\n", encoding="utf-8")
    return manifest


def test_four_queries_share_mixture_and_crop(tmp_path):
    manifest = build_two_split_fixture(tmp_path)
    dataset = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=1024, crop_seed=7, require_fixture=True)
    samples = [dataset[i] for i in range(4)]
    assert {sample["item_id"] for sample in samples} == {"fixture-train"}
    assert len({sample["crop_offset"] for sample in samples}) == 1
    assert all(torch.equal(sample["mixture"], samples[0]["mixture"]) for sample in samples)
    assert [sample["target_source_index"] for sample in samples] == [0, 1, 2, 3]
    assert [sample["timbre_id"] for sample in samples] == [0, 0, 1, 2]
    assert [sample["position_id"] for sample in samples] == [3, 0, 1, 2]


def test_split_filter_and_deterministic_crop(tmp_path):
    manifest = build_two_split_fixture(tmp_path)
    first = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=1024, crop_seed=99, require_fixture=True)
    second = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=1024, crop_seed=99, require_fixture=True)
    assert first[0]["crop_offset"] == second[0]["crop_offset"]
    assert torch.equal(first[0]["mixture"], second[0]["mixture"])
    valid = PaperC0WavDataset(manifest, tmp_path / "valid", "validation", crop_samples=1024, crop_seed=99, require_fixture=True)
    assert {valid[i]["split"] for i in range(len(valid))} == {"validation"}


def test_epoch_and_crop_slot_change_window_but_same_identity_repeats(tmp_path):
    manifest = build_two_split_fixture(tmp_path)
    a = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=1024, crop_seed=99, epoch=0, crop_slot=0, require_fixture=True)
    b = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=1024, crop_seed=99, epoch=1, crop_slot=0, require_fixture=True)
    c = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=1024, crop_seed=99, epoch=0, crop_slot=1, require_fixture=True)
    repeat = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=1024, crop_seed=99, epoch=0, crop_slot=0, require_fixture=True)
    assert a[0]["crop_offset"] == repeat[0]["crop_offset"]
    assert (a[0]["crop_offset"], a[4 - 1]["crop_offset"]) == (a[0]["crop_offset"], a[3]["crop_offset"])
    assert b[0]["crop_offset"] != a[0]["crop_offset"] or c[0]["crop_offset"] != a[0]["crop_offset"]


def test_windowed_io_reads_only_mixture_and_target_by_default(tmp_path):
    manifest = build_two_split_fixture(tmp_path)
    dataset = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=512, crop_seed=3, require_fixture=True)
    sample = dataset[0]
    assert len(dataset.last_read_log) == 2
    assert {entry["frames"] for entry in dataset.last_read_log} == {512}
    assert "all_targets" not in sample
    diagnostic = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=512, crop_seed=3, all_targets=True, require_fixture=True)
    diagnostic[0]
    assert len(diagnostic.last_read_log) == 5


def test_windowed_values_match_full_file_slice_and_identity_adapter(tmp_path):
    manifest = build_two_split_fixture(tmp_path)
    dataset = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=512, crop_seed=3, require_fixture=True)
    sample = dataset[2]
    record = dataset.records[0]
    full, _ = sf.read(str(tmp_path / "train" / record["stems"]["viola"]), dtype="float32", always_2d=True)
    start = sample["crop_offset"]
    assert np.array_equal(sample["target"].numpy(), full[start : start + 512].T)
    identity = adapt_manifest_identity(record)
    assert identity["source_index_by_part"]["viola"] == 2
    assert identity["position_ids"]["viola"] != identity["source_index_by_part"]["viola"]


def test_stft_shapes_and_shared_time_coordinates(tmp_path):
    manifest = build_two_split_fixture(tmp_path)
    dataset = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=3968, crop_seed=1, require_fixture=True)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    transform = AudioTransform(AudioConfig(sample_rate=8000, n_fft=256, win_length=256, hop_length=128, num_frames=32))
    mixture_spec = transform.stft(batch["mixture"])
    target_spec = transform.stft(batch["target"])
    assert mixture_spec.shape == target_spec.shape
    assert mixture_spec.shape[:2] == (4, 4)
    assert mixture_spec.shape[-1] == 129
    assert mixture_spec.shape[2] == 32


def test_full_stft_contract_44100_and_roundtrip_channel_identity():
    transform = AudioTransform(AudioConfig(sample_rate=44100, n_fft=2048, win_length=2048, hop_length=1024, num_frames=32))
    waveform = torch.zeros(1, 2, (32 - 1) * 1024)
    waveform[0, 0, 1000] = 1.0
    waveform[0, 1, 2000] = -0.5
    packed = transform.stft(waveform)
    assert packed.shape == (1, 4, 32, 1025)
    restored = transform.istft(packed, waveform.shape[-1])
    assert torch.max(torch.abs(restored - waveform)) < 1e-4
    assert torch.argmax(torch.abs(restored[0, 0])) == 1000
    assert torch.argmax(torch.abs(restored[0, 1])) == 2000


def test_mult_worker_reproduces_same_crop_identity(tmp_path):
    manifest = build_two_split_fixture(tmp_path)
    dataset = PaperC0WavDataset(manifest, tmp_path / "train", "train", crop_samples=512, crop_seed=17, require_fixture=True)
    single = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    multi = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)))
    assert torch.equal(single["mixture"], multi["mixture"])
    assert torch.equal(single["target"], multi["target"])
