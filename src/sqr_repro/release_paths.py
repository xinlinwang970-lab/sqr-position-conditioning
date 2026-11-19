"""Portable paths and fail-fast input checks for the historical entry points.

This module changes setup only. The legacy training and evaluation mathematics
remain in the corresponding experiment scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf

from .config import load_config
from .data import LAYOUTS, PARTS, validate_manifest_record


def _path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def prepare_training(default_config, *, repo_root, baseline=False, custom=False, argv=None):
    repo = Path(repo_root).resolve()
    parser = argparse.ArgumentParser(
        description="Historical experiment; path overrides do not change its numerical protocol."
    )
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--data-root", type=Path, required=custom)
    parser.add_argument("--val-data-root", type=Path, required=custom)
    parser.add_argument("--test-data-root", type=Path, required=custom)
    parser.add_argument("--run-root", type=Path)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-manifest", type=Path, required=custom and split != "test")
    if baseline:
        parser.add_argument("--cache-root", type=Path, required=True,
                            help="Explicit dense cache; original cache/audio identity is not bundled.")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate manifests and WAV headers without building a model or training.")
    args = parser.parse_args(argv)
    # Explicit command-line paths are relative to the caller. Bundled defaults
    # are relative to the repository, including when invoked from another cwd.
    config_path = args.config.resolve()
    config = load_config(config_path)
    config["data"]["root"] = str(args.data_root.resolve() if args.data_root else _path(config["data"]["root"], repo))
    config["paths"]["run_root"] = str(args.run_root.resolve() if args.run_root else _path(config["paths"]["run_root"], repo))
    args.data_root = Path(config["data"]["root"])
    args.val_data_root = args.val_data_root.resolve() if args.val_data_root else args.data_root
    args.test_data_root = args.test_data_root.resolve() if args.test_data_root else args.data_root
    defaults = {"train": "sqr_paper_split_train.jsonl", "val": "sqr_paper_split_val.jsonl", "test": "sqr_paper_split_test.jsonl"}
    if baseline:
        defaults = {"train": "historical_baseline/train.jsonl", "val": "historical_baseline/val.jsonl", "test": "historical_baseline/test.jsonl"}
        args.cache_root = args.cache_root.resolve()
    for split, default in defaults.items():
        supplied = getattr(args, f"{split}_manifest")
        setattr(args, f"{split}_manifest", supplied.resolve() if supplied else repo / "manifests" / default)
    config["data"]["manifest"] = str(args.train_manifest)
    args.config = config_path
    return config, args


def read_records(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def validate_manifest(path, root, *, split=None, min_samples=None):
    records = read_records(path)
    seen = set()
    for record in records:
        if any(not str(record.get(key, "")).strip() for key in ("item_id", "composition_id", "performance_id")):
            raise ValueError("Manifest IDs must be nonempty")
        if record["item_id"] in seen:
            raise ValueError(f"Duplicate item: {record['item_id']}")
        seen.add(record["item_id"])
        accepted_splits = {"val", "validation"} if split == "val" else {split}
        if split is not None and record.get("split") not in accepted_splits:
            raise ValueError(f"Unexpected split for {record['item_id']}: expected {split}")
        validate_manifest_record(record, root, require_audio=True)
        if record["layout_id"] not in LAYOUTS or record["position_ids"] != LAYOUTS[record["layout_id"]]:
            raise ValueError(f"Position labels disagree with the declared layout: {record['item_id']}")
        if (int(record["sample_rate"]), int(record["channels"])) != (44100, 2):
            raise ValueError("This release expects 44100 Hz stereo WAV audio")
        if min_samples is not None and int(record["num_samples"]) < min_samples:
            raise ValueError(f"Training record too short for one crop: {record['item_id']}")
        for relative in (record["mixture"], *record["stems"].values()):
            file = Path(root) / relative
            info = sf.info(str(file))
            if (info.samplerate, info.channels, info.frames) != (44100, 2, int(record["num_samples"])):
                raise ValueError(f"WAV header differs from manifest: {file}")
    return records


def validate_inputs(config, args, *, expected_train_records=None, baseline=False):
    train = read_records(args.train_manifest)
    if expected_train_records is not None and len(train) != expected_train_records:
        raise ValueError(f"Historical protocol requires {expected_train_records} training records; got {len(train)}")
    if expected_train_records == 59 and any(r.get("layout_id") != "type2" for r in train):
        raise ValueError("Historical SQR 59-record protocol requires type2-only training")
    crop = int(config["audio"]["segment_samples"])
    splits = {}
    for split in ("train", "val", "test"):
        root = args.data_root if split == "train" else getattr(args, f"{split}_data_root")
        records = validate_manifest(getattr(args, f"{split}_manifest"), root, split=split,
                                    min_samples=crop if split == "train" else None)
        expected = {"val": 10 if baseline else 12, "test": 12}.get(split)
        if expected_train_records is not None and expected is not None and len(records) != expected:
            raise ValueError(f"Historical protocol requires {expected} {split} records; got {len(records)}")
        splits[split] = records
    train_works = {r["composition_id"] for r in splits["train"]}
    val_works = {r["composition_id"] for r in splits["val"]}
    if train_works & val_works:
        raise ValueError(f"Train/val composition-id overlap: {sorted(train_works & val_works)}")
    report = {"status": "INPUT_HEADERS_VALID", "records": {key: len(value) for key, value in splits.items()},
              "data_root": str(args.data_root), "val_data_root": str(args.val_data_root),
              "test_data_root": str(args.test_data_root), "crop_samples": crop,
              "checks": "manifest schema, IDs, paths, headers; no full-waveform or acoustic QA"}
    if baseline:
        index = json.loads((args.cache_root / "cache_index.json").read_text(encoding="utf-8"))
        if (index["sample_rate"], index["crop_samples"]) != (44100, crop):
            raise ValueError("Dense cache sample rate/crop does not match config")
        cached = [r for r in index["records"] if r["split"] == "train"]
        if len(cached) != len(train) or {r["item_id"] for r in cached} != {r["item_id"] for r in train}:
            raise ValueError("Dense cache train IDs differ from selected manifest")
        for record in cached:
            if int(record["num_segments"]) < 1:
                raise ValueError(f"No complete training slab in cache: {record['item_id']}")
            if not (args.cache_root / record["npz_path"]).is_file():
                raise FileNotFoundError(args.cache_root / record["npz_path"])
        report["dense_cache"] = str(args.cache_root)
    print(json.dumps(report, indent=2))
    return report


def write_launch_record(config, args, *, script, runtime_checkpointing, batch_size, seed, target_epochs):
    """Save effective overrides without overwriting historical result files."""
    script = Path(script)
    payload = {"config": config, "entrypoint": script.name,
               "entrypoint_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
               "actual_script_settings": {"activation_checkpoint_wrappers": runtime_checkpointing,
                   "batch_size": batch_size, "seed": seed, "target_epochs": target_epochs,
                   "resume_restores_rng": False, "inference": "legacy_hann_ola"},
               "paths": {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)}}
    out = Path(config["paths"]["run_root"]) / "release_launch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
