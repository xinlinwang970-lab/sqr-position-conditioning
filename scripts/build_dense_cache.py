#!/usr/bin/env python3
"""Build unnormalized dense-crop NPZ files using an explicit split selection.

This restores the original whole-record cache format, with portable paths and
input checks. It does not establish byte identity with the unavailable original
baseline cache. Existing cache destinations are never overwritten.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from sqr_repro.data import PARTS
from sqr_repro.release_paths import validate_manifest

CROP_SAMPLES = 179200


def build_cache(data_root, output_root, manifests):
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Use a new cache directory: {output_root}")
    selected = {split: validate_manifest(path, data_root, split=split,
                                         min_samples=CROP_SAMPLES if split == "train" else None)
                for split, path in manifests.items()}
    # Cache item IDs become filenames; fail before writing anything.
    for records in selected.values():
        for record in records:
            item = record["item_id"]
            if item in {".", ".."} or Path(item).name != item or "/" in item or "\\" in item:
                raise ValueError(f"Unsafe cache item_id: {item}")
    index = {"crop_samples": CROP_SAMPLES, "sample_rate": 44100, "parts": list(PARTS), "records": [],
             "normalization": "none; input WAV float samples are preserved",
             "manifest_sha256": {split: hashlib.sha256(Path(path).read_bytes()).hexdigest() for split, path in manifests.items()}}
    for split, records in selected.items():
        directory = output_root / split
        directory.mkdir(parents=True, exist_ok=True)
        for record in records:
            mixture, _ = sf.read(data_root / record["mixture"], dtype="float32", always_2d=True)
            waves = {part: sf.read(data_root / record["stems"][part], dtype="float32", always_2d=True)[0].T.copy() for part in PARTS}
            if not np.isfinite(mixture).all() or any(not np.isfinite(wave).all() for wave in waves.values()):
                raise ValueError(f"Non-finite waveform: {record['item_id']}")
            n = len(mixture)
            item = record["item_id"]
            destination = directory / f"{item}.npz"
            pos = record["position_ids"]
            np.savez_compressed(destination, mixture=mixture.T.copy(), **waves,
                position_ids=np.array([pos[part] for part in PARTS], dtype=np.int32),
                sample_rate=np.int64(44100), crop_samples=np.int64(CROP_SAMPLES), num_segments=np.int64(n // CROP_SAMPLES),
                item_id=item, composition_id=record["composition_id"], layout_id=record["layout_id"], split=split)
            index["records"].append({"item_id": item, "composition_id": record["composition_id"],
                "layout_id": record["layout_id"], "split": split, "npz_path": destination.relative_to(output_root).as_posix(),
                "num_samples": n, "num_segments": n // CROP_SAMPLES, "position_ids": pos,
                "npz_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
            print(f"Cached {split}/{item}", flush=True)
    (output_root / "cache_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-manifest", type=Path, required=True)
    args = parser.parse_args()
    index = build_cache(args.data_root, args.output_root,
                        {split: getattr(args, f"{split}_manifest") for split in ("train", "val", "test")})
    print(json.dumps({"records": len(index["records"]), "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
