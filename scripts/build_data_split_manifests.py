#!/usr/bin/env python3
"""Build sqr_paper_split_{train,val,test}.jsonl from local data/ tree.

Layout convention (release dirs like "2-38-op76-n2"):
  train: /ceph/leon/MSS/data/train/stringqud_reverb_two_seating/2-*  (59 x type2 only,
         matching E1cosine "pure type2, no type1 contamination")
  val:   /ceph/leon/MSS/data/val/stringqud_reverb_two_seating/*      (6 works x 2 seatings)
  test:  /ceph/leon/MSS/data/test/URMP_test_set/*                    (6 works x 2 seatings)

Stem files are named "<seat>_<Part>.wav" where seat in 1..4 is the PHYSICAL
seating position (position_id = seat - 1). The part sitting at each seat
differs between seating types, e.g.:
  1-1-aasesdeath (type1): 1_Violin_2 2_Viola 3_Cello 4_Violin_1  -> pos0..3 = V2,Va,C,V1
  2-1-aasesdeath (type2): 1_Cello 2_Viola 3_Violin_2 4_Violin_1  -> pos0..3 = C,Va,V2,V1
which matches the B1/quartet-data manifests' type2 layout
{violin1:3, violin2:2, viola:1, cello:0}. Files are therefore located by
globbing the part name, and position_ids derived from the filename prefix.
"""

import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf

DATA_ROOT = Path("/ceph/leon/MSS/data")
OUT_DIR = DATA_ROOT / "manifests"

PART_TO_SUFFIX = {
    "violin1": "_Violin_1.wav",
    "violin2": "_Violin_2.wav",
    "viola": "_Viola.wav",
    "cello": "_Cello.wav",
}
TIMBRE_IDS = {"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}
TIMBRE_IDS = {"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}
SEATING_POSITION_MAP = {
    1: {"pos0": "violin1", "pos1": "viola", "pos2": "cello", "pos3": "violin2"},
    2: {"pos0": "violin1", "pos1": "violin2", "pos2": "viola", "pos3": "cello"},
}
POSITION_IDS = {
    t: {part: int(pos[3:]) for pos, part in m.items()}
    for t, m in SEATING_POSITION_MAP.items()
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record(item_dir, rel_parent, split, performance_id, source_dataset):
    item_id = item_dir.name
    seating_type = int(item_id.split("-")[0])
    composition_id = "-".join(item_id.split("-")[1:])
    mix_files = sorted(item_dir.glob("mix_*.wav"))
    assert len(mix_files) == 1, (
        f"{item_id}: expected 1 mix file, found {len(mix_files)}"
    )
    mixture = mix_files[0]

    audio_paths = {"mixture": mixture}
    position_ids = {}
    for part, suffix in PART_TO_SUFFIX.items():
        matches = [
            p for p in item_dir.glob(f"*{suffix}") if not p.name.startswith("mix_")
        ]
        assert len(matches) == 1, (
            f"{item_id}: expected 1 file for {part}, found {len(matches)}"
        )
        path = matches[0]
        seat = int(path.name.split("_")[0])
        assert 1 <= seat <= 4, f"{item_id}: bad seat prefix in {path.name}"
        position_ids[part] = seat - 1
        audio_paths[part] = path
    assert sorted(position_ids.values()) == [0, 1, 2, 3], (
        f"{item_id}: positions {position_ids}"
    )

    infos = {key: sf.info(str(path)) for key, path in audio_paths.items()}
    rates = {info.samplerate for info in infos.values()}
    chans = {info.channels for info in infos.values()}
    frames = {info.frames for info in infos.values()}
    assert rates == {44100}, f"{item_id}: sample rates {rates}"
    assert chans == {2}, f"{item_id}: channels {chans}"
    assert len(frames) == 1, f"{item_id}: inconsistent lengths {frames}"
    num_samples = frames.pop()

    return {
        "item_id": item_id,
        "composition_id": composition_id,
        "performance_id": performance_id,
        "work_id": composition_id,
        "movement_id": "full",
        "chunk_id": item_id,
        "split": split,
        "layout_id": f"type{seating_type}",
        "seating_type": seating_type,
        "is_fixture": False,
        "mixture": str(Path(rel_parent) / item_id / mixture.name),
        "stems": {
            part: str(Path(rel_parent) / item_id / audio_paths[part].name)
            for part in PART_TO_SUFFIX
        },
        "position_ids": position_ids,
        "timbre_ids": dict(TIMBRE_IDS),
        "sample_rate": 44100,
        "channels": 2,
        "num_samples": num_samples,
        "duration_seconds": num_samples / 44100.0,
        "mixture_hash": sha256(mixture),
        "stem_hashes": {part: sha256(audio_paths[part]) for part in PART_TO_SUFFIX},
        "hash_method": "SHA-256 of complete WAV file bytes",
        "source_dataset": source_dataset,
    }


def collect(
    src_dir, rel_parent, split, performance_id, source_dataset, name_filter=None
):
    records = []
    for item_dir in sorted(src_dir.iterdir()):
        if not item_dir.is_dir() or item_dir.name.startswith("__"):
            continue
        if name_filter and not name_filter(item_dir.name):
            continue
        records.append(
            build_record(item_dir, rel_parent, split, performance_id, source_dataset)
        )
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_records = collect(
        DATA_ROOT / "train/stringqud_reverb_two_seating",
        "train/stringqud_reverb_two_seating",
        "train",
        "stringqud_reverb_two_seating",
        "StringQuD reverb two-seating release (local data/train)",
        name_filter=lambda name: name.startswith("2-"),
    )
    val_records = collect(
        DATA_ROOT / "val/stringqud_reverb_two_seating",
        "val/stringqud_reverb_two_seating",
        "val",
        "stringqud_reverb_two_seating",
        "StringQuD reverb two-seating release (local data/val)",
    )
    test_records = collect(
        DATA_ROOT / "test/URMP_test_set",
        "test/URMP_test_set",
        "test",
        "urmp_test_set",
        "URMP test set release (local data/test)",
    )

    outputs = {
        "sqr_paper_split_train.jsonl": train_records,
        "sqr_paper_split_val.jsonl": val_records,
        "sqr_paper_split_test.jsonl": test_records,
    }
    for fname, records in outputs.items():
        out_path = OUT_DIR / fname
        with open(out_path, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        n_type1 = sum(1 for r in records if r["layout_id"] == "type1")
        n_type2 = sum(1 for r in records if r["layout_id"] == "type2")
        works = sorted({r["work_id"] for r in records})
        print(
            f"{fname}: {len(records)} records ({n_type1} type1, {n_type2} type2), {len(works)} works"
        )

    split_seen = {}
    for fname, records in outputs.items():
        for r in records:
            split_seen.setdefault(r["work_id"], set()).add(fname)
    leaks = {w: s for w, s in split_seen.items() if len(s) > 1}
    assert not leaks, f"work-level split leakage: {leaks}"
    print("No work-level leakage across splits.")


if __name__ == "__main__":
    sys.exit(main())
