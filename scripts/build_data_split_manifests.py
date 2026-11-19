#!/usr/bin/env python3
"""Rebuild frozen SQR split manifests from a shared extracted audio tree.

Item directories may be anywhere below --data-root; a physical val directory
is not required. Seat prefixes are 1-based, target identities come from part
names, and position IDs are prefix minus one. No WAV is moved or normalized.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]

PART_TO_SUFFIX = {
    "violin1": "_Violin_1.wav",
    "violin2": "_Violin_2.wav",
    "viola": "_Viola.wav",
    "cello": "_Cello.wav",
}
TIMBRE_IDS = {"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}
POSITION_IDS = {
    1: {"violin1": 3, "violin2": 0, "viola": 1, "cello": 2},
    2: {"violin1": 3, "violin2": 2, "viola": 1, "cello": 0},
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
    assert position_ids == POSITION_IDS[seating_type], (
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
        "mixture": (Path(rel_parent) / item_id / mixture.name).as_posix(),
        "stems": {
            part: (Path(rel_parent) / item_id / audio_paths[part].name).as_posix()
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


def rebuild(data_root, selection_dir, output_dir, *, allow_hash_differences=False):
    """Select frozen IDs without requiring a physically separated val directory.

    The two layouts may be anywhere under data_root. Ambiguous/missing item
    directories fail closed. Audio is read and hashed, never moved or normalized.
    """
    data_root, selection_dir, output_dir = map(Path, (data_root, selection_dir, output_dir))
    templates = {}
    for split in ("train", "val", "test"):
        file = selection_dir / f"sqr_paper_split_{split}.jsonl"
        templates[split] = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not templates[split] or len({r["item_id"] for r in templates[split]}) != len(templates[split]):
            raise ValueError(f"Empty or duplicate selection: {file}")
    selected = {r["item_id"] for rows in templates.values() for r in rows}
    locations = {key: [] for key in selected}
    for candidate in data_root.rglob("*"):
        if candidate.name in locations and candidate.is_dir():
            locations[candidate.name].append(candidate)
    outputs, differences = {}, []
    identities = {}
    for split, rows in templates.items():
        records = []
        for template in rows:
            item = template["item_id"]
            found = locations[item]
            if len(found) != 1:
                raise ValueError(f"{item}: expected one directory under data-root, found {len(found)}")
            directory = found[0]
            record = build_record(directory, directory.parent.relative_to(data_root), split,
                                  template["performance_id"], template.get("source_dataset", "SQR release"))
            for key in ("mixture_hash", "stem_hashes", "num_samples", "position_ids"):
                if key in template and record[key] != template[key]:
                    differences.append({"item_id": item, "field": key, "expected": template[key], "actual": record[key]})
            previous = identities.setdefault(record["composition_id"], split)
            if previous != split:
                raise ValueError(f"Composition-id overlap between {previous} and {split}: {record['composition_id']}")
            records.append(record)
        outputs[split] = records
    if differences and not allow_hash_differences:
        raise ValueError(f"Audio differs from frozen manifests ({len(differences)} fields). Use --allow-hash-differences only for a separately reported dataset variant.")
    # Fail before writing if any destination already exists.
    destinations = [output_dir / f"sqr_paper_split_{split}.jsonl" for split in outputs]
    destinations.append(output_dir / "build_report.json")
    for file in destinations:
        if file.exists():
            raise FileExistsError(f"Use a new output directory; file exists: {file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in outputs.items():
        (output_dir / f"sqr_paper_split_{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    report = {"records": {split: len(rows) for split, rows in outputs.items()},
              "audio_matches_frozen_manifests": not differences, "differences": differences,
              "selection": "frozen composition IDs; this is not a musicological overlap audit"}
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Rebuild portable manifests from extracted SQR WAVs without moving audio")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, default=REPO_ROOT / "manifests")
    parser.add_argument("--allow-hash-differences", action="store_true")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.data_root.resolve(), args.selection_dir, args.output_dir,
                             allow_hash_differences=args.allow_hash_differences), indent=2))


if __name__ == "__main__":
    sys.exit(main())
