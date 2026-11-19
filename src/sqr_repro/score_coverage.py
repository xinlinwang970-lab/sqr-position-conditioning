from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .audio import AudioConfig
from .score import events_to_piano_roll, score_hash
from .score_toy import ScoreMixture, ScoreQuery, _pattern_events, _render_events
from .toy import array_hash, stable_toy_seed


TRAIN_PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5))
COMPOSITIONAL_PAIRS = ((0, 5), (1, 4), (2, 3))


def generate_p2b_corpus(config: dict) -> dict[str, list[ScoreMixture]]:
    coverage = config["score_coverage"]
    train_pairs = tuple(tuple(sorted(pair)) for pair in coverage["train_pairs"])
    compositional_pairs = tuple(tuple(sorted(pair)) for pair in coverage["compositional_pairs"])
    audio = AudioConfig.from_dict(config["audio"])
    base_seed = int(config["experiment"]["seed"])
    splits = {"train": [], "id_probe": [], "compositional_ood": []}
    for split, pairs, count in (
        ("train", train_pairs, int(coverage["train_mixtures_per_pair"])),
        ("id_probe", train_pairs, int(coverage["id_probe_mixtures_per_pair"])),
        ("compositional_ood", compositional_pairs, int(coverage["compositional_mixtures_per_pair"])),
    ):
        for pair in pairs:
            for index in range(count):
                seed = stable_toy_seed(base_seed, "P2B", split, pair, index)
                order = (0, 1) if index % 2 == 0 else (1, 0)
                splits[split].append(_make_mixture(config, audio, split, pair, index, seed, order))
    return splits


def validate_p2b_corpus(splits: dict[str, list[ScoreMixture]], config: dict) -> dict:
    errors: list[str] = []
    coverage = config["score_coverage"]
    train_pairs = {tuple(sorted(pair)) for pair in coverage["train_pairs"]}
    comp_pairs = {tuple(sorted(pair)) for pair in coverage["compositional_pairs"]}
    seen_train_primitives = set(value for pair in train_pairs for value in pair)
    if seen_train_primitives != set(range(6)):
        errors.append(f"train primitives are not exactly P0-P5: {sorted(seen_train_primitives)}")
    if train_pairs != set(TRAIN_PAIRS):
        errors.append(f"train pair set differs from preregistered set: {sorted(train_pairs)}")
    if comp_pairs != set(COMPOSITIONAL_PAIRS):
        errors.append(f"compositional pair set differs from preregistered set: {sorted(comp_pairs)}")
    if train_pairs & comp_pairs:
        errors.append("train/compositional pair overlap")
    hashes: dict[str, set[str]] = {}
    source_hashes: dict[str, set[str]] = {}
    pair_rows = []
    balance_rows = []
    for split, mixtures in splits.items():
        counts_by_pair: dict[tuple[int, int], int] = {}
        order_counts: dict[tuple[int, int], list[int]] = {}
        for mixture in mixtures:
            pair = tuple(sorted(mixture.pattern_pair))
            counts_by_pair[pair] = counts_by_pair.get(pair, 0) + 1
            order = 0 if mixture.pattern_pair == pair else 1
            order_counts.setdefault(pair, [0, 0])[order] += 1
            hashes.setdefault(mixture.mixture_hash, set()).add(split)
            for source in mixture.sources:
                source_hashes.setdefault(array_hash(source), set()).add(split)
            if not np.allclose(mixture.mixture, mixture.sources.sum(axis=0), atol=1e-7, rtol=1e-7):
                errors.append(f"{mixture.mixture_id}: mixture sum mismatch")
            for query in mixture.queries:
                if query.timbre_id != 0 or query.position_id != 0:
                    errors.append(f"{query.query_id}: condition IDs changed")
        for pair, count in sorted(counts_by_pair.items()):
            pair_rows.append({"split": split, "pair_family": f"pair_{pair[0]}_{pair[1]}", "primitive_a": pair[0], "primitive_b": pair[1], "mixture_count": count, "query_count": count * 2})
        for pair, counts in sorted(order_counts.items()):
            balance_rows.append({"split": split, "pair_family": f"pair_{pair[0]}_{pair[1]}", "source_order_ab_count": counts[0], "source_order_ba_count": counts[1], "balanced": counts[0] == counts[1]})
            if counts[0] != counts[1]:
                errors.append(f"{split} {pair}: source order is not balanced")
    mixture_leakage = {key: value for key, value in hashes.items() if len(value) > 1}
    source_leakage = {key: value for key, value in source_hashes.items() if len(value) > 1}
    if mixture_leakage:
        errors.append("mixture hash leakage across P2B splits")
    if source_leakage:
        errors.append("source hash leakage across P2B splits")
    id_pairs = {tuple(sorted(mixture.pattern_pair)) for mixture in splits["id_probe"]}
    comp_pairs = {tuple(sorted(mixture.pattern_pair)) for mixture in splits["compositional_ood"]}
    if not id_pairs <= train_pairs:
        errors.append("ID probe contains unseen pair family")
    if not all(set(pair) <= seen_train_primitives for pair in comp_pairs):
        errors.append("compositional OOD contains unseen primitive")
    return {"ok": not errors, "errors": errors, "train_mixtures": len(splits["train"]), "id_probe_mixtures": len(splits["id_probe"]), "compositional_ood_mixtures": len(splits["compositional_ood"]), "train_queries": len(splits["train"]) * 2, "id_probe_queries": len(splits["id_probe"]) * 2, "compositional_ood_queries": len(splits["compositional_ood"]) * 2, "mixture_hash_leakage_count": len(mixture_leakage), "source_hash_leakage_count": len(source_leakage), "pair_rows": pair_rows, "balance_rows": balance_rows}


def write_p2b_manifests(splits: dict[str, list[ScoreMixture]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, mixtures in splits.items():
        lines = [json.dumps(_manifest_record(mixture), sort_keys=True) for mixture in mixtures]
        (out / f"{split}_manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest_record(mixture: ScoreMixture) -> dict:
    pair = tuple(sorted(mixture.pattern_pair))
    return {"mixture_id": mixture.mixture_id, "split": mixture.split, "seed": mixture.seed, "mixture_hash": mixture.mixture_hash, "source_hashes": [array_hash(source) for source in mixture.sources], "pair_family": f"pair_{pair[0]}_{pair[1]}", "source_order": list(mixture.pattern_pair), "queries": [{"query_id": query.query_id, "target_source_index": query.target_source_index, "timbre_id": query.timbre_id, "position_id": query.position_id, "primitive_score_id": score_hash(query.score), "score_content_hash": score_hash(query.score)} for query in mixture.queries]}


def _make_mixture(config, audio, split, pair, index, seed, order) -> ScoreMixture:
    rng = np.random.default_rng(seed)
    source_patterns = (pair[order[0]], pair[order[1]])
    events = [_pattern_events(pattern, audio) for pattern in source_patterns]
    scores = [events_to_piano_roll(item, num_frames=audio.num_frames, sample_rate=audio.sample_rate, hop_length=audio.hop_length, center=audio.center) for item in events]
    sources = np.stack([_render_events(item, audio.sample_rate, audio.segment_samples, phase=float(rng.uniform(0, 2 * np.pi))) for item in events]).astype(np.float32)
    mixture = sources.sum(axis=0).astype(np.float32)
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.9:
        sources *= 0.9 / peak
        mixture *= 0.9 / peak
    mixture_id = f"p2b_{split}_pair_{pair[0]}_{pair[1]}_{index:04d}"
    queries = tuple(ScoreQuery(f"{mixture_id}_q{source_index}", 0, 0, source_index, scores[source_index]) for source_index in range(2))
    return ScoreMixture(mixture_id, split, seed, audio.sample_rate, mixture, sources, queries, source_patterns)


def write_rows(rows: list[dict], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_hash(splits: dict[str, list[ScoreMixture]]) -> str:
    payload = "\n".join(json.dumps(_manifest_record(mixture), sort_keys=True) for split in sorted(splits) for mixture in splits[split])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
