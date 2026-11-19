from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioConfig, AudioTransform
from .score import ScoreEvent, events_to_piano_roll, score_hash
from .toy import array_hash, stable_toy_seed


@dataclass(frozen=True)
class ScoreQuery:
    query_id: str
    timbre_id: int
    position_id: int
    target_source_index: int
    score: np.ndarray

    @property
    def score_id(self) -> str:
        return score_hash(self.score)


@dataclass
class ScoreMixture:
    mixture_id: str
    split: str
    seed: int
    sample_rate: int
    mixture: np.ndarray
    sources: np.ndarray
    queries: tuple[ScoreQuery, ...]
    pattern_pair: tuple[int, int]

    @property
    def mixture_hash(self) -> str:
        return array_hash(self.mixture)

    def target_for_query(self, query: ScoreQuery) -> np.ndarray:
        return self.sources[query.target_source_index]

    def manifest_record(self) -> dict:
        return {
            "mixture_id": self.mixture_id,
            "split": self.split,
            "seed": self.seed,
            "sample_rate": self.sample_rate,
            "channels": 2,
            "num_samples": int(self.mixture.shape[-1]),
            "mixture_hash": self.mixture_hash,
            "source_hashes": [array_hash(source) for source in self.sources],
            "pattern_pair": list(self.pattern_pair),
            "queries": [
                {
                    "query_id": query.query_id,
                    "condition_id": {
                        "timbre_id": query.timbre_id,
                        "position_id": query.position_id,
                    },
                    "target_source_index": query.target_source_index,
                    "score_hash": query.score_id,
                }
                for query in self.queries
            ],
        }


class ScoreQueryDataset:
    def __init__(self, mixtures: list[ScoreMixture], audio_config: AudioConfig):
        self.mixtures = mixtures
        self.transform = AudioTransform(audio_config)
        self.records: list[tuple[int, ScoreQuery]] = []
        for mixture_index, mixture in enumerate(mixtures):
            for query in mixture.queries:
                self.records.append((mixture_index, query))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        import torch

        mixture_index, query = self.records[index]
        mixture = self.mixtures[mixture_index]
        mixture_wave = torch.from_numpy(mixture.mixture).unsqueeze(0)
        target_wave = torch.from_numpy(mixture.target_for_query(query)).unsqueeze(0)
        return {
            "spec": self.transform.stft(mixture_wave)[0],
            "target": self.transform.stft(target_wave)[0],
            "score_roll": torch.from_numpy(query.score.copy()),
            "timbre_id": torch.tensor(query.timbre_id, dtype=torch.long),
            "position_id": torch.tensor(query.position_id, dtype=torch.long),
            "mixture_id": mixture.mixture_id,
            "query_id": query.query_id,
            "target_source_index": query.target_source_index,
            "score_hash": query.score_id,
        }


class BalancedScoreBatchSampler:
    def __init__(self, dataset: ScoreQueryDataset, batch_size: int = 4):
        if batch_size < 2:
            raise ValueError("Score-S training requires batch_size >= 2")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.groups = self._build_groups()
        self.cursor = 0
        self.completed_batches = 0
        self.access_counts: dict[str, int] = {}

    def _build_groups(self) -> list[list[int]]:
        groups: dict[str, list[int]] = {}
        for index, (mixture_index, _query) in enumerate(self.dataset.records):
            mixture_id = self.dataset.mixtures[mixture_index].mixture_id
            groups.setdefault(mixture_id, []).append(index)
        values = [indices for _key, indices in sorted(groups.items())]
        if not values or any(len(indices) != 2 for indices in values):
            raise ValueError("Score-S sampler requires exactly two queries per mixture")
        return values

    def next_batch_indices(self) -> list[int]:
        batch: list[int] = []
        while len(batch) < self.batch_size:
            batch.extend(self.groups[self.cursor % len(self.groups)])
            self.cursor += 1
        batch = batch[: self.batch_size]
        for index in batch:
            key = str(index)
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
        self.completed_batches += 1
        return batch

    def state_dict(self) -> dict:
        return {
            "cursor": self.cursor,
            "completed_batches": self.completed_batches,
            "access_counts": dict(self.access_counts),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("Score-S sampler state must be a mapping")
        required = {"cursor", "completed_batches", "access_counts"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"Score-S sampler state is missing: {', '.join(missing)}")
        cursor = int(state["cursor"])
        completed = int(state["completed_batches"])
        if cursor < 0 or completed < 0:
            raise ValueError("Score-S sampler cursor and completed_batches must be non-negative")
        counts = {str(key): int(value) for key, value in dict(state["access_counts"]).items()}
        if any(value < 0 for value in counts.values()):
            raise ValueError("Score-S sampler access counts must be non-negative")
        if any(int(key) < 0 or int(key) >= len(self.dataset) for key in counts):
            raise ValueError("Score-S sampler access count index is out of range")
        self.cursor = cursor
        self.completed_batches = completed
        self.access_counts = counts

    def access_count_rows(self) -> list[dict]:
        rows = []
        for key, visits in sorted(self.access_counts.items(), key=lambda item: int(item[0])):
            index = int(key)
            mixture_index, query = self.dataset.records[index]
            mixture = self.dataset.mixtures[mixture_index]
            rows.append(
                {
                    "dataset_index": index,
                    "mixture_id": mixture.mixture_id,
                    "split": mixture.split,
                    "query_id": query.query_id,
                    "target_source_index": query.target_source_index,
                    "score_hash": query.score_id,
                    "visits": int(visits),
                }
            )
        return rows


def collate_score_batch(items: list[dict]) -> dict:
    import torch

    return {
        "spec": torch.stack([item["spec"] for item in items]),
        "target": torch.stack([item["target"] for item in items]),
        "score_roll": torch.stack([item["score_roll"] for item in items]),
        "timbre_id": torch.stack([item["timbre_id"] for item in items]),
        "position_id": torch.stack([item["position_id"] for item in items]),
        "mixture_id": [item["mixture_id"] for item in items],
        "query_id": [item["query_id"] for item in items],
        "target_source_index": [item["target_source_index"] for item in items],
    }


def generate_score_corpus(config: dict) -> list[ScoreMixture]:
    score_s = config["score_s"]
    audio = AudioConfig.from_dict(config["audio"])
    seed = int(config["experiment"]["seed"])
    result: list[ScoreMixture] = []
    counts = {"train": int(score_s["train_mixtures"]), "probe": int(score_s["probe_mixtures"])}
    for split, count in counts.items():
        for index in range(count):
            mixture_seed = stable_toy_seed(seed, "Score-S", split, index)
            pair = _pattern_pair(split, index, int(score_s["num_patterns"]))
            result.append(_generate_score_mixture(config, audio, split, index, mixture_seed, pair))
    return result


def validate_score_corpus(mixtures: list[ScoreMixture], config: dict) -> dict:
    errors: list[str] = []
    mixture_hash_splits: dict[str, set[str]] = {}
    source_hash_splits: dict[str, set[str]] = {}
    for mixture in mixtures:
        mixture_hash_splits.setdefault(mixture.mixture_hash, set()).add(mixture.split)
        for source in mixture.sources:
            source_hash_splits.setdefault(array_hash(source), set()).add(mixture.split)
        if not np.allclose(mixture.mixture, mixture.sources.sum(axis=0), atol=1.0e-7, rtol=1.0e-7):
            errors.append(f"{mixture.mixture_id}: mixture is not exact source sum")
        if len(mixture.queries) != 2 or any(query.timbre_id != 0 or query.position_id != 0 for query in mixture.queries):
            errors.append(f"{mixture.mixture_id}: Score-S queries must share condition ids (0, 0)")
        for query in mixture.queries:
            if query.score.shape != (128, int(config["audio"]["num_frames"])):
                errors.append(f"{query.query_id}: score shape is invalid")
            if query.target_source_index not in {0, 1}:
                errors.append(f"{query.query_id}: target source index is invalid")
    mixture_leakage = {key: value for key, value in mixture_hash_splits.items() if len(value) > 1}
    source_leakage = {key: value for key, value in source_hash_splits.items() if len(value) > 1}
    if mixture_leakage:
        errors.append("train/probe mixture hash leakage detected")
    if source_leakage:
        errors.append("train/probe source hash leakage detected")
    return {
        "ok": not errors,
        "errors": errors,
        "num_mixtures": len(mixtures),
        "num_queries": sum(len(mixture.queries) for mixture in mixtures),
        "mixture_hash_leakage_count": len(mixture_leakage),
        "source_hash_leakage_count": len(source_leakage),
    }


def write_score_manifest(mixtures: list[ScoreMixture], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(__import__("json").dumps(mixture.manifest_record(), sort_keys=True) for mixture in mixtures) + "\n", encoding="utf-8")


def _pattern_pair(split: str, index: int, num_patterns: int) -> tuple[int, int]:
    if num_patterns < 6:
        raise ValueError("Score-S requires at least six deterministic pattern families")
    train_pairs = ((0, 1), (2, 3), (0, 2), (1, 3))
    probe_pairs = ((4, 5), (0, 5), (2, 4))
    pairs = train_pairs if split == "train" else probe_pairs
    return pairs[index % len(pairs)]


def _generate_score_mixture(config, audio: AudioConfig, split: str, index: int, seed: int, pair: tuple[int, int]) -> ScoreMixture:
    rng = np.random.default_rng(seed)
    length = audio.segment_samples
    events_a = _pattern_events(pair[0], audio)
    events_b = _pattern_events(pair[1], audio)
    score_a = events_to_piano_roll(events_a, num_frames=audio.num_frames, sample_rate=audio.sample_rate, hop_length=audio.hop_length, center=audio.center)
    score_b = events_to_piano_roll(events_b, num_frames=audio.num_frames, sample_rate=audio.sample_rate, hop_length=audio.hop_length, center=audio.center)
    source_a = _render_events(events_a, audio.sample_rate, length, phase=float(rng.uniform(0, 2 * np.pi)))
    source_b = _render_events(events_b, audio.sample_rate, length, phase=float(rng.uniform(0, 2 * np.pi)))
    sources = np.stack([source_a, source_b]).astype(np.float32)
    mixture = sources.sum(axis=0)
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.9:
        sources *= 0.9 / peak
        mixture *= 0.9 / peak
    mixture_id = f"{split}_score_s_{index:04d}"
    queries = (
        ScoreQuery(f"{mixture_id}_q0", 0, 0, 0, score_a),
        ScoreQuery(f"{mixture_id}_q1", 0, 0, 1, score_b),
    )
    return ScoreMixture(mixture_id, split, seed, audio.sample_rate, mixture.astype(np.float32), sources, queries, pair)


def _pattern_events(pattern: int, audio: AudioConfig) -> tuple[ScoreEvent, ...]:
    unit = float(audio.segment_samples) / float(audio.sample_rate) / 4.0
    patterns = {
        0: ((0, 60), (2, 64)),
        1: ((1, 67), (3, 72)),
        2: ((0, 60), (1, 67), (2, 64), (3, 72)),
        3: ((0, 72), (1, 64), (2, 67), (3, 60)),
        4: ((0, 62), (2, 69)),
        5: ((1, 69), (3, 62)),
    }
    return tuple(
        ScoreEvent("score_s", start * unit, (start + 1) * unit, pitch, 1.0)
        for start, pitch in patterns[int(pattern)]
    )


def _render_events(events: tuple[ScoreEvent, ...], sample_rate: int, length: int, phase: float) -> np.ndarray:
    wave = np.zeros((2, length), dtype=np.float32)
    for event in events:
        left = max(0, int(round(event.onset_seconds * sample_rate)))
        right = min(length, int(round(event.offset_seconds * sample_rate)))
        if right <= left:
            continue
        t = np.arange(right - left, dtype=np.float32) / float(sample_rate)
        frequency = 440.0 * (2.0 ** ((event.midi_pitch - 69) / 12.0))
        note = 0.65 * np.sin(2 * np.pi * frequency * t + phase)
        note += 0.16 * np.sin(2 * np.pi * 2 * frequency * t + phase / 2.0)
        envelope = np.ones_like(note)
        fade = min(len(note) // 8, max(1, sample_rate // 100))
        envelope[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False)
        wave[:, left:right] += (note * envelope * float(event.velocity))[None, :]
    return wave
