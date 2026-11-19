from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterator

import numpy as np

from .audio import AudioConfig, AudioTransform


@dataclass(frozen=True)
class ToyQuery:
    query_id: str
    timbre_id: int
    position_id: int
    source_index: int


@dataclass
class ToyMixture:
    mixture_id: str
    task: str
    split: str
    seed: int
    sample_rate: int
    mixture: np.ndarray
    sources: np.ndarray
    source_timbres: tuple[int, ...]
    source_positions: tuple[int, ...]
    queries: tuple[ToyQuery, ...]

    @property
    def mixture_hash(self) -> str:
        return array_hash(self.mixture)

    def target_for_query(self, query: ToyQuery) -> np.ndarray:
        return self.sources[query.source_index]

    def manifest_record(self) -> dict:
        return {
            "mixture_id": self.mixture_id,
            "task": self.task,
            "split": self.split,
            "seed": self.seed,
            "sample_rate": self.sample_rate,
            "channels": 2,
            "num_samples": int(self.mixture.shape[-1]),
            "mixture_hash": self.mixture_hash,
            "source_hashes": [array_hash(source) for source in self.sources],
            "source_timbres": list(self.source_timbres),
            "source_positions": list(self.source_positions),
            "queries": [
                {
                    "query_id": query.query_id,
                    "timbre_id": query.timbre_id,
                    "position_id": query.position_id,
                    "source_index": query.source_index,
                }
                for query in self.queries
            ],
        }


class ToyQueryDataset:
    def __init__(self, mixtures: list[ToyMixture], audio_config: AudioConfig):
        self.mixtures = mixtures
        self.transform = AudioTransform(audio_config)
        self.records: list[tuple[int, ToyQuery]] = []
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
        spec = self.transform.stft(mixture_wave)[0]
        target = self.transform.stft(target_wave)[0]
        return {
            "spec": spec,
            "target": target,
            "timbre_id": torch.tensor(query.timbre_id, dtype=torch.long),
            "position_id": torch.tensor(query.position_id, dtype=torch.long),
            "mixture_id": mixture.mixture_id,
            "query_id": query.query_id,
            "task": mixture.task,
            "source_index": query.source_index,
        }


class BalancedToyBatchSampler:
    def __init__(self, dataset: ToyQueryDataset, batch_size: int = 4, tasks: tuple[str, ...] | None = None):
        if batch_size < 2:
            raise ValueError("Toy training requires batch_size >= 2")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.tasks = tasks or tuple(sorted({mixture.task for mixture in dataset.mixtures}))
        self.groups_by_task = self._build_groups()
        self.cursors = {task: 0 for task in self.tasks}
        self.completed_batches = 0
        self.access_counts: dict[str, int] = {}

    def next_batch_indices(self) -> list[int]:
        if "TP" in self.tasks and len(self.tasks) == 1:
            batch = self._next_group("TP")
            batch = batch[: self.batch_size]
            self._record_access(batch)
            self.completed_batches += 1
            return batch

        batch: list[int] = []
        task_cycle = [task for task in ("T", "P") if task in self.tasks]
        if not task_cycle:
            task_cycle = list(self.tasks)
        while len(batch) < self.batch_size:
            for task in task_cycle:
                batch.extend(self._next_group(task))
                if len(batch) >= self.batch_size:
                    break
        batch = batch[: self.batch_size]
        self._record_access(batch)
        self.completed_batches += 1
        return batch

    def state_dict(self) -> dict:
        return {
            "cursors": dict(self.cursors),
            "completed_batches": self.completed_batches,
            "access_counts": dict(self.access_counts),
        }

    def load_state_dict(self, state: dict) -> None:
        self.cursors = {task: int(value) for task, value in state["cursors"].items()}
        self.completed_batches = int(state["completed_batches"])
        self.access_counts = {str(key): int(value) for key, value in state.get("access_counts", {}).items()}

    def access_count_rows(self) -> list[dict]:
        rows = []
        for index_key, visits in sorted(self.access_counts.items(), key=lambda item: int(item[0])):
            index = int(index_key)
            mixture_index, query = self.dataset.records[index]
            mixture = self.dataset.mixtures[mixture_index]
            rows.append(
                {
                    "dataset_index": index,
                    "mixture_id": mixture.mixture_id,
                    "task": mixture.task,
                    "split": mixture.split,
                    "query_id": query.query_id,
                    "timbre_id": query.timbre_id,
                    "position_id": query.position_id,
                    "source_index": query.source_index,
                    "visits": int(visits),
                }
            )
        return rows

    def _build_groups(self) -> dict[str, list[list[int]]]:
        groups: dict[tuple[str, str], list[int]] = {}
        for index, (mixture_index, _query) in enumerate(self.dataset.records):
            mixture = self.dataset.mixtures[mixture_index]
            groups.setdefault((mixture.task, mixture.mixture_id), []).append(index)
        by_task: dict[str, list[list[int]]] = {task: [] for task in self.tasks}
        for (task, _mixture_id), indices in sorted(groups.items()):
            if task in by_task:
                by_task[task].append(indices)
        missing = [task for task, groups_for_task in by_task.items() if not groups_for_task]
        if missing:
            raise ValueError(f"No toy query groups for task(s): {', '.join(missing)}")
        return by_task

    def _next_group(self, task: str) -> list[int]:
        groups = self.groups_by_task[task]
        cursor = self.cursors[task] % len(groups)
        self.cursors[task] += 1
        return list(groups[cursor])

    def _record_access(self, indices: list[int]) -> None:
        for index in indices:
            key = str(int(index))
            self.access_counts[key] = self.access_counts.get(key, 0) + 1


def collate_toy_batch(items: list[dict]) -> dict:
    import torch

    return {
        "spec": torch.stack([item["spec"] for item in items], dim=0),
        "target": torch.stack([item["target"] for item in items], dim=0),
        "timbre_id": torch.stack([item["timbre_id"] for item in items], dim=0),
        "position_id": torch.stack([item["position_id"] for item in items], dim=0),
        "mixture_id": [item["mixture_id"] for item in items],
        "query_id": [item["query_id"] for item in items],
        "task": [item["task"] for item in items],
        "source_index": [item["source_index"] for item in items],
    }


def generate_toy_corpus(config: dict, tasks: tuple[str, ...] | None = None) -> list[ToyMixture]:
    toy = config["toy"]
    selected_tasks = tasks or tuple(toy["tasks"])
    sample_rate = int(config["audio"]["sample_rate"])
    num_samples = int(config["audio"]["segment_samples"])
    base_seed = int(config["experiment"]["seed"])
    mixtures: list[ToyMixture] = []
    for split, count_key in (("train", "train_mixtures_per_task"), ("probe", "probe_mixtures_per_task")):
        for task in selected_tasks:
            if task == "TP":
                count = int(toy["tp_train_mixtures"] if split == "train" else toy["tp_probe_mixtures"])
            else:
                count = int(toy[count_key])
            for index in range(count):
                seed = _stable_seed(base_seed, split, task, index)
                mixtures.append(_generate_one(config, task, split, index, seed, sample_rate, num_samples))
    return mixtures


def toy_scene_key(mixture: ToyMixture) -> str:
    if mixture.task == "T":
        timbre_pair = "-".join(str(value) for value in sorted(mixture.source_timbres))
        positions = set(mixture.source_positions)
        if len(positions) != 1:
            raise ValueError(f"{mixture.mixture_id}: Toy-T scene must have one shared position")
        return f"T:timbres={timbre_pair}:position={next(iter(positions))}"
    if mixture.task == "P":
        timbres = set(mixture.source_timbres)
        if len(timbres) != 1:
            raise ValueError(f"{mixture.mixture_id}: Toy-P scene must have one shared timbre")
        position_pair = "-".join(str(value) for value in sorted(mixture.source_positions))
        return f"P:timbre={next(iter(timbres))}:positions={position_pair}"
    if mixture.task == "TP":
        timbre_values = "-".join(str(value) for value in mixture.source_timbres)
        position_values = "-".join(str(value) for value in mixture.source_positions)
        return f"TP:timbres={timbre_values}:positions={position_values}"
    raise ValueError(f"Unsupported toy task: {mixture.task}")


def all_toy_scenes(task: str) -> list[tuple]:
    if task == "T":
        return [(tuple(pair), position) for pair in combinations((0, 1, 2), 2) for position in range(4)]
    if task == "P":
        return [(timbre, tuple(pair)) for timbre in range(3) for pair in combinations((0, 1, 2, 3), 2)]
    raise ValueError(f"Unsupported coverage task: {task}")


def scene_label(task: str, scene: tuple) -> str:
    if task == "T":
        timbre_pair, position = scene
        return f"t{timbre_pair[0]}-{timbre_pair[1]}_p{position}"
    if task == "P":
        timbre, position_pair = scene
        return f"t{timbre}_p{position_pair[0]}-{position_pair[1]}"
    raise ValueError(f"Unsupported coverage task: {task}")


def scene_key_from_spec(task: str, scene: tuple) -> str:
    if task == "T":
        timbre_pair, position = scene
        return f"T:timbres={timbre_pair[0]}-{timbre_pair[1]}:position={position}"
    if task == "P":
        timbre, position_pair = scene
        return f"P:timbre={timbre}:positions={position_pair[0]}-{position_pair[1]}"
    raise ValueError(f"Unsupported coverage task: {task}")


def generate_toy_mixture_for_scene(
    config: dict,
    task: str,
    split: str,
    mixture_id: str,
    seed: int,
    scene: tuple,
) -> ToyMixture:
    sample_rate = int(config["audio"]["sample_rate"])
    num_samples = int(config["audio"]["segment_samples"])
    rng = np.random.default_rng(seed)
    if task == "T":
        timbre_pair, position = scene
        order = rng.permutation(2)
        timbres = tuple(int(timbre_pair[int(index)]) for index in order)
        positions = (int(position), int(position))
    elif task == "P":
        timbre, position_pair = scene
        order = rng.permutation(2)
        timbres = (int(timbre), int(timbre))
        positions = tuple(int(position_pair[int(index)]) for index in order)
    else:
        raise ValueError(f"Unsupported coverage task: {task}")
    return _generate_from_assignments(
        config,
        task=task,
        split=split,
        mixture_id=mixture_id,
        seed=seed,
        sample_rate=sample_rate,
        num_samples=num_samples,
        timbres=timbres,
        positions=positions,
        rng=rng,
    )


def validate_toy_corpus(mixtures: list[ToyMixture], config: dict) -> dict:
    errors: list[str] = []
    max_sum_error = 0.0
    min_energy = float("inf")
    max_abs_correlation = 0.0
    split_by_hash: dict[str, set[str]] = {}
    for mixture in mixtures:
        reconstructed = mixture.sources.sum(axis=0)
        denom = float(np.linalg.norm(mixture.mixture) + 1.0e-12)
        sum_error = float(np.linalg.norm(mixture.mixture - reconstructed) / denom)
        max_sum_error = max(max_sum_error, sum_error)
        if sum_error > 1.0e-5:
            errors.append(f"{mixture.mixture_id}: mixture does not equal source sum")
        split_by_hash.setdefault(mixture.mixture_hash, set()).add(mixture.split)
        for query in mixture.queries:
            target = mixture.target_for_query(query)
            energy = float(np.sum(target**2))
            min_energy = min(min_energy, energy)
            if energy <= float(config["toy"]["sanity_min_target_energy"]):
                errors.append(f"{query.query_id}: target is silent or near silent")
            if mixture.source_timbres[query.source_index] != query.timbre_id:
                errors.append(f"{query.query_id}: timbre query does not map to target")
            if mixture.source_positions[query.source_index] != query.position_id:
                errors.append(f"{query.query_id}: position query does not map to target")
        for i in range(len(mixture.sources)):
            for j in range(i + 1, len(mixture.sources)):
                corr = abs(_correlation(mixture.sources[i], mixture.sources[j]))
                max_abs_correlation = max(max_abs_correlation, corr)
        if mixture.task == "T":
            positions = {query.position_id for query in mixture.queries}
            timbres = {query.timbre_id for query in mixture.queries}
            if len(positions) != 1 or len(timbres) != 2:
                errors.append(f"{mixture.mixture_id}: Toy-T must vary timbre at one position")
        if mixture.task == "P":
            positions = {query.position_id for query in mixture.queries}
            timbres = {query.timbre_id for query in mixture.queries}
            if len(positions) != 2 or len(timbres) != 1:
                errors.append(f"{mixture.mixture_id}: Toy-P must vary position at one timbre")
        if mixture.task == "TP" and len(mixture.queries) != 4:
            errors.append(f"{mixture.mixture_id}: Toy-TP must expose four legal queries")

    leakage = {hash_value: splits for hash_value, splits in split_by_hash.items() if len(splits) > 1}
    if leakage:
        errors.append("train/probe leakage detected by mixture hash")
    threshold = float(config["toy"]["sanity_max_abs_correlation"])
    if max_abs_correlation > threshold:
        errors.append(f"source abs correlation {max_abs_correlation:.6f} exceeds {threshold:.6f}")
    return {
        "ok": not errors,
        "errors": errors,
        "num_mixtures": len(mixtures),
        "max_mixture_sum_relative_l2": max_sum_error,
        "min_target_energy": min_energy if min_energy != float("inf") else 0.0,
        "max_source_abs_correlation": max_abs_correlation,
        "train_probe_hash_leakage_count": len(leakage),
    }


def write_toy_manifest(mixtures: list[ToyMixture], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(mixture.manifest_record(), sort_keys=True) for mixture in mixtures]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_query_records(mixtures: list[ToyMixture]) -> Iterator[tuple[ToyMixture, ToyQuery]]:
    for mixture in mixtures:
        for query in mixture.queries:
            yield mixture, query


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def stable_toy_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _generate_one(
    config: dict,
    task: str,
    split: str,
    index: int,
    seed: int,
    sample_rate: int,
    num_samples: int,
) -> ToyMixture:
    rng = np.random.default_rng(seed)
    if task == "T":
        timbres = tuple(int(v) for v in rng.choice([0, 1, 2], size=2, replace=False))
        position = int(rng.integers(0, 4))
        positions = (position, position)
    elif task == "P":
        timbre = int(rng.integers(0, 3))
        timbres = (timbre, timbre)
        positions = tuple(int(v) for v in rng.choice([0, 1, 2, 3], size=2, replace=False))
    elif task == "TP":
        timbres = (0, 0, 1, 2)
        positions = tuple(int(v) for v in rng.permutation(4))
    else:
        raise ValueError(f"Unsupported toy task: {task}")

    mixture_id = f"{split}_{task}_{index:04d}"
    return _generate_from_assignments(
        config,
        task=task,
        split=split,
        mixture_id=mixture_id,
        seed=seed,
        sample_rate=sample_rate,
        num_samples=num_samples,
        timbres=timbres,
        positions=positions,
        rng=rng,
    )


def _generate_from_assignments(
    config: dict,
    task: str,
    split: str,
    mixture_id: str,
    seed: int,
    sample_rate: int,
    num_samples: int,
    timbres: tuple[int, ...],
    positions: tuple[int, ...],
    rng,
) -> ToyMixture:
    sources = []
    for source_index, (timbre, position) in enumerate(zip(timbres, positions, strict=True)):
        mono = _harmonic_source(config, rng, timbre, source_index, sample_rate, num_samples)
        sources.append(_stereo_image(mono, position, config))
    source_array = np.stack(sources, axis=0).astype(np.float32)
    mixture = source_array.sum(axis=0)
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.9:
        scale = 0.9 / peak
        source_array *= scale
        mixture *= scale
    queries = tuple(
        ToyQuery(
            query_id=f"{mixture_id}_q{source_index}",
            timbre_id=int(timbre),
            position_id=int(position),
            source_index=source_index,
        )
        for source_index, (timbre, position) in enumerate(zip(timbres, positions, strict=True))
    )
    return ToyMixture(
        mixture_id=mixture_id,
        task=task,
        split=split,
        seed=seed,
        sample_rate=sample_rate,
        mixture=mixture.astype(np.float32),
        sources=source_array.astype(np.float32),
        source_timbres=tuple(int(v) for v in timbres),
        source_positions=tuple(int(v) for v in positions),
        queries=queries,
    )


def _harmonic_source(config: dict, rng, timbre: int, source_index: int, sample_rate: int, num_samples: int) -> np.ndarray:
    min_events = int(config["toy"]["min_note_events"])
    max_events = int(config["toy"]["max_note_events"])
    events = int(rng.integers(min_events, max_events + 1))
    base_freqs = np.array([220.0, 246.94, 277.18, 329.63, 392.0], dtype=np.float32)
    harmonic_weights = {
        0: (1.0, 0.32, 0.18, 0.08),
        1: (1.0, 0.55, 0.24, 0.12),
        2: (1.0, 0.22, 0.1, 0.04),
    }[int(timbre)]
    signal = np.zeros(num_samples, dtype=np.float32)
    event_len = max(1, num_samples // events)
    for event in range(events):
        start = event * event_len
        end = num_samples if event == events - 1 else min(num_samples, (event + 1) * event_len)
        if end <= start:
            continue
        freq = float(rng.choice(base_freqs) * (1.0 + 0.015 * rng.normal()) * (1.0 + 0.01 * source_index))
        t = np.arange(end - start, dtype=np.float32) / float(sample_rate)
        phase = float(rng.uniform(0, 2 * np.pi))
        note = np.zeros_like(t)
        for harmonic, weight in enumerate(harmonic_weights, start=1):
            harmonic_freq = freq * harmonic
            if harmonic_freq >= sample_rate / 2:
                continue
            note += float(weight) * np.sin(2 * np.pi * harmonic_freq * t + phase / harmonic)
        envelope = _smooth_envelope(len(t))
        signal[start:end] += (0.045 * (0.85 + 0.3 * rng.random()) * note * envelope).astype(np.float32)
    return signal


def _smooth_envelope(length: int) -> np.ndarray:
    if length <= 2:
        return np.ones(length, dtype=np.float32)
    attack = max(1, min(length // 4, 512))
    release = attack
    envelope = np.ones(length, dtype=np.float32)
    envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
    return envelope


def _stereo_image(mono: np.ndarray, position: int, config: dict) -> np.ndarray:
    degrees = float(config["toy"]["position_degrees"][position])
    theta = math.radians(degrees)
    gains = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    return gains[:, None] * mono[None, :]


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = left.reshape(-1).astype(np.float64)
    y = right.reshape(-1).astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _stable_seed(base_seed: int, split: str, task: str, index: int) -> int:
    payload = f"{base_seed}:{split}:{task}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)
