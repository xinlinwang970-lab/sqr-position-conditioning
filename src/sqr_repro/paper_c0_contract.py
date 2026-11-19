"""Paper C0 four-query data contract.

This module is intentionally independent from the historical loaders.  It defines
the identity and validation rules needed before a formal SQR loader is allowed to
consume real data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


EXPECTED_PARTS = ("violin1", "violin2", "viola", "cello")
EXPECTED_TIMBRES = {"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}
VALID_SEATING_TYPES = {1, 2}

# The paper labels the positions pos1..pos4 (0..3), but the PDF Figure 1 is a
# visual source rather than a machine-readable metadata table.  These are
# explicit candidate maps transcribed as hypotheses and must remain
# `verified=False` until the downloaded SQR metadata confirms them.
SEATING_POSITION_MAP = {
    1: {"pos0": "violin1", "pos1": "viola", "pos2": "cello", "pos3": "violin2"},
    2: {"pos0": "violin1", "pos1": "violin2", "pos2": "viola", "pos3": "cello"},
}
SEATING_POSITION_MAP_VERIFIED = False


@dataclass(frozen=True)
class PaperC0Source:
    source_index: int
    part_id: str
    timbre_id: int
    position_id: int
    path: str
    waveform_hash: str


@dataclass(frozen=True)
class PaperC0Mixture:
    mixture_id: str
    work_id: str
    movement_id: str
    chunk_id: str
    seating_type: int
    split: str
    mixture_path: str
    mixture_hash: str
    sources: tuple[PaperC0Source, ...]


@dataclass(frozen=True)
class PaperC0Query:
    mixture_id: str
    query_id: str
    target_source_index: int
    timbre_id: int
    position_id: int
    target_waveform_hash: str


@dataclass(frozen=True)
class SeatingMappingEvidence:
    seating_type: int
    position_mapping: tuple[tuple[int, str], ...]
    source_reference: str
    source_path: str
    source_sha256: str
    mapping_sha256: str
    verification_record_hash: str
    verification_status: str


def mapping_sha256(position_mapping: tuple[tuple[int, str], ...]) -> str:
    payload = json.dumps(sorted(position_mapping), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mixture(mixture: PaperC0Mixture) -> None:
    """Raise ValueError when a mixture violates the formal four-source contract."""
    if not mixture.mixture_id or not mixture.work_id or not mixture.movement_id or not mixture.chunk_id:
        raise ValueError("mixture/work/movement/chunk identifiers must be non-empty")
    if mixture.seating_type not in VALID_SEATING_TYPES:
        raise ValueError("seating_type must be 1 or 2")
    if mixture.split not in {"train", "validation", "test", "probe"}:
        raise ValueError("split must be train, validation, test, or probe")
    if len(mixture.sources) != 4:
        raise ValueError("a formal mixture must contain exactly four sources")
    if {source.source_index for source in mixture.sources} != set(range(4)):
        raise ValueError("source_index must be a permutation of 0..3")
    if {source.part_id for source in mixture.sources} != set(EXPECTED_PARTS):
        raise ValueError("parts must be violin1, violin2, viola, cello exactly")
    if len({source.position_id for source in mixture.sources}) != 4:
        raise ValueError("the four sources must have unique seating positions")
    if any(source.position_id not in range(4) for source in mixture.sources):
        raise ValueError("position_id must be in 0..3")
    for source in mixture.sources:
        expected = EXPECTED_TIMBRES[source.part_id]
        if source.timbre_id != expected:
            raise ValueError(f"{source.part_id} must use timbre_id={expected}")
        if not source.path or not source.waveform_hash:
            raise ValueError("source path and waveform_hash are required")
    if not mixture.mixture_path or not mixture.mixture_hash:
        raise ValueError("mixture path and mixture_hash are required")
    if mixture.split == "train" and mixture.seating_type != 2:
        raise ValueError("formal train split is restricted to seating type #2")


def queries_for_mixture(mixture: PaperC0Mixture) -> tuple[PaperC0Query, ...]:
    validate_mixture(mixture)
    ordered = sorted(mixture.sources, key=lambda source: source.source_index)
    return tuple(
        PaperC0Query(
            mixture_id=mixture.mixture_id,
            query_id=f"{mixture.mixture_id}:q{source.source_index}",
            target_source_index=source.source_index,
            timbre_id=source.timbre_id,
            position_id=source.position_id,
            target_waveform_hash=source.waveform_hash,
        )
        for source in ordered
    )


def validate_seating_mapping(
    mixture: PaperC0Mixture,
    seating_map: Mapping[int, Mapping[str, str]] = SEATING_POSITION_MAP,
) -> None:
    """Validate physical position labels independently from source ordering."""
    validate_mixture(mixture)
    expected = seating_map.get(mixture.seating_type)
    if expected is None:
        raise ValueError(f"no seating map for type {mixture.seating_type}")
    expected_by_position = {int(key.removeprefix("pos")): value for key, value in expected.items()}
    actual_by_position = {source.position_id: source.part_id for source in mixture.sources}
    if actual_by_position != expected_by_position:
        raise ValueError(
            f"seating type {mixture.seating_type} mapping mismatch: "
            f"expected {expected_by_position}, got {actual_by_position}"
        )


def validate_verified_seating_mapping(
    mixture: PaperC0Mixture,
    evidence: SeatingMappingEvidence,
    trusted_sources: Mapping[str, str] | None = None,
) -> None:
    if evidence.seating_type != mixture.seating_type:
        raise ValueError("seating evidence type mismatch")
    if evidence.verification_status != "VERIFIED":
        raise ValueError("seating mapping evidence is not VERIFIED")
    if not evidence.source_reference or not evidence.source_path:
        raise ValueError("verified seating evidence requires a trusted source and path")
    if trusted_sources is None or evidence.source_reference not in trusted_sources:
        raise ValueError("seating evidence source is not in the trusted registry")
    actual_source_hash = file_sha256(evidence.source_path)
    if actual_source_hash != evidence.source_sha256 or actual_source_hash != trusted_sources[evidence.source_reference]:
        raise ValueError("seating evidence source hash mismatch")
    expected = dict(evidence.position_mapping)
    actual = {source.position_id: source.part_id for source in mixture.sources}
    if actual != expected:
        raise ValueError(f"verified seating mapping mismatch: expected {expected}, got {actual}")
    if mapping_sha256(evidence.position_mapping) != evidence.mapping_sha256:
        raise ValueError("seating mapping hash mismatch")
    record_payload = f"{evidence.source_reference}|{evidence.source_sha256}|{evidence.mapping_sha256}".encode("utf-8")
    if hashlib.sha256(record_payload).hexdigest() != evidence.verification_record_hash:
        raise ValueError("seating verification record hash mismatch")


def validate_dataset_splits(mixtures: Iterable[PaperC0Mixture]) -> None:
    """Reject work/movement/seating/chunk identity leakage across splits."""
    rows = tuple(mixtures)
    validate_collection(rows)
    by_work: dict[str, set[str]] = {}
    by_movement: dict[tuple[str, str], set[str]] = {}
    by_chunk: dict[tuple[str, str, str], set[str]] = {}
    for mixture in rows:
        by_work.setdefault(mixture.work_id, set()).add(mixture.split)
        by_movement.setdefault((mixture.work_id, mixture.movement_id), set()).add(mixture.split)
        by_chunk.setdefault((mixture.work_id, mixture.movement_id, mixture.chunk_id), set()).add(mixture.split)
    for label, groups in (("work", by_work), ("movement", by_movement), ("chunk", by_chunk)):
        leaked = {key: sorted(value) for key, value in groups.items() if len(value) > 1}
        if leaked:
            raise ValueError(f"{label} crosses dataset splits: {leaked}")


def validate_formal_train_collection(
    mixtures: Iterable[PaperC0Mixture],
    mapping_evidence: Mapping[int, SeatingMappingEvidence] | None = None,
    trusted_sources: Mapping[str, str] | None = None,
) -> None:
    """Formal C0 training input guard; fixture sampler is not a trainer loader."""
    rows = tuple(mixtures)
    if not rows:
        raise ValueError("formal train collection must not be empty")
    validate_dataset_splits(rows)
    for mixture in rows:
        if mixture.split != "train":
            raise ValueError("formal train collection cannot contain validation/test/probe")
        if mixture.seating_type != 2:
            raise ValueError("formal train collection cannot contain seating type #1")
    if mapping_evidence is None:
        raise ValueError("formal train collection requires verified seating evidence")
    for mixture in rows:
        evidence = mapping_evidence.get(mixture.seating_type)
        if evidence is None:
            raise ValueError(f"missing verified seating evidence for type {mixture.seating_type}")
        validate_verified_seating_mapping(mixture, evidence, trusted_sources)


class PaperC0FourQuerySampler:
    """Fixture-only sampler: one mixture yields all four legal queries.

    It is deliberately not the formal Paper C0 epoch/batch sampler.
    """

    def __init__(self, mixtures: Sequence[PaperC0Mixture]):
        self.mixtures = tuple(mixtures)
        if not self.mixtures:
            raise ValueError("at least one mixture is required")
        for mixture in self.mixtures:
            validate_mixture(mixture)
        if len({mixture.mixture_id for mixture in self.mixtures}) != len(self.mixtures):
            raise ValueError("fixture sampler requires globally unique mixture_id values")
        self.manifest_digest = self._compute_manifest_digest()
        self.access_counts = {mixture.mixture_id: 0 for mixture in self.mixtures}
        self.query_access_counts = {
            query.query_id: 0
            for mixture in self.mixtures
            for query in queries_for_mixture(mixture)
        }
        self._query_to_mixture = {
            query.query_id: mixture.mixture_id
            for mixture in self.mixtures
            for query in queries_for_mixture(mixture)
        }
        self._next_index = 0

    def _compute_manifest_digest(self) -> str:
        rows = []
        for mixture in self.mixtures:
            rows.append({
                "mixture_id": mixture.mixture_id,
                "work_id": mixture.work_id,
                "movement_id": mixture.movement_id,
                "chunk_id": mixture.chunk_id,
                "split": mixture.split,
                "seating_type": mixture.seating_type,
                "mixture_path": mixture.mixture_path,
                "mixture_hash": mixture.mixture_hash,
                "sources": [
                    {
                        "source_index": source.source_index,
                        "part_id": source.part_id,
                        "timbre_id": source.timbre_id,
                        "position_id": source.position_id,
                        "path": source.path,
                        "waveform_hash": source.waveform_hash,
                    }
                    for source in mixture.sources
                ],
            })
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def iter_updates(self, num_updates: int) -> Iterator[tuple[PaperC0Mixture, tuple[PaperC0Query, ...]]]:
        if num_updates < 0:
            raise ValueError("num_updates must be non-negative")
        for _ in range(num_updates):
            mixture = self.mixtures[self._next_index % len(self.mixtures)]
            self._next_index += 1
            queries = queries_for_mixture(mixture)
            self.access_counts[mixture.mixture_id] += 1
            for query in queries:
                self.query_access_counts[query.query_id] += 1
            yield mixture, queries

    def trajectory(self, num_updates: int) -> list[str]:
        return [mixture.mixture_id for mixture, _ in self.iter_updates(num_updates)]

    def preview_trajectory(self, num_updates: int) -> list[str]:
        if num_updates < 0:
            raise ValueError("num_updates must be non-negative")
        return [self.mixtures[(self._next_index + i) % len(self.mixtures)].mixture_id for i in range(num_updates)]

    def state_dict(self) -> dict[str, object]:
        return {
            "mixture_ids": [mixture.mixture_id for mixture in self.mixtures],
            "next_index": self._next_index,
            "access_counts": dict(self.access_counts),
            "query_access_counts": dict(self.query_access_counts),
            "manifest_digest": self.manifest_digest,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {"mixture_ids", "next_index", "access_counts", "query_access_counts", "manifest_digest"}
        if set(state) != required:
            raise ValueError("sampler state fields mismatch")
        if list(state["mixture_ids"]) != [mixture.mixture_id for mixture in self.mixtures]:
            raise ValueError("sampler mixture identity mismatch")
        next_index = state["next_index"]
        if isinstance(next_index, bool) or not isinstance(next_index, int) or next_index < 0:
            raise ValueError("sampler next_index must be a non-negative integer")
        if state["manifest_digest"] != self.manifest_digest:
            raise ValueError("sampler manifest digest mismatch")
        access_counts = dict(state["access_counts"])
        query_access_counts = dict(state["query_access_counts"])
        if set(access_counts) != set(self.access_counts) or set(query_access_counts) != set(self.query_access_counts):
            raise ValueError("sampler count keys mismatch")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in access_counts.values()):
            raise ValueError("mixture access counts must be non-negative integers")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in query_access_counts.values()):
            raise ValueError("query access counts must be non-negative integers")
        if sum(access_counts.values()) != next_index:
            raise ValueError("sampler cursor and mixture access counts disagree")
        full_cycles, remainder = divmod(next_index, len(self.mixtures))
        expected_access_counts = {
            mixture.mixture_id: full_cycles + int(index < remainder)
            for index, mixture in enumerate(self.mixtures)
        }
        if access_counts != expected_access_counts:
            raise ValueError("sampler counts do not match fixed cyclic trajectory")
        for query_id, mixture_id in self._query_to_mixture.items():
            if query_access_counts[query_id] != access_counts[mixture_id]:
                raise ValueError("query and mixture access counts disagree")
        self._next_index = next_index
        self.access_counts = {str(k): int(v) for k, v in access_counts.items()}
        self.query_access_counts = {str(k): int(v) for k, v in query_access_counts.items()}


def validate_collection(mixtures: Iterable[PaperC0Mixture]) -> None:
    seen: set[str] = set()
    for mixture in mixtures:
        validate_mixture(mixture)
        if mixture.mixture_id in seen:
            raise ValueError(f"duplicate mixture_id: {mixture.mixture_id}")
        seen.add(mixture.mixture_id)
