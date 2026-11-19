"""Real-I/O W24 engineering pilot entry, bounded and fail-closed."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .dataset import PaperC0WavDataset
from .model_factory import build_model
from .trainer import PaperC0SyntheticTrainer, PaperC0TrainerError
from ..paper_c0_contract import (
    PaperC0Mixture,
    PaperC0Source,
    SeatingMappingEvidence,
    validate_formal_train_collection,
    validate_verified_seating_mapping,
)


class RealPilotError(PaperC0TrainerError):
    pass


class RealPilotSampler:
    """Deterministic query sampler for real four-query records; max 20 updates is enforced by the trainer."""

    def __init__(self, dataset_length: int, batch_size: int):
        if dataset_length <= 0 or batch_size <= 0:
            raise RealPilotError("real pilot dataset and batch size must be positive")
        self.dataset_length = int(dataset_length)
        self.batch_size = int(batch_size)
        self.cursor = 0

    def preview_batch(self) -> list[int]:
        base = self.cursor * self.batch_size
        return [(base + offset) % self.dataset_length for offset in range(self.batch_size)]

    def commit(self) -> None:
        self.cursor += 1

    def state_dict(self) -> dict[str, int]:
        return {"dataset_length": self.dataset_length, "batch_size": self.batch_size, "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"dataset_length", "batch_size", "cursor"}:
            raise RealPilotError("real pilot sampler state fields mismatch")
        if int(state["dataset_length"]) != self.dataset_length or int(state["batch_size"]) != self.batch_size:
            raise RealPilotError("real pilot sampler identity mismatch")
        if isinstance(state["cursor"], bool) or not isinstance(state["cursor"], int) or state["cursor"] < 0:
            raise RealPilotError("real pilot sampler cursor must be non-negative integer")
        self.cursor = state["cursor"]


def _record_to_mixture(record: dict[str, Any]) -> PaperC0Mixture:
    layout = str(record.get("layout_id", ""))
    if not layout.lower().startswith("type"):
        raise RealPilotError("real manifest requires explicit layout_id/type seating")
    seating_type = int(layout.lower().removeprefix("type"))
    try:
        sources = tuple(
            PaperC0Source(
                source_index=index,
                part_id=part,
                timbre_id={"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}[part],
                position_id=int(record["position_ids"][part]),
                path=str(record["stems"][part]),
                waveform_hash=str(record["stem_hashes"][part]),
            )
            for index, part in enumerate(("violin1", "violin2", "viola", "cello"))
        )
        return PaperC0Mixture(
            mixture_id=str(record["item_id"]),
            work_id=str(record.get("work_id", record["composition_id"])),
            movement_id=str(record.get("movement_id", record["performance_id"])),
            chunk_id=str(record.get("chunk_id", record["item_id"])),
            seating_type=seating_type,
            split=str(record["split"]),
            mixture_path=str(record["mixture"]),
            mixture_hash=str(record["mixture_hash"]),
            sources=sources,
        )
    except KeyError as exc:
        raise RealPilotError(f"real manifest lacks complete source identity: {exc}") from exc


def validate_real_pilot_dataset(
    dataset,
    mapping_evidence: Mapping[int, SeatingMappingEvidence],
    trusted_sources: Mapping[str, str],
    allow_test_audio: bool = False,
) -> tuple[PaperC0Mixture, ...]:
    records = tuple(dataset.records)
    if not records:
        raise RealPilotError("real pilot dataset is empty")
    if any(bool(record.get("is_fixture")) for record in records) and not allow_test_audio:
        raise RealPilotError("synthetic fixture cannot enter real-I/O pilot")
    mixtures = tuple(_record_to_mixture(record) for record in records)
    validate_formal_train_collection(mixtures, mapping_evidence, trusted_sources)
    for mixture in mixtures:
        validate_verified_seating_mapping(mixture, mapping_evidence[mixture.seating_type], trusted_sources)
    return mixtures


class RealPilotTrainer(PaperC0SyntheticTrainer):
    """W24 engineering-only trainer using real WAV Dataset and the bounded update/checkpoint path."""

    def __init__(
        self,
        config: dict[str, Any],
        dataset,
        mapping_evidence: Mapping[int, SeatingMappingEvidence],
        trusted_sources: Mapping[str, str],
        device: str = "cpu",
        allow_test_audio: bool = False,
    ):
        if config.get("experiment", {}).get("run_mode") != "w24_engineering_pilot":
            raise RealPilotError("real pilot requires run_mode=w24_engineering_pilot")
        if int(config["training"].get("max_attempted_updates", 0)) > 20:
            raise RealPilotError("real pilot is bounded at 20 attempted updates")
        self.real_mixtures = validate_real_pilot_dataset(dataset, mapping_evidence, trusted_sources, allow_test_audio)
        super().__init__(
            config,
            dataset,
            device=device,
            run_mode="w24_engineering_pilot",
            sampler=RealPilotSampler(len(dataset), int(config["training"]["batch_size"])),
        )

    def _make_sampler(self):
        return RealPilotSampler(len(self.dataset), int(self.config["training"]["batch_size"]))


def build_real_pilot_dataset(config: dict[str, Any], manifest_path: str | Path, data_root: str | Path):
    return PaperC0WavDataset(
        manifest_path,
        data_root,
        "train",
        crop_samples=int(config["audio"]["segment_samples"]),
        crop_seed=int(config["experiment"].get("data_seed", config["experiment"].get("seed", 2026))),
        require_fixture=False,
    )
