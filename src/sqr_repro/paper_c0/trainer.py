"""Bounded synthetic Paper C0 trainer for R2.2 infrastructure validation."""

from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import default_collate

from ..audio import AudioConfig, AudioTransform
from ..checkpoint import capture_rng_state, restore_rng_state, save_training_checkpoint
from ..losses import packed_complex_mse
from .model_factory import ModelFactoryError, build_model


class PaperC0TrainerError(RuntimeError):
    pass


class SyntheticQuerySampler:
    """Deterministic sequential sampler with resumable cursor."""

    def __init__(self, dataset_length: int, batch_size: int):
        if dataset_length <= 0 or batch_size <= 0:
            raise ValueError("dataset_length and batch_size must be positive")
        self.dataset_length = int(dataset_length)
        self.batch_size = int(batch_size)
        self.cursor = 0

    def preview_batch(self) -> list[int]:
        base = self.cursor * self.batch_size
        return [(base + offset) % self.dataset_length for offset in range(self.batch_size)]

    def commit(self) -> None:
        self.cursor += 1

    def state_dict(self) -> dict[str, int]:
        return {
            "dataset_length": self.dataset_length,
            "batch_size": self.batch_size,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"dataset_length", "batch_size", "cursor"}:
            raise PaperC0TrainerError("synthetic sampler state fields mismatch")
        if int(state["dataset_length"]) != self.dataset_length or int(state["batch_size"]) != self.batch_size:
            raise PaperC0TrainerError("synthetic sampler identity mismatch")
        cursor = state["cursor"]
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise PaperC0TrainerError("synthetic sampler cursor must be non-negative integer")
        self.cursor = cursor


def _dataset_digest(dataset) -> str:
    records = []
    for record in dataset.records:
        records.append({
            "item_id": record.get("item_id"),
            "work_id": record.get("work_id", record.get("composition_id")),
            "movement_id": record.get("movement_id", record.get("performance_id")),
            "chunk_id": record.get("chunk_id", record.get("item_id")),
            "composition_id": record.get("composition_id"),
            "performance_id": record.get("performance_id"),
            "split": record.get("split"),
            "seating_type": record.get("seating_type"),
            "layout_id": record.get("layout_id"),
            "sample_rate": record.get("sample_rate"),
            "num_samples": record.get("num_samples"),
            "mixture": record.get("mixture"),
            "mixture_hash": record.get("mixture_hash"),
            "stems": record.get("stems"),
            "stem_hashes": record.get("stem_hashes"),
            "position_ids": record.get("position_ids"),
            "parts": [
                {
                    "part_id": part,
                    "source_index": index,
                    "timbre_id": {"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}[part],
                    "position_id": record.get("position_ids", {}).get(part),
                    "path": record.get("stems", {}).get(part),
                    "hash": record.get("stem_hashes", {}).get(part),
                }
                for index, part in enumerate(("violin1", "violin2", "viola", "cello"))
            ],
            "provenance": record.get("provenance", record.get("source_provenance")),
        })
    payload = {
        "manifest_path": str(dataset.manifest_path.resolve()),
        "root": str(dataset.root.resolve()),
        "split": dataset.split,
        "crop_samples": dataset.crop_samples,
        "crop_seed": dataset.crop_seed,
        "all_targets": dataset.all_targets,
        "records": records,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


class PaperC0SyntheticTrainer:
    def __init__(self, config: dict[str, Any], dataset, device: str = "cpu", run_mode: str = "synthetic_smoke", sampler=None):
        if config.get("experiment", {}).get("run_mode") != run_mode:
            raise PaperC0TrainerError(f"trainer run mode mismatch: expected {run_mode}")
        if run_mode not in {"synthetic_smoke", "w24_engineering_pilot"}:
            raise PaperC0TrainerError("bounded trainer only accepts synthetic_smoke or w24_engineering_pilot")
        if int(config["training"].get("gradient_accumulation_steps", 1)) != 1:
            raise PaperC0TrainerError("R2.2 synthetic trainer requires gradient_accumulation_steps=1")
        if str(config["training"].get("precision", "fp32")) != "fp32":
            raise PaperC0TrainerError("R2.2 synthetic trainer requires FP32")
        if device == "cuda" and not torch.cuda.is_available():
            raise PaperC0TrainerError("CUDA requested but unavailable")
        self.config = config
        self.dataset = dataset
        self.device = device
        self.model, self.model_report = build_model(config, run_mode)
        self.model.to(device)
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            betas=tuple(config["training"]["adam_betas"]),
            eps=float(config["training"]["adam_eps"]),
            weight_decay=float(config["training"]["weight_decay"]),
            foreach=bool(config["training"]["foreach"]),
        )
        self.transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
        self.sampler = sampler or SyntheticQuerySampler(len(dataset), int(config["training"]["batch_size"]))
        self.dataset_digest = _dataset_digest(dataset)
        self.completed_updates = 0
        self.attempted_updates = 0
        self.max_attempted_updates = int(config["training"].get("max_attempted_updates", 20))
        self.failed = False

    def _make_sampler(self):
        return SyntheticQuerySampler(len(self.dataset), int(self.config["training"]["batch_size"]))

    def _batch(self) -> dict[str, Any]:
        self.dataset.epoch = int(self.completed_updates)
        self.dataset.crop_slot = int(self.sampler.cursor)
        return default_collate([self.dataset[index] for index in self.sampler.preview_batch()])

    def update(self) -> dict[str, Any]:
        if self.failed:
            raise PaperC0TrainerError("trainer is fail-stopped after a failed update")
        if self.attempted_updates >= self.max_attempted_updates:
            raise PaperC0TrainerError("maximum attempted update budget reached")
        consumed_indices = self.sampler.preview_batch()
        self.attempted_updates += 1
        try:
            batch = self._batch()
            mixture = batch["mixture"].to(self.device)
            target = batch["target"].to(self.device)
            timbre = batch["timbre_id"].to(self.device).long()
            position = batch["position_id"].to(self.device).long()
            mixture_spec = self.transform.stft(mixture)
            target_spec = self.transform.stft(target)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(mixture_spec, timbre, position)
            loss = packed_complex_mse(prediction, target_spec)
            if not torch.isfinite(loss):
                raise PaperC0TrainerError("non-finite loss")
            loss.backward()
            gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise PaperC0TrainerError("non-finite or missing gradients")
            self.optimizer.step()
        except Exception:
            self.failed = True
            raise
        self.sampler.commit()
        self.completed_updates += 1
        return {
            "completed_updates": self.completed_updates,
            "attempted_updates": self.attempted_updates,
            "loss": float(loss.detach().cpu()),
            "batch_indices": consumed_indices,
            "crop_offsets": [int(batch["crop_offset"][index]) for index in range(len(batch["crop_offset"]))],
            "query_ids": [
                f"{item}::{part}::source{source}"
                for item, part, source in zip(batch["item_id"], batch["part_id"], batch["target_source_index"])
            ],
            "optimizer_updates": self.completed_updates,
        }

    def train_updates(self, updates: int) -> list[dict[str, Any]]:
        if updates < 1:
            raise ValueError("updates must be positive")
        return [self.update() for _ in range(updates)]

    def save(self, path: str | Path) -> None:
        self.dataset.epoch = int(self.completed_updates)
        self.dataset.crop_slot = int(self.sampler.cursor)
        save_training_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.config,
            self.completed_updates,
            capture_rng_state(),
            self.sampler.state_dict(),
            metadata={
                "trainer": type(self).__name__,
                "model_report": self.model_report,
                "dataset_digest": self.dataset_digest,
                "device": self.device,
                "architecture_identity": self.model_report["architecture_version"],
                "dataset_digest": self.dataset_digest,
                "dataset_runtime_state": {"epoch": int(self.dataset.epoch), "crop_slot": int(self.dataset.crop_slot)},
            },
            attempted_updates=self.attempted_updates,
            max_attempted_updates=self.max_attempted_updates,
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        required = {
            "format_version", "model_state_dict", "optimizer_state_dict", "config", "completed_updates",
            "attempted_updates", "max_attempted_updates", "rng_state", "sampler_state", "metadata",
        }
        if set(checkpoint) < required:
            raise PaperC0TrainerError("checkpoint missing trainer state fields")
        if checkpoint["format_version"] != 2 or checkpoint["config"] != self.config:
            raise PaperC0TrainerError("checkpoint format/config mismatch")
        metadata = checkpoint["metadata"]
        if metadata.get("dataset_digest") != self.dataset_digest:
            raise PaperC0TrainerError("checkpoint dataset digest mismatch")
        if metadata.get("architecture_identity") != self.model_report["architecture_version"]:
            raise PaperC0TrainerError("checkpoint architecture identity mismatch")
        runtime_state = metadata.get("dataset_runtime_state")
        if not isinstance(runtime_state, dict) or set(runtime_state) != {"epoch", "crop_slot"}:
            raise PaperC0TrainerError("checkpoint crop state is missing or malformed")
        completed = checkpoint["completed_updates"]
        attempted = checkpoint["attempted_updates"]
        max_attempted = checkpoint["max_attempted_updates"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (completed, attempted, max_attempted)):
            raise PaperC0TrainerError("checkpoint update counters are invalid")
        if completed > attempted or attempted > max_attempted:
            raise PaperC0TrainerError("checkpoint update counter ordering is invalid")
        if checkpoint["sampler_state"].get("cursor") != completed:
            raise PaperC0TrainerError("checkpoint sampler/completed update mismatch")
        if runtime_state["epoch"] != completed or runtime_state["crop_slot"] != completed:
            raise PaperC0TrainerError("checkpoint crop state does not match completed updates")
        # Stage model, optimizer and sampler before changing the live trainer.
        staged_model = copy.deepcopy(self.model)
        staged_optimizer = torch.optim.Adam(
            staged_model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            betas=tuple(self.config["training"]["adam_betas"]),
            eps=float(self.config["training"]["adam_eps"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
            foreach=bool(self.config["training"]["foreach"]),
        )
        staged_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        staged_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        staged_sampler = self._make_sampler()
        staged_sampler.load_state_dict(checkpoint["sampler_state"])
        current_rng = capture_rng_state()
        try:
            restore_rng_state(checkpoint["rng_state"])
        except Exception as exc:
            restore_rng_state(current_rng)
            raise PaperC0TrainerError("checkpoint RNG state is invalid") from exc
        restore_rng_state(current_rng)
        self.model.load_state_dict(staged_model.state_dict(), strict=True)
        self.optimizer.load_state_dict(staged_optimizer.state_dict())
        self.sampler.load_state_dict(staged_sampler.state_dict())
        restore_rng_state(checkpoint["rng_state"])
        self.completed_updates = completed
        self.attempted_updates = attempted
        self.max_attempted_updates = max_attempted
        self.dataset.epoch = int(runtime_state["epoch"])
        self.dataset.crop_slot = int(runtime_state["crop_slot"])
        self.failed = False
        self.model.train()
        return checkpoint
