#!/usr/bin/env python3
"""Historical train_author_release_12ep_warmup_cosine: historical 60-record dense baseline.

Portable setup only; see docs/KNOWN_LIMITATIONS.md before interpreting results.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import json
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf
from torch.utils.data import default_collate

from sqr_repro.config import load_config
from sqr_repro.release_paths import prepare_training, validate_inputs, write_launch_record
from sqr_repro.paper_c0.dataset import PARTS
from sqr_repro.paper_c0.model_factory import build_model
from sqr_repro.audio import AudioConfig, AudioTransform
from sqr_repro.models.separator import Separator
from sqr_repro.losses import packed_complex_mse


CACHE_ROOT = REPO_ROOT / "data/cache_dense_author_release"
BATCH_SIZE = 4
TARGET_EPOCHS = 12
SEED = 2026
VAL_SPLIT = "val"

PART_TO_TIMBRE = {"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}


def compute_sdr(estimate, reference):
    ref_energy = torch.sum(reference ** 2)
    err = estimate - reference
    return 10 * torch.log10(ref_energy / (torch.sum(err ** 2) + 1e-8))


class DenseCropCache:
    """In-memory cache of cached .npz records for one split."""

    def __init__(self, split: str):
        self.split = split
        self.cache_dir = CACHE_ROOT / split
        index = json.loads((CACHE_ROOT / "cache_index.json").read_text())
        self.records_meta = [r for r in index["records"] if r["split"] == split]
        if not self.records_meta:
            raise RuntimeError(f"no records in cache for split={split}")
        self.crop_samples = int(index["crop_samples"])
        # Lazy load: populate dict on first access; keeps startup time small
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _load_one(self, item_id: str):
        if item_id in self._cache:
            return self._cache[item_id]
        path = self.cache_dir / f"{item_id}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        blob = np.load(path)
        rec = {k: blob[k] for k in ("mixture", "violin1", "violin2", "viola", "cello", "position_ids")}
        rec["item_id"] = str(blob["item_id"])
        rec["composition_id"] = str(blob["composition_id"])
        rec["layout_id"] = str(blob["layout_id"])
        rec["num_segments"] = int(blob["num_segments"])
        self._cache[item_id] = rec
        return rec

    def load_all(self):
        for m in self.records_meta:
            self._load_one(m["item_id"])

    def __len__(self):
        return len(self.records_meta)


class DenseSlabSampler:
    """Yields (record_idx, segment_idx, part_idx) shuffled per epoch."""

    def __init__(self, cache: DenseCropCache, batch_size: int, seed: int):
        self.cache = cache
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        # Build full query list once
        self._base_queries = []
        for rec_idx, meta in enumerate(cache.records_meta):
            for seg in range(int(meta["num_segments"])):
                for part in range(len(PARTS)):
                    self._base_queries.append((rec_idx, seg, part))
        self._queries_per_epoch = len(self._base_queries)
        self._updates_per_epoch = self._queries_per_epoch // batch_size
        self._rebuild_queue()

    @property
    def queries_per_epoch(self):
        return self._queries_per_epoch

    @property
    def updates_per_epoch(self):
        return self._updates_per_epoch

    def _rebuild_queue(self):
        rng = random.Random(self.seed + self.epoch)
        self._queue = list(self._base_queries)
        rng.shuffle(self._queue)
        self._cursor = 0

    def next_batch(self):
        if self._cursor + self.batch_size > len(self._queue):
            self.epoch += 1
            self._rebuild_queue()
        b = self._queue[self._cursor:self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return b

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self._cursor, "seed": self.seed}


def fetch_query(cache: DenseCropCache, rec_idx: int, seg_idx: int, part_idx: int):
    meta = cache.records_meta[rec_idx]
    rec = cache._load_one(meta["item_id"])
    part = PARTS[part_idx]
    cs = cache.crop_samples
    start = seg_idx * cs
    mix = rec["mixture"][:, start:start + cs]      # (2, cs)
    tgt = rec[part][:, start:start + cs]
    timbre_id = PART_TO_TIMBRE[part]
    pos_id = int(rec["position_ids"][part_idx])
    return {
        "mixture": torch.from_numpy(np.ascontiguousarray(mix)),
        "target": torch.from_numpy(np.ascontiguousarray(tgt)),
        "item_id": rec["item_id"],
        "composition_id": rec["composition_id"],
        "layout_id": rec["layout_id"],
        "part_id": part,
        "timbre_id": torch.tensor(timbre_id),
        "position_id": torch.tensor(pos_id),
        "segment_idx": seg_idx,
    }


def inference_full_length(model, mixture, transform, timbre_id, position_id, device, num_frames, hop_length, overlap_ratio=0.5):
    C, T = mixture.shape
    seg_samples = (num_frames - 1) * hop_length
    hop_samples = int(seg_samples * (1 - overlap_ratio))
    if T < seg_samples:
        mixture = torch.nn.functional.pad(mixture, (0, seg_samples - T))
        T = mixture.shape[-1]
    output = torch.zeros(C, T, dtype=torch.float32)
    weight = torch.zeros(T, dtype=torch.float32)
    window = torch.hann_window(seg_samples, periodic=True)
    starts = list(range(0, max(T - seg_samples + 1, 1), hop_samples))
    if not starts or starts[-1] + seg_samples < T:
        starts.append(max(T - seg_samples, 0))
    for start in starts:
        end = min(start + seg_samples, T)
        chunk = mixture[:, start:end]
        if chunk.shape[-1] < seg_samples:
            chunk = torch.nn.functional.pad(chunk, (0, seg_samples - chunk.shape[-1]))
        spec = transform.stft(chunk.unsqueeze(0).to(device))
        t = torch.tensor([timbre_id], dtype=torch.long, device=device)
        p = torch.tensor([position_id], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(spec, t, p)
        wave = transform.istft(out, length=seg_samples)[0].cpu()
        w = window[:end - start]
        output[:, start:end] += wave[:, :end - start] * w
        weight[start:end] += w
    return (output / torch.clamp(weight, min=1e-8))[:, :T]


def evaluate_full_length(model, manifest_path, data_root, device, config):
    records = [json.loads(l) for l in open(manifest_path)]
    sr = config["audio"]["sample_rate"]
    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    num_frames = config["audio"]["num_frames"]
    hop_length = config["audio"]["hop_length"]
    results = []
    model.eval()
    for record in records:
        item_id = record["item_id"]
        mix, _ = sf.read(str(Path(data_root) / record["mixture"]), dtype="float32", always_2d=True)
        mixture = torch.from_numpy(mix.T.copy())
        for part_id in PARTS:
            estimate = inference_full_length(
                model, mixture, transform,
                PART_TO_TIMBRE[part_id], record["position_ids"][part_id],
                device, num_frames, hop_length,
            )
            ref, _ = sf.read(str(Path(data_root) / record["stems"][part_id]), dtype="float32", always_2d=True)
            reference = torch.from_numpy(ref.T.copy())
            min_len = min(estimate.shape[-1], reference.shape[-1])
            sdr = compute_sdr(estimate[..., :min_len], reference[..., :min_len]).item()
            results.append({
                "item_id": item_id,
                "part_id": part_id,
                "layout_id": record["layout_id"],
                "sdr": sdr,
            })
    model.train()
    return results


def main():
    global CACHE_ROOT
    config, launch = prepare_training(REPO_ROOT / "configs/author_release_12ep_warmup_cosine.yaml",
                                      repo_root=REPO_ROOT, baseline=True, custom=False)
    validate_inputs(config, launch, expected_train_records=60, baseline=True)
    if launch.check_only:
        return
    if config["training"]["batch_size"] != BATCH_SIZE or config["experiment"]["seed"] != SEED:
        raise ValueError("Historical entrypoint uses fixed batch size and seed; see script constants")
    write_launch_record(config, launch, script=__file__, runtime_checkpointing=False,
                        batch_size=BATCH_SIZE, seed=SEED, target_epochs=TARGET_EPOCHS)
    CACHE_ROOT = launch.cache_root
    run_dir = Path(config["paths"]["run_root"])
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 70)
    print(f"DENSE SLAB TRAIN: c=124, type2-only, paper-route, {TARGET_EPOCHS} epochs")
    print("=" * 70)

    train_cache = DenseCropCache("train")
    print(f"Train cache: {len(train_cache)} records, crop_samples={train_cache.crop_samples}")
    t0 = time.time()
    train_cache.load_all()
    print(f"Loaded all train .npz into RAM in {time.time()-t0:.1f}s")

    sampler = DenseSlabSampler(train_cache, BATCH_SIZE, SEED)
    total_updates = sampler.updates_per_epoch * TARGET_EPOCHS
    checkpoint_every = sampler.updates_per_epoch  # every epoch
    print(f"Sampler: {sampler.queries_per_epoch} queries/epoch, {sampler.updates_per_epoch} updates/epoch")
    print(f"Total updates for {TARGET_EPOCHS} epochs: {total_updates}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, report = build_model(config, "w24_engineering_pilot")
    model.to(device)
    model.train()
    params = report["parameters"]
    print(f"Model parameters: {params['total']:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["adam_betas"]),
        eps=float(config["training"]["adam_eps"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    # Ablation: warmup + cosine annealing scheduler
    # Warmup: linear 1e-5 -> base_lr over WARMUP_UPDATES
    # Cosine: base_lr -> FINAL_LR over remaining updates
    base_lr = float(config["training"]["learning_rate"])  # 5e-4
    WARMUP_UPDATES = 500
    FINAL_LR = 5e-6
    import math
    def lr_lambda(update):
        if update < WARMUP_UPDATES:
            # linear warmup from 1e-5 to base_lr (in relative scale)
            start_ratio = 1e-5 / base_lr  # 0.02
            return start_ratio + (1.0 - start_ratio) * (update / WARMUP_UPDATES)
        # cosine annealing
        progress = (update - WARMUP_UPDATES) / max(1, total_updates - WARMUP_UPDATES)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        min_ratio = FINAL_LR / base_lr  # 5e-6 / 5e-4 = 0.01
        return min_ratio + (1.0 - min_ratio) * cos
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"SCHEDULER: warmup_cosine (warmup={WARMUP_UPDATES} upd, final_lr={FINAL_LR})")
    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))

    print(f"\n{'='*70}\nTRAINING (dense slab, checkpoint every epoch={checkpoint_every} upd)\n{'='*70}")

    loss_history = []
    val_history = []
    start_time = time.time()
    completed = 0

    val_manifest = launch.val_manifest
    test_manifest = launch.test_manifest
    data_root = config["data"]["root"]

    while completed < total_updates:
        batch_queries = sampler.next_batch()
        batch_items = [fetch_query(train_cache, ri, si, pi) for (ri, si, pi) in batch_queries]
        batch = default_collate(batch_items)
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        timbre = batch["timbre_id"].to(device).long()
        position = batch["position_id"].to(device).long()

        mixture_spec = transform.stft(mixture)
        target_spec = transform.stft(target)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(mixture_spec, timbre, position)
        loss = packed_complex_mse(prediction, target_spec)
        if not torch.isfinite(loss):
            print(f"  WARNING: non-finite loss at update {completed+1}, skipping")
            completed += 1
            continue
        loss.backward()
        optimizer.step()
        scheduler.step()
        completed += 1

        loss_history.append({"update": completed, "loss": float(loss.detach().cpu())})

        if completed % 100 == 0:
            elapsed = time.time() - start_time
            speed = completed / elapsed if elapsed > 0 else 0
            eta_s = (total_updates - completed) / speed if speed > 0 else 0
            epoch_num = completed / sampler.updates_per_epoch
            avg_loss = np.mean([h["loss"] for h in loss_history[-100:]])
            print(f"Update {completed}/{total_updates} (epoch {epoch_num:.2f}): "
                  f"loss={float(loss):.4f} avg100={avg_loss:.4f} "
                  f"({speed:.1f} upd/s, ETA {eta_s/3600:.2f}h)", flush=True)

        if completed % checkpoint_every == 0:
            ckpt_path = run_dir / f"checkpoint_{completed:06d}.pt"
            torch.save({
                "format_version": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "completed_updates": completed,
                "sampler_state": sampler.state_dict(),
                "epoch": sampler.epoch,
            }, ckpt_path)
            print(f"  -> Checkpoint: {ckpt_path}", flush=True)

            val_results = evaluate_full_length(model, val_manifest, launch.val_data_root, device, config)
            val_sdrs = [r["sdr"] for r in val_results]
            by_layout = defaultdict(list)
            for r in val_results:
                by_layout[r["layout_id"]].append(r["sdr"])
            val_log = {
                "update": completed,
                "epoch": completed / sampler.updates_per_epoch,
                "mean_sdr": float(np.mean(val_sdrs)),
                "median_sdr": float(np.median(val_sdrs)),
                "type1_median": float(np.median(by_layout.get("type1", [float("nan")]))),
                "type2_median": float(np.median(by_layout.get("type2", [float("nan")]))),
                "n": len(val_sdrs),
            }
            val_history.append(val_log)
            print(f"  -> VAL @ ep{val_log['epoch']:.1f}: mean={val_log['mean_sdr']:+.2f} median={val_log['median_sdr']:+.2f} "
                  f"type1={val_log['type1_median']:+.2f} type2={val_log['type2_median']:+.2f}", flush=True)
            model.train()

    # Final ckpt
    final_ckpt = run_dir / "checkpoint_final.pt"
    torch.save({
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "completed_updates": completed,
        "sampler_state": sampler.state_dict(),
    }, final_ckpt)
    print(f"Final checkpoint: {final_ckpt}")

    history_path = run_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({"loss_history": loss_history, "val_history": val_history}, f)

    # Final test
    print(f"\n{'='*70}\nFINAL TEST (full-length overlap-add)\n{'='*70}")
    test_model = Separator(config).to(device)
    ckpt = torch.load(final_ckpt, map_location=device, weights_only=True)
    test_model.load_state_dict(ckpt["model_state_dict"])
    test_results = evaluate_full_length(test_model, test_manifest, launch.test_data_root, device, config)

    sdrs = [r["sdr"] for r in test_results]
    print(f"Samples: {len(sdrs)}")
    print(f"Mean  SDR: {np.mean(sdrs):+.2f} +/- {np.std(sdrs):.2f} dB")
    print(f"Median SDR: {np.median(sdrs):+.2f} dB")
    by_part = defaultdict(list)
    by_seating = defaultdict(list)
    for r in test_results:
        by_part[r["part_id"]].append(r["sdr"])
        by_seating["type1" if r["item_id"].startswith("1-") else "type2"].append(r["sdr"])
    print("\nBy part (MEDIAN):")
    for part in PARTS:
        s = by_part[part]
        print(f"  {part:8s}: median={np.median(s):+.2f}, mean={np.mean(s):+.2f}")
    print("\nBy seating (MEDIAN):")
    print(f"  type2 (same-seating):  {np.median(by_seating['type2']):+.2f} dB")
    print(f"  type1 (cross-seating): {np.median(by_seating['type1']):+.2f} dB")

    results_path = run_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nResults: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
