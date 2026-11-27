#!/usr/bin/env python3
"""E1cosine-B1: PureSwap on split quartet-dataset (60 records, no oversampling).

Same as E1cosine but with quartet-data1 (12 movements → 60 equal segments):
  - Train: /mnt/sda/MSS/quartet-data1/train/train_manifest.jsonl (60 type2 records)
  - 60 records × 4 parts = 240 samples/ep, batch=4 → 60 upd/ep
  - No oversampling needed — natural alignment with E1cosine structure.

All else identical: P_SWAP=0.25, warmup_cosine LR, 1000 epochs.
"""

import sys

sys.path.insert(0, "/mnt/sda/MSS/sqr-position-repro/src")

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sqr_repro.config import load_config
from sqr_repro.paper_c0.dataset import PARTS, PaperC0WavDataset
from sqr_repro.paper_c0.model_factory import build_model
from sqr_repro.audio import AudioConfig, AudioTransform
from sqr_repro.evaluation import (
    evaluate_full_length_legacy_v1 as evaluate_full_length,
    summarize_wavesdr as summarize,
)
from sqr_repro.models.separator import Separator
from sqr_repro.losses import packed_complex_mse


CONFIG_PATH = "/mnt/sda/MSS/sqr-position-repro/configs/PureSwap_E1cosine-B1_1000ep.yaml"
TRAIN_MANIFEST = "/mnt/sda/MSS/quartet-data1/train/train_manifest.jsonl"
VAL_MANIFEST = "/mnt/sda/MSS/quartet-data/Val/val_manifest.jsonl"
TEST_MANIFEST = "/mnt/sda/MSS/data/manifests/sqr_paper_split_test.jsonl"

BATCH_SIZE = 4
TARGET_EPOCHS = 1000
EVAL_EVERY_EPOCHS = 25
SEED = 2026
GPU_MEM_FRACTION = 0.55
P_SWAP = 0.25
WARMUP_UPDATES = 500
FINAL_LR = 5e-6











def build_epoch_loader(config, crop_samples, epoch):
    ds = PaperC0WavDataset(
        manifest_path=TRAIN_MANIFEST,
        root=config["data"]["root"],
        split="train",
        crop_samples=crop_samples,
        crop_seed=SEED,
        epoch=epoch,
    )
    g = torch.Generator()
    g.manual_seed(SEED + epoch)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=g,
    )


def main():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION, 0)

    config = load_config(CONFIG_PATH)
    crop_samples = (config["audio"]["num_frames"] - 1) * config["audio"]["hop_length"]
    run_dir = Path(config["paths"]["run_root"])
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    updates_per_epoch = (
        len([json.loads(l) for l in open(TRAIN_MANIFEST)]) * len(PARTS)
    ) // BATCH_SIZE  # 60
    total_updates = updates_per_epoch * TARGET_EPOCHS
    assert total_updates <= int(config["training"]["max_attempted_updates"]), (
        total_updates,
        config["training"]["max_attempted_updates"],
    )
    eval_every_updates = EVAL_EVERY_EPOCHS * updates_per_epoch

    train_records = [json.loads(l) for l in open(TRAIN_MANIFEST)]
    val_records = [json.loads(l) for l in open(VAL_MANIFEST)]
    train_type1 = sum(1 for r in train_records if r["layout_id"] == "type1")
    train_type2 = sum(1 for r in train_records if r["layout_id"] == "type2")
    val_type1 = sum(1 for r in val_records if r["layout_id"] == "type1")
    val_type2 = sum(1 for r in val_records if r["layout_id"] == "type2")

    print("=" * 70)
    print(
        "PureSwap(E1cosine-B1): split quartet-dataset + position-swap p=0.25 + warmup_cosine"
    )
    print(
        f"  Train: {len(train_records)} records ({train_type1}x type1, {train_type2}x type2)"
    )
    print(
        f"  Val:   {len(val_records)} records ({val_type1}x type1, {val_type2}x type2)"
    )
    print(f"  No oversampling — natural 240 samples/ep")
    print(
        f"  crop_samples={crop_samples}  batch={BATCH_SIZE}  "
        f"upd/ep={updates_per_epoch}  total_upd={total_updates}"
    )
    print(f"  eval every {EVAL_EVERY_EPOCHS} ep = {eval_every_updates} upd")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, report = build_model(config, "w24_engineering_pilot")
    model.to(device)
    model.train()
    print(f"Model parameters: {report['parameters']['total']:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["adam_betas"]),
        eps=float(config["training"]["adam_eps"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    base_lr = float(config["training"]["learning_rate"])

    def lr_lambda(update):
        if update < WARMUP_UPDATES:
            start_ratio = 1e-5 / base_lr
            return start_ratio + (1.0 - start_ratio) * (update / WARMUP_UPDATES)
        progress = (update - WARMUP_UPDATES) / max(1, total_updates - WARMUP_UPDATES)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        min_ratio = FINAL_LR / base_lr
        return min_ratio + (1.0 - min_ratio) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(
        f"SCHEDULER: warmup_cosine (warmup={WARMUP_UPDATES}, base={base_lr}, final={FINAL_LR})"
    )

    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    train_data_root = config["data"]["root"]
    val_data_root = "/mnt/sda/MSS/quartet-data/Val"
    test_data_root = "/mnt/sda/MSS/quartet-data"

    loss_history = []
    val_history = []
    completed = 0
    start_epoch = 0

    ckpts = sorted(
        (p for p in run_dir.glob("checkpoint_*.pt") if p.stem.split("_")[1].isdigit()),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if ckpts:
        last = ckpts[-1]
        print(
            f"[resume] found {len(ckpts)} checkpoints in {run_dir}, "
            f"loading latest: {last.name}",
            flush=True,
        )
        ckpt = torch.load(last, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        completed = int(ckpt["completed_updates"])
        start_epoch = completed // updates_per_epoch
        print(
            f"[resume] completed_updates={completed} -> resume from epoch {start_epoch}",
            flush=True,
        )
        scheduler.step(completed)
        print(
            f"[resume] scheduler advanced to step {completed}, "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

    start_time = time.time()

    for epoch in range(start_epoch, TARGET_EPOCHS):
        loader = build_epoch_loader(config, crop_samples, epoch)
        for batch in loader:
            if completed >= total_updates:
                break
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            timbre = batch["timbre_id"].to(device).long()
            position = batch["position_id"].to(device).long()

            if random.random() < P_SWAP:
                perm = torch.randperm(4, device=position.device)
                position = perm[position]

            mixture_spec = transform.stft(mixture)
            target_spec = transform.stft(target)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(mixture_spec, timbre, position)
            loss = packed_complex_mse(prediction, target_spec)
            if not torch.isfinite(loss):
                print(
                    f"  WARNING: non-finite loss at update {completed + 1}, skipping",
                    flush=True,
                )
                completed += 1
                continue
            loss.backward()
            optimizer.step()
            scheduler.step()
            completed += 1
            loss_history.append(
                {
                    "update": completed,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                }
            )

            if completed % 100 == 0:
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                eta_h = (total_updates - completed) / speed / 3600 if speed > 0 else 0
                avg100 = float(np.mean([h["loss"] for h in loss_history[-100:]]))
                print(
                    f"Update {completed}/{total_updates} (ep {epoch}): "
                    f"loss={float(loss):.4f} avg100={avg100:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"({speed:.1f} upd/s, ETA {eta_h:.1f}h)",
                    flush=True,
                )

            if completed % eval_every_updates == 0 or completed == total_updates:
                ckpt_path = run_dir / f"checkpoint_{completed:06d}.pt"
                torch.save(
                    {
                        "format_version": 2,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "completed_updates": completed,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )
                print(f"  -> Checkpoint: {ckpt_path}", flush=True)
                val_results = evaluate_full_length(
                    model, VAL_MANIFEST, val_data_root, device, config
                )
                val_log = {
                    "update": completed,
                    "epoch": epoch,
                    **summarize(val_results),
                }
                val_history.append(val_log)
                print(
                    f"  -> VAL @ ep{epoch}: mean={val_log['mean_sdr']:+.2f} "
                    f"median={val_log['median_sdr']:+.2f} "
                    f"type1={val_log['type1_median']:+.2f} "
                    f"type2={val_log['type2_median']:+.2f} (n={val_log['n']})",
                    flush=True,
                )
                model.train()
        if completed >= total_updates:
            break

    final_ckpt = run_dir / "checkpoint_final.pt"
    torch.save(
        {
            "format_version": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "completed_updates": completed,
            "val_history": val_history,
        },
        final_ckpt,
    )
    print(f"\nFinal checkpoint: {final_ckpt}", flush=True)

    test_results = evaluate_full_length(
        model, TEST_MANIFEST, test_data_root, device, config
    )
    test_log = summarize(test_results)
    print(
        f"\nTEST @ final: mean={test_log['mean_sdr']:+.2f} median={test_log['median_sdr']:+.2f}"
    )
    print(f"  type1 median: {test_log['type1_median']:+.2f}")
    print(f"  type2 median: {test_log['type2_median']:+.2f}")
    print(f"  n = {test_log['n']}")

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(
            {
                "final": test_log,
                "by_track": [
                    {
                        "composition_id": r["composition_id"],
                        "part_id": r["part_id"],
                        "layout_id": r["layout_id"],
                        "sdr": r["sdr"],
                    }
                    for r in test_results
                ],
            },
            f,
            indent=2,
        )
    print(f"Results written to {run_dir}/test_results.json")


if __name__ == "__main__":
    main()
