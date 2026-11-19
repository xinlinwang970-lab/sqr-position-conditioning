#!/usr/bin/env python3
"""Historical train_PureSwap_E1cosine-B_1000ep: custom quartet data; provenance unresolved.

Portable setup only; see docs/KNOWN_LIMITATIONS.md before interpreting results.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from sqr_repro.config import load_config
from sqr_repro.release_paths import prepare_training, validate_inputs, write_launch_record
from sqr_repro.paper_c0.dataset import PARTS, PaperC0WavDataset
from sqr_repro.paper_c0.model_factory import build_model
from sqr_repro.audio import AudioConfig, AudioTransform
from sqr_repro.models.separator import Separator
from sqr_repro.losses import packed_complex_mse


CONFIG_PATH = REPO_ROOT / "configs/PureSwap_E1cosine-B_1000ep.yaml"
TRAIN_MANIFEST = REPO_ROOT / "manifests/sqr_paper_split_train.jsonl"
VAL_MANIFEST = REPO_ROOT / "manifests/sqr_paper_split_val.jsonl"
TEST_MANIFEST = REPO_ROOT / "manifests/sqr_paper_split_test.jsonl"

BATCH_SIZE = 4
TARGET_EPOCH_SAMPLES = 236  # match E1cosine epoch size
TARGET_EPOCHS = 1000
EVAL_EVERY_EPOCHS = 25
SEED = 2026
GPU_MEM_FRACTION = 0.55
P_SWAP = 0.25
WARMUP_UPDATES = 500
FINAL_LR = 5e-6

PART_TO_TIMBRE = {"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}


class OversampleWrapper:
    """Wraps a base dataset to produce target_len samples per epoch.

    Given 12 records × 4 parts = 48 unique (record, part) pairs, cycles
    crop_slot (0,1,2,3,4) to yield 5×48=240 unique crops, then truncates
    to target_len=236. Each (record, part) appears ~4.92× per epoch with
    different random crops.
    """

    def __init__(self, base_dataset, target_len):
        self.base = base_dataset
        self.target_len = int(target_len)
        self.base_len = len(base_dataset)

    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):
        base_idx = idx % self.base_len
        self.base.crop_slot = idx // self.base_len
        return self.base[base_idx]

    def __getattr__(self, name):
        return getattr(self.base, name)


def compute_sdr(estimate, reference):
    ref_energy = torch.sum(reference ** 2)
    err = estimate - reference
    return 10 * torch.log10(ref_energy / (torch.sum(err ** 2) + 1e-8))


def inference_full_length(model, mixture, transform, timbre_id, position_id,
                          device, num_frames, hop_length, overlap_ratio=0.5):
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
        t_stft = spec.shape[2]
        pad_f = (8 - t_stft % 8) % 8
        if pad_f:
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_f))
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
    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    num_frames = config["audio"]["num_frames"]
    hop_length = config["audio"]["hop_length"]
    results = []
    model.eval()
    for record in records:
        mix, _ = sf.read(str(Path(data_root) / record["mixture"]),
                         dtype="float32", always_2d=True)
        mixture = torch.from_numpy(mix.T.copy())
        for part_id in PARTS:
            estimate = inference_full_length(
                model, mixture, transform,
                PART_TO_TIMBRE[part_id], record["position_ids"][part_id],
                device, num_frames, hop_length,
            )
            ref, _ = sf.read(str(Path(data_root) / record["stems"][part_id]),
                             dtype="float32", always_2d=True)
            reference = torch.from_numpy(ref.T.copy())
            min_len = min(estimate.shape[-1], reference.shape[-1])
            sdr = compute_sdr(estimate[..., :min_len], reference[..., :min_len]).item()
            results.append({
                "item_id": record["item_id"],
                "composition_id": record["composition_id"],
                "part_id": part_id,
                "layout_id": record["layout_id"],
                "sdr": sdr,
            })
    model.train()
    return results


def summarize(results):
    sdrs = [r["sdr"] for r in results]
    by_layout = defaultdict(list)
    for r in results:
        by_layout[r["layout_id"]].append(r["sdr"])
    return {
        "mean_sdr": float(np.mean(sdrs)),
        "median_sdr": float(np.median(sdrs)),
        "type1_median": float(np.median(by_layout.get("type1", [float("nan")]))),
        "type2_median": float(np.median(by_layout.get("type2", [float("nan")]))),
        "n": len(sdrs),
    }


def build_epoch_loader(config, crop_samples, epoch):
    base_ds = PaperC0WavDataset(
        manifest_path=TRAIN_MANIFEST,
        root=config["data"]["root"],
        split="train",
        crop_samples=crop_samples,
        crop_seed=SEED,
        epoch=epoch,
    )
    ds = OversampleWrapper(base_ds, TARGET_EPOCH_SAMPLES)
    g = torch.Generator()
    g.manual_seed(SEED + epoch)
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True, generator=g,
    )


def main():
    global TRAIN_MANIFEST, VAL_MANIFEST, TEST_MANIFEST
    config, launch = prepare_training(REPO_ROOT / "configs/PureSwap_E1cosine-B_1000ep.yaml",
                                      repo_root=REPO_ROOT, baseline=False, custom=True)
    validate_inputs(config, launch, expected_train_records=None, baseline=False)
    if launch.check_only:
        return
    if config["training"]["batch_size"] != BATCH_SIZE or config["experiment"]["seed"] != SEED:
        raise ValueError("Historical entrypoint uses fixed batch size and seed; see script constants")
    write_launch_record(config, launch, script=__file__, runtime_checkpointing=False,
                        batch_size=BATCH_SIZE, seed=SEED, target_epochs=TARGET_EPOCHS)
    TRAIN_MANIFEST, VAL_MANIFEST, TEST_MANIFEST = launch.train_manifest, launch.val_manifest, launch.test_manifest
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION, 0)

    crop_samples = (config["audio"]["num_frames"] - 1) * config["audio"]["hop_length"]
    run_dir = Path(config["paths"]["run_root"])
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    updates_per_epoch = TARGET_EPOCH_SAMPLES // BATCH_SIZE  # 59
    total_updates = updates_per_epoch * TARGET_EPOCHS
    assert total_updates <= int(config["training"]["max_attempted_updates"]), (
        total_updates, config["training"]["max_attempted_updates"])
    eval_every_updates = EVAL_EVERY_EPOCHS * updates_per_epoch

    # Count actual train/val records for logging
    train_records = [json.loads(l) for l in open(TRAIN_MANIFEST)]
    val_records = [json.loads(l) for l in open(VAL_MANIFEST)]
    train_type1 = sum(1 for r in train_records if r["layout_id"] == "type1")
    train_type2 = sum(1 for r in train_records if r["layout_id"] == "type2")
    val_type1 = sum(1 for r in val_records if r["layout_id"] == "type1")
    val_type2 = sum(1 for r in val_records if r["layout_id"] == "type2")

    print("=" * 70)
    print("PureSwap(E1cosine-B): new quartet-dataset + position-swap p=0.25 + warmup_cosine")
    print(f"  Train: {len(train_records)} records ({train_type1}x type1, {train_type2}x type2)")
    print(f"  Val:   {len(val_records)} records ({val_type1}x type1, {val_type2}x type2)")
    print(f"  Oversample: {len(train_records) * len(PARTS)} base → {TARGET_EPOCH_SAMPLES} samples/ep")
    print(f"  crop_samples={crop_samples}  batch={BATCH_SIZE}  "
          f"upd/ep={updates_per_epoch}  total_upd={total_updates}")
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
    print(f"SCHEDULER: warmup_cosine (warmup={WARMUP_UPDATES}, base={base_lr}, final={FINAL_LR})")

    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    train_data_root = config["data"]["root"]
    val_data_root = launch.val_data_root
    test_data_root = launch.test_data_root

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
        print(f"[resume] found {len(ckpts)} checkpoints in {run_dir}, "
              f"loading latest: {last.name}", flush=True)
        ckpt = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        completed = int(ckpt["completed_updates"])
        start_epoch = completed // updates_per_epoch
        print(f"[resume] completed_updates={completed} -> resume from epoch {start_epoch}", flush=True)
        scheduler.step(completed)
        print(f"[resume] scheduler advanced to step {completed}, "
              f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

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

            # --- E1 position-swap corruption ---------------------------
            if random.random() < P_SWAP:
                perm = torch.randperm(4, device=position.device)
                position = perm[position]
            # -------------------------------------------------------------

            mixture_spec = transform.stft(mixture)
            target_spec = transform.stft(target)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(mixture_spec, timbre, position)
            loss = packed_complex_mse(prediction, target_spec)
            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss at update {completed+1}, skipping", flush=True)
                completed += 1
                continue
            loss.backward()
            optimizer.step()
            scheduler.step()
            completed += 1
            loss_history.append({"update": completed, "epoch": epoch,
                                 "loss": float(loss.detach().cpu())})

            if completed % 100 == 0:
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                eta_h = (total_updates - completed) / speed / 3600 if speed > 0 else 0
                avg100 = float(np.mean([h["loss"] for h in loss_history[-100:]]))
                print(f"Update {completed}/{total_updates} (ep {epoch}): "
                      f"loss={float(loss):.4f} avg100={avg100:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"({speed:.1f} upd/s, ETA {eta_h:.1f}h)", flush=True)

            if completed % eval_every_updates == 0 or completed == total_updates:
                ckpt_path = run_dir / f"checkpoint_{completed:06d}.pt"
                torch.save({
                    "format_version": 2,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "completed_updates": completed,
                    "epoch": epoch,
                }, ckpt_path)
                print(f"  -> Checkpoint: {ckpt_path}", flush=True)
                val_results = evaluate_full_length(model, VAL_MANIFEST, val_data_root, device, config)
                val_log = {"update": completed, "epoch": epoch, **summarize(val_results)}
                val_history.append(val_log)
                print(f"  -> VAL @ ep{epoch}: mean={val_log['mean_sdr']:+.2f} "
                      f"median={val_log['median_sdr']:+.2f} "
                      f"type1={val_log['type1_median']:+.2f} "
                      f"type2={val_log['type2_median']:+.2f} (n={val_log['n']})", flush=True)
                model.train()
        if completed >= total_updates:
            break

    final_ckpt = run_dir / "checkpoint_final.pt"
    torch.save({
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "completed_updates": completed,
        "val_history": val_history,
    }, final_ckpt)
    print(f"\nFinal checkpoint: {final_ckpt}", flush=True)

    test_results = evaluate_full_length(model, TEST_MANIFEST, test_data_root, device, config)
    test_log = summarize(test_results)
    print(f"\nTEST @ final: mean={test_log['mean_sdr']:+.2f} median={test_log['median_sdr']:+.2f}")
    print(f"  type1 median: {test_log['type1_median']:+.2f}")
    print(f"  type2 median: {test_log['type2_median']:+.2f}")
    print(f"  n = {test_log['n']}")

    with open(run_dir / "test_results.json", "w") as f:
        json.dump({"final": test_log, "by_track": [
            {"composition_id": r["composition_id"], "part_id": r["part_id"],
             "layout_id": r["layout_id"], "sdr": r["sdr"]}
            for r in test_results
        ]}, f, indent=2)
    print(f"Results written to {run_dir}/test_results.json")


if __name__ == "__main__":
    main()
