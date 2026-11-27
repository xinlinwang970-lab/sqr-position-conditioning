#!/usr/bin/env python3
"""Growth24 candidate B: 1000-epoch real-data plain training (no intervention).

Protocol (matches the candidate-A paper protocol except no anti-shortcut
intervention):
  - train manifest: 59 type2 StringQuD records (sqr_paper_split_train.jsonl)
  - model: growth24_projected_b_v1 (split_stage_paper route, stage widths
    96/144/144/240, growth=24, controller H=256) -- 46,017,316 trainable params
  - optimizer: Adam, warmup_cosine LR (500 upd warmup -> cosine 5e-4 -> 5e-6)
  - dropout 0.1, batch 4, 59 upd/ep (= 59 records * 4 parts / batch 4), 1000 ep
  - val: 6-paper-works split (type1 + type2), eval every 25 ep
  - test: URMP 6 works, FINAL-ONLY at end
  - auto-resume: picks up from latest checkpoint_<step>.pt in run_root

Run entirely from github_release with local data inside /ceph/leon/MSS/data.
"""

import sys

sys.path.insert(0, "/ceph/leon/MSS/github_release/src")

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


CONFIG_PATH = (
    "/ceph/leon/MSS/github_release/configs/growth24_projected_b_v1_mdp_constlr_train.yaml"
)
DATA_ROOT = "/ceph/leon/MSS/data"
TRAIN_MANIFEST = str(Path(DATA_ROOT) / "manifests/sqr_paper_split_train.jsonl")
VAL_MANIFEST = str(Path(DATA_ROOT) / "manifests/sqr_paper_split_val.jsonl")
TEST_MANIFEST = str(Path(DATA_ROOT) / "manifests/sqr_paper_split_test.jsonl")

BATCH_SIZE = 4
TARGET_EPOCHS = 1000
EVAL_EVERY_EPOCHS = 25
SEED = 2026
WARMUP_UPDATES = 500
FINAL_LR = 5e-6











def build_epoch_loader(config, crop_samples, epoch):
    """Re-instantiate PaperC0WavDataset for the given epoch -> new random crops."""
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


def apply_activation_checkpointing(model):
    """Wrap every heavy block of Separator with gradient checkpointing."""
    import functools

    import torch.utils.checkpoint as checkpoint

    wrapped = 0
    for attr in ("encoders", "decoders"):
        container = getattr(model, attr, None)
        if container is None:
            continue
        for block in container:
            original_forward = block.forward

            @functools.wraps(original_forward)
            def checkpointed_forward(*args, _orig=original_forward, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint.checkpoint(
                        _orig, *args, use_reentrant=False, **kwargs
                    )
                return _orig(*args, **kwargs)

            block.forward = checkpointed_forward
            wrapped += 1
    middle = getattr(model, "middle", None)
    if middle is not None:
        original_forward = middle.forward

        @functools.wraps(original_forward)
        def checkpointed_middle(*args, _orig=original_forward, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint.checkpoint(
                    _orig, *args, use_reentrant=False, **kwargs
                )
            return _orig(*args, **kwargs)

        middle.forward = checkpointed_middle
        wrapped += 1
    print(f"[info] activation_checkpointing wrapped {wrapped} blocks", flush=True)


def main():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(1.0, 0)

    config = load_config(CONFIG_PATH)
    crop_samples = (config["audio"]["num_frames"] - 1) * config["audio"]["hop_length"]
    run_dir = Path(config["paths"]["run_root"])
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    updates_per_epoch = (59 * len(PARTS)) // BATCH_SIZE  # 59
    total_updates = updates_per_epoch * TARGET_EPOCHS
    assert total_updates <= int(config["training"]["max_attempted_updates"]), (
        total_updates,
        config["training"]["max_attempted_updates"],
    )
    eval_every_updates = EVAL_EVERY_EPOCHS * updates_per_epoch

    print("=" * 70)
    print(
        "Growth24 Candidate B: 46M projected, widths 96/144/144/240 growth=24 H=256, PLAIN"
    )
    print(
        f"  crop_samples={crop_samples}  batch={BATCH_SIZE}  "
        f"upd/ep={updates_per_epoch}  total_upd={total_updates}"
    )
    print(f"  eval every {EVAL_EVERY_EPOCHS} ep = {eval_every_updates} upd")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, report = build_model(config, "w24_engineering_pilot")
    model.to(device)
    apply_activation_checkpointing(model)
    model.train()
    print(f"Model parameters: {report['parameters']['total']:,}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["adam_betas"]),
        eps=float(config["training"]["adam_eps"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    # Constant LR: no scheduler (paper-strict §4.2)
    print(
        f"SCHEDULER: constant (lr={config['training']['learning_rate']}, paper-strict §4.2)",
        flush=True,
    )

    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    data_root = config["data"]["root"]

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
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        completed = int(ckpt["completed_updates"])
        start_epoch = completed // updates_per_epoch
        print(
            f"[resume] completed_updates={completed} -> resume from epoch {start_epoch}",
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
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
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
                    model, VAL_MANIFEST, data_root, device, config
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

    test_results = evaluate_full_length(model, TEST_MANIFEST, data_root, device, config)
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
    with open(run_dir / "train_curves.jsonl", "w") as f:
        for row in loss_history:
            f.write(json.dumps(row) + "\n")
    with open(run_dir / "val_history.json", "w") as f:
        json.dump(val_history, f, indent=2)


if __name__ == "__main__":
    main()
