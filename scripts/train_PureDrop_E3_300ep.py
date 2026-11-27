#!/usr/bin/env python3
"""E3 (PureDrop, p=0.25): missing position hint via zeroing of e_p.

E1-E4 anti-shortcut matrix cell E3: "提示可靠性退化"谱系中的缺失情形 —
inside ConditionEmbeddings, zero the position embedding with p=0.25 per sample
(train-mode only); e_m = 0.5*e_t on dropped samples, forcing a no-hint fallback
mode. Tests whether absence-of-hint breaks the GPoCM lookup-table shortcut.

Ablation:
  - train manifest: 59 type2 (sqr_paper_split_train.jsonl, NO type1 contamination)
  - model: mid_dual_paper route (encoder bypass GPoCM; e_m feeds decoder only)
  - optimizer: Adam CONSTANT lr=5e-4 (same protocol as B0/E1; the ONLY variable
    is the position-corruption mechanism: noise vs swap vs drop)
  - dropout 0.1, batch 4, 59 upd/ep, 300 ep screening horizon (= 17700 upd)
  - corruption: per-sample e_p gaussian noise, sigma=0.1, train-mode only;
    val/test evaluate with clean embeddings
  - val: 6-paper-works split (type1 + type2), eval every 25 ep
  - test: URMP 6 works, run at end of the 300ep screening
  - auto-resume: picks up from latest checkpoint_<step>.pt in run_root

Run entirely from /ceph/leon/MSS with local data split manifests.
"""

import sys

sys.path.insert(0, "/ceph/leon/MSS/sqr-position-repro/src")

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
    "/ceph/leon/MSS/sqr-position-repro/configs/PureDrop_E3_300ep_p025.yaml"
)
TRAIN_MANIFEST = "/ceph/leon/MSS/data/manifests/sqr_paper_split_train.jsonl"
VAL_MANIFEST = "/ceph/leon/MSS/data/manifests/sqr_paper_split_val.jsonl"
TEST_MANIFEST = "/ceph/leon/MSS/data/manifests/sqr_paper_split_test.jsonl"

BATCH_SIZE = 4
TARGET_EPOCHS = 300  # 300ep screening horizon per E1-E4 matrix promotion gate
EVAL_EVERY_EPOCHS = 25
SEED = 2026
GPU_MEM_FRACTION = 1.0  # idle GPU: full 32GB budget allowed
CORRUPT_MODE = "drop"  # E3: zero-out position embedding e_p
NOISE_STD = None
DROP_PROB = 0.25  # E3: per-sample drop probability











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
    """Wrap every heavy block of Separator with gradient checkpointing.

    Math-equivalent recompute-to-save-memory: stores only block inputs and
    recomputes block internals in backward. Required on 32GB V100S for the
    fp32 c124 model (full retained activations are ~28 GiB).
    Note: BatchNorm running stats get one extra momentum update per step
    (recompute runs in train mode); identical condition for the whole run.
    """
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


def apply_position_corruption(model, mode, noise_std=None, drop_prob=None):
    """Train-time corruption of position embedding e_p (E2 noise / E3 drop).

    Patches model.embeddings (ConditionEmbeddings): in train mode only,
    per-sample corruption of e_p, then rebuilds e_m = 0.5*(e_t+e_p) so the
    mid_dual_paper route consumes the corrupted hint. Eval/val/test run with
    clean embeddings (module training flag is False under model.eval()).
    """
    import functools

    emb = model.embeddings
    original_forward = emb.forward

    @functools.wraps(original_forward)
    def corrupted_forward(timbre_id, position_id):
        e_t, e_m, e_p = original_forward(timbre_id, position_id)
        if emb.training and torch.is_grad_enabled():
            if mode == "noise":
                e_p = e_p + float(noise_std) * torch.randn_like(e_p)
            elif mode == "drop":
                keep = (
                    torch.rand(e_p.shape[0], 1, device=e_p.device) >= float(drop_prob)
                ).to(e_p.dtype)
                e_p = e_p * keep
            else:
                raise ValueError(f"unknown corrupt mode: {mode}")
            e_m = 0.5 * (e_t + e_p)
        return e_t, e_m, e_p

    emb.forward = corrupted_forward
    print(
        f"[info] position corruption: mode={mode} noise_std={noise_std} "
        f"drop_prob={drop_prob} (train-only, per-sample)",
        flush=True,
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

    updates_per_epoch = (59 * len(PARTS)) // BATCH_SIZE  # 59
    total_updates = updates_per_epoch * TARGET_EPOCHS
    assert total_updates <= int(config["training"]["max_attempted_updates"]), (
        total_updates,
        config["training"]["max_attempted_updates"],
    )
    eval_every_updates = EVAL_EVERY_EPOCHS * updates_per_epoch

    print("=" * 70)
    print("PureDrop(E3): 59x-type2 + const lr=5e-4 + e_p zeroed with p=0.25")
    print(
        "  E3 ablation: per-sample deletion of position embedding (train-mode only), "
        "same protocol as B0/E1"
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
    apply_position_corruption(
        model,
        mode=CORRUPT_MODE,
        noise_std=NOISE_STD if CORRUPT_MODE == "noise" else None,
        drop_prob=DROP_PROB if CORRUPT_MODE == "drop" else None,
    )
    model.train()
    print(f"Model parameters: {report['parameters']['total']:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["adam_betas"]),
        eps=float(config["training"]["adam_eps"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    # E2 matrix protocol: constant LR=5e-4, no scheduler (identical to B0/E1,
    # so the ONLY variable vs B0/E1 is the position-corruption mechanism).
    base_lr = float(config["training"]["learning_rate"])
    print(f"SCHEDULER: constant lr={base_lr} (B0/E1 matrix protocol)")

    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    data_root = config["data"]["root"]

    loss_history = []
    val_history = []
    completed = 0
    start_epoch = 0

    # ---------------- AUTO-RESUME: pick up the latest checkpoint in run_dir ----------------
    # NOTE: skip checkpoint_final.pt (non-numeric step name would crash int()).
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
        # Recompute which epoch we should START from.
        # `completed` is summed across already-finished epochs; epoch counting is 0-indexed.
        start_epoch = completed // updates_per_epoch
        print(
            f"[resume] completed_updates={completed} -> resume from epoch {start_epoch}",
            flush=True,
        )
        # Constant LR: nothing to advance on resume.

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

            # E2 corruption lives INSIDE model.embeddings (train-mode only,
            # per-sample gaussian noise on e_p) — no batch-level swap here.

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
                    f"lr={base_lr:.2e} "
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

    # Final full-length test set evaluation (paper protocol).
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


if __name__ == "__main__":
    main()
