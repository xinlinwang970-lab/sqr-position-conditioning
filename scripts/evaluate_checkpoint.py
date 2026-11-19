#!/usr/bin/env python3
"""Evaluate a specific checkpoint on a full-length SDR manifest (val or test).

Reuses the exact evaluation protocol of the training scripts:
full-length, 50%-overlap sliding-window inference, per-part SDR vs stems,
summarized as mean/median/type1_median/type2_median.

Usage:
  python evaluate_checkpoint.py --config <yaml> --checkpoint <ckpt.pt> \
      --manifest <val|test jsonl> [--root <data_root>] [--tag <label>]
"""

import argparse
import json
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sqr_repro.config import load_config
from sqr_repro.release_paths import validate_manifest
from sqr_repro.paper_c0.dataset import PARTS
from sqr_repro.paper_c0.model_factory import build_model
from sqr_repro.audio import AudioConfig, AudioTransform


PART_TO_TIMBRE = {"violin1": 0, "violin2": 0, "viola": 1, "cello": 2}


def compute_sdr(estimate, reference):
    ref_energy = torch.sum(reference**2)
    err = estimate - reference
    return 10 * torch.log10(ref_energy / (torch.sum(err**2) + 1e-8))


def inference_full_length(
    model,
    mixture,
    transform,
    timbre_id,
    position_id,
    device,
    num_frames,
    hop_length,
    overlap_ratio=0.5,
):
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
        w = window[: end - start]
        output[:, start:end] += wave[:, : end - start] * w
        weight[start:end] += w
    return (output / torch.clamp(weight, min=1e-8))[:, :T]


def evaluate_full_length(
    model, manifest_path, data_root, device, config, part_filter=None
):
    records = [json.loads(l) for l in open(manifest_path)]
    transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
    num_frames = config["audio"]["num_frames"]
    hop_length = config["audio"]["hop_length"]
    results = []
    model.eval()
    for record in records:
        mix, _ = sf.read(
            str(Path(data_root) / record["mixture"]), dtype="float32", always_2d=True
        )
        mixture = torch.from_numpy(mix.T.copy())
        for part_id in PARTS:
            if part_filter == "type2" and record["layout_id"] != "type2":
                continue
            estimate = inference_full_length(
                model,
                mixture,
                transform,
                PART_TO_TIMBRE[part_id],
                record["position_ids"][part_id],
                device,
                num_frames,
                hop_length,
            )
            ref, _ = sf.read(
                str(Path(data_root) / record["stems"][part_id]),
                dtype="float32",
                always_2d=True,
            )
            reference = torch.from_numpy(ref.T.copy())
            min_len = min(estimate.shape[-1], reference.shape[-1])
            sdr = compute_sdr(estimate[..., :min_len], reference[..., :min_len]).item()
            results.append(
                {
                    "item_id": record["item_id"],
                    "composition_id": record["composition_id"],
                    "part_id": part_id,
                    "layout_id": record["layout_id"],
                    "sdr": sdr,
                }
            )
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--output", type=Path, help="Explicit new result JSON; existing files are never overwritten")
    args = ap.parse_args()

    out = args.output or Path(args.checkpoint).with_suffix(f".{args.tag}.json")
    if out.exists():
        raise FileExistsError(f"Choose a new --output; file already exists: {out}")

    config = load_config(args.config)
    data_root = Path(args.root).resolve() if args.root else (REPO_ROOT / config["data"]["root"]).resolve()
    validate_manifest(args.manifest, data_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model(config, "w24_engineering_pilot")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"Loaded {Path(args.checkpoint).name} (completed_updates={ckpt.get('completed_updates')})",
        flush=True,
    )
    print(
        f"Config param count sanity via build only; evaluating on: {args.manifest}",
        flush=True,
    )

    results = evaluate_full_length(model, args.manifest, data_root, device, config)
    log = summarize(results)
    print(
        f"[{args.tag}] mean={log['mean_sdr']:+.2f} median={log['median_sdr']:+.2f} "
        f"type1={log['type1_median']:+.2f} type2={log['type2_median']:+.2f} (n={log['n']})",
        flush=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "summary": log, "by_track": results}, f, indent=2)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
