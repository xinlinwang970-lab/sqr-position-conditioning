#!/usr/bin/env python3
"""Evaluate a specific checkpoint with full-length WaveSDR (val or test).

Reuses the exact evaluation protocol of the training scripts:
full-length legacy_v1 sliding-window inference, per-part WaveSDR vs stems,
summarized as mean/median/type1_median/type2_median.

Usage:
  python evaluate_checkpoint.py --config <yaml> --checkpoint <ckpt.pt> \
      --manifest <val|test jsonl> [--root <data_root>] [--tag <label>]
"""

import argparse
import json
import sys

sys.path.insert(0, "/ceph/leon/MSS/github_release/src")

from pathlib import Path

import torch

from sqr_repro.config import load_config
from sqr_repro.paper_c0.model_factory import build_model
from sqr_repro.evaluation import (
    evaluate_full_length_legacy_v1 as evaluate_full_length,
    summarize_wavesdr as summarize,
)












def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()

    config = load_config(args.config)
    data_root = args.root or config["data"]["root"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model(config, "w24_engineering_pilot")
    ckpt = torch.load(args.checkpoint, map_location=device)
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
    out = Path(args.checkpoint).with_suffix(f".{args.tag}.json")
    with open(out, "w") as f:
        json.dump({"tag": args.tag, "summary": log, "by_track": results}, f, indent=2)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
