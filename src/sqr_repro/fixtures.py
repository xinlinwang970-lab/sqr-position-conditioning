from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .data import LAYOUTS, PARTS


def make_fixtures(out_dir: str | Path, seed: int = 2026, sample_rate: int = 44100, seconds: float = 3.2) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = int(sample_rate * seconds)
    t = np.arange(n, dtype=np.float32) / sample_rate
    freqs = {"violin1": 440.0, "violin2": 554.37, "viola": 329.63, "cello": 130.81}
    pans = {"violin1": 0.75, "violin2": -0.5, "viola": -0.1, "cello": 0.45}

    stems = {}
    for idx, part in enumerate(PARTS):
        phase = rng.uniform(0, 2 * np.pi)
        tone = 0.08 * np.sin(2 * np.pi * freqs[part] * t + phase)
        tone += 0.02 * np.sin(2 * np.pi * freqs[part] * 2.01 * t + phase / 2)
        envelope = np.linspace(0.2, 1.0, n, dtype=np.float32)
        envelope *= 0.9 + 0.1 * np.sin(2 * np.pi * (idx + 1) * t / seconds)
        mono = (tone * envelope).astype(np.float32)
        left_gain = np.sqrt((1.0 - pans[part]) / 2.0)
        right_gain = np.sqrt((1.0 + pans[part]) / 2.0)
        stereo = np.stack([left_gain * mono, right_gain * mono], axis=1).astype(np.float32)
        stems[part] = stereo
        sf.write(out / f"{part}.wav", stereo, sample_rate, subtype="FLOAT")

    mixture = sum(stems.values()).astype(np.float32)
    sf.write(out / "mixture.wav", mixture, sample_rate, subtype="FLOAT")

    manifest = {
        "item_id": "ENGINEERING_TEST_ONLY_fixture_type1",
        "composition_id": "ENGINEERING_TEST_ONLY_fixture",
        "performance_id": "ENGINEERING_TEST_ONLY_take",
        "source_dataset": "fixture",
        "split": "train",
        "layout_id": "type1",
        "mixture": "mixture.wav",
        "stems": {part: f"{part}.wav" for part in PARTS},
        "position_ids": LAYOUTS["type1"],
        "sample_rate": sample_rate,
        "channels": 2,
        "num_samples": n,
        "is_fixture": True,
    }
    manifest_path = out / "manifest.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest_path

