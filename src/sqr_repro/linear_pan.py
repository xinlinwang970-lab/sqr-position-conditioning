"""Linear-pan re-rendering for position augmentation.

The paper's author render pipeline (EnsembleSet stereo -> B-format rotate to slot)
cannot be replicated exactly because `rotateBformat.m` is not open-sourced.
We replace it with simple constant-power linear panning onto 4 discrete slots.

Four slots (constant power, angles in degrees on the front arc):
  slot 0: far-left    pan angle -45 deg -> gains (cos(11.25°+45°), sin(...)) style
  slot 1: left-near   pan angle -15 deg
  slot 2: right-near  pan angle +15 deg
  slot 3: far-right   pan angle +45 deg

We use constant-power panning: L = cos(theta), R = sin(theta) with theta in [0, pi/2]
mapping to far-left..far-right.

If input is stereo (2, T), we take the mono mid, apply pan gains to form a new stereo.
"""

from __future__ import annotations

import numpy as np

SLOT_ANGLES_DEG = {0: -45.0, 1: -15.0, 2: +15.0, 3: +45.0}


def pan_gains_for_slot(slot_id: int) -> tuple[float, float]:
    """Return (left_gain, right_gain) for a slot using constant-power panning."""
    if slot_id not in SLOT_ANGLES_DEG:
        raise ValueError(f"invalid slot {slot_id}")
    angle = SLOT_ANGLES_DEG[slot_id]
    # map angle from [-45, +45] to theta in [0, pi/2]
    theta = (angle + 45.0) / 90.0 * (np.pi / 2.0)
    return float(np.cos(theta)), float(np.sin(theta))


def pan_stem(stem: np.ndarray, slot_id: int) -> np.ndarray:
    """Pan a stem to a slot.

    Args:
        stem: (2, T) float32 or (T,) float32
        slot_id: 0..3
    Returns:
        (2, T) float32 stereo
    """
    if stem.ndim == 2:
        # mid channel
        mono = 0.5 * (stem[0] + stem[1])
    elif stem.ndim == 1:
        mono = stem
    else:
        raise ValueError(f"unexpected stem shape {stem.shape}")
    lg, rg = pan_gains_for_slot(slot_id)
    out = np.stack([mono * lg, mono * rg], axis=0).astype(np.float32)
    return out


def render_mixture(stems: dict[str, np.ndarray], layout: dict[str, int]) -> np.ndarray:
    """Render a mixture by panning each stem to its slot and summing.

    Args:
        stems: {part_name: (2, T)}
        layout: {part_name: slot_id}
    Returns:
        (2, T) float32 mixture, peak-normalized to |max| = 1.0
    """
    parts = sorted(stems.keys())
    if sorted(layout.keys()) != parts:
        raise ValueError("layout and stems must have same parts")
    T = stems[parts[0]].shape[-1]
    mix = np.zeros((2, T), dtype=np.float32)
    for part in parts:
        mix += pan_stem(stems[part], int(layout[part]))
    # peak normalize like author MATLAB
    peak = np.abs(mix).max()
    if peak > 1e-8:
        mix = mix / peak
    return mix


def make_random_layout(rng: np.random.Generator, parts: list[str]) -> dict[str, int]:
    """Random permutation of parts -> slots {0,1,2,3}."""
    if len(parts) != 4:
        raise ValueError("expected 4 parts")
    slots = rng.permutation(4)
    return {part: int(slot) for part, slot in zip(parts, slots)}
