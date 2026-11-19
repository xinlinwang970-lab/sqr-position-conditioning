from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreEvent:
    part_id: str
    onset_seconds: float
    offset_seconds: float
    midi_pitch: int
    velocity: float = 1.0


def events_to_piano_roll(
    events: list[ScoreEvent] | tuple[ScoreEvent, ...],
    *,
    num_frames: int,
    sample_rate: int,
    hop_length: int,
    num_pitches: int = 128,
    center: bool = True,
) -> np.ndarray:
    """Render binary activity using the actual centered-STFT frame timestamps.

    With center=True, frame t is assigned timestamp t * hop_length / sample_rate,
    matching torch.stft's frame-center convention. Intervals are half-open:
    onset <= frame_time < offset. Padding does not create score activity.
    """
    if num_frames < 1 or num_pitches < 1 or sample_rate < 1 or hop_length < 1:
        raise ValueError("num_frames, num_pitches, sample_rate, and hop_length must be positive")
    if not center:
        raise ValueError("P2 score contract currently requires audio.center=true")
    roll = np.zeros((num_pitches, num_frames), dtype=np.float32)
    frame_times = np.arange(num_frames, dtype=np.float64) * float(hop_length) / float(sample_rate)
    for event in events:
        if event.offset_seconds <= event.onset_seconds:
            raise ValueError(f"{event.part_id}: offset must be greater than onset")
        pitch = int(event.midi_pitch)
        if pitch < 0 or pitch >= num_pitches:
            raise ValueError(f"{event.part_id}: midi_pitch {pitch} outside [0, {num_pitches})")
        active = (frame_times >= float(event.onset_seconds)) & (frame_times < float(event.offset_seconds))
        roll[pitch, active] = 1.0
    return roll


def crop_piano_roll(roll: np.ndarray, start_frame: int, num_frames: int) -> np.ndarray:
    value = _validate_roll(roll)
    start = int(start_frame)
    length = int(num_frames)
    if start < 0 or length < 1:
        raise ValueError("start_frame must be non-negative and num_frames must be positive")
    result = np.zeros((value.shape[0], length), dtype=np.float32)
    left = max(0, start)
    right = min(value.shape[1], start + length)
    if right > left:
        result[:, left - start : right - start] = value[:, left:right]
    return result


def shift_piano_roll(roll: np.ndarray, shift_frames: int) -> np.ndarray:
    value = _validate_roll(roll)
    shift = int(shift_frames)
    result = np.zeros_like(value)
    if shift >= 0:
        if shift < value.shape[1]:
            result[:, shift:] = value[:, : value.shape[1] - shift]
    else:
        amount = -shift
        if amount < value.shape[1]:
            result[:, : value.shape[1] - amount] = value[:, amount:]
    return result


def pitch_shift_piano_roll(roll: np.ndarray, shift_semitones: int) -> np.ndarray:
    value = _validate_roll(roll)
    shift = int(shift_semitones)
    result = np.zeros_like(value)
    source_left = max(0, -shift)
    source_right = min(value.shape[0], value.shape[0] - shift)
    target_left = max(0, shift)
    target_right = target_left + max(0, source_right - source_left)
    if source_right > source_left and target_right <= value.shape[0]:
        result[target_left:target_right] = value[source_left:source_right]
    return result


def swap_piano_rolls(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_value = _validate_roll(left)
    right_value = _validate_roll(right)
    if left_value.shape != right_value.shape:
        raise ValueError("swapped score rolls must have identical shapes")
    return right_value.copy(), left_value.copy()


def zero_piano_roll_like(roll: np.ndarray) -> np.ndarray:
    return np.zeros_like(_validate_roll(roll))


def score_hash(roll: np.ndarray) -> str:
    value = _validate_roll(roll)
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate_roll(roll: np.ndarray) -> np.ndarray:
    value = np.asarray(roll, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("piano roll must have shape [num_pitches, num_frames]")
    if not np.isfinite(value).all():
        raise ValueError("piano roll contains NaN or Inf")
    if not np.isin(value, [0.0, 1.0]).all():
        raise ValueError("P2 binary piano roll must contain only 0 or 1")
    return value
