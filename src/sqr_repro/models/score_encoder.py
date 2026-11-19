from __future__ import annotations

import torch
import torch.nn as nn


class ScoreEncoder(nn.Module):
    """Encode a time-varying MIDI pitch activity roll for middle-stage injection."""

    def __init__(self, num_pitches: int, hidden_channels: int, output_channels: int):
        super().__init__()
        self.num_pitches = int(num_pitches)
        self.hidden_channels = int(hidden_channels)
        self.output_channels = int(output_channels)
        self.net = nn.Sequential(
            nn.Conv1d(self.num_pitches, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, self.hidden_channels),
            nn.SiLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, self.hidden_channels),
            nn.SiLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, self.hidden_channels),
            nn.SiLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, self.hidden_channels),
            nn.SiLU(),
        )
        self.score_projection = nn.Conv1d(self.hidden_channels, self.output_channels, kernel_size=1)
        nn.init.zeros_(self.score_projection.weight)
        nn.init.zeros_(self.score_projection.bias)

    def forward(self, score_roll: torch.Tensor) -> torch.Tensor:
        return self.score_projection(self.encode(score_roll))

    def encode(self, score_roll: torch.Tensor) -> torch.Tensor:
        if score_roll.ndim != 3:
            raise ValueError("score_roll must have shape [B, num_pitches, T]")
        if score_roll.shape[1] != self.num_pitches:
            raise ValueError(f"Expected score pitch dimension {self.num_pitches}, got {score_roll.shape[1]}")
        if score_roll.shape[-1] % 8 != 0:
            raise ValueError("score_roll time length must be divisible by 8")
        if not torch.is_floating_point(score_roll):
            raise ValueError("score_roll must be a floating point tensor")
        if not torch.isfinite(score_roll).all():
            raise ValueError("score_roll contains NaN or Inf")
        return self.net(score_roll)

    def project(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 3 or encoded.shape[1] != self.hidden_channels:
            raise ValueError("encoded score must have shape [B, hidden_channels, T]")
        return self.score_projection(encoded)

    def projection_is_zero(self) -> bool:
        return bool(torch.count_nonzero(self.score_projection.weight).item() == 0 and torch.count_nonzero(self.score_projection.bias).item() == 0)
