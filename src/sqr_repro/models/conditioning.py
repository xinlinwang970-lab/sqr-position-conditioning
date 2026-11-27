from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, num_timbres: int = 3, num_positions: int = 4):
        super().__init__()
        self.num_timbres = int(num_timbres)
        self.num_positions = int(num_positions)
        self.timbre = nn.Embedding(self.num_timbres, embedding_dim)
        # Position: trainable embedding (paper-compliant).
        # Reverted from spatial encoding (cos/sin) which proved too restrictive.
        self.position = nn.Embedding(self.num_positions, embedding_dim)

    def forward(self, timbre_id, position_id):
        e_t = self.timbre(timbre_id)
        e_p = self.position(position_id)
        e_m = 0.5 * (e_t + e_p)
        return e_t, e_m, e_p


class ConditionParameterGenerator(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        channels: int | list[int] | tuple[int, ...],
        n_stages: int,
        hidden_dims: list[int],
        dropout: float,
        hidden_norm: str = "layer_norm",
        output_norm: str = "none",
    ):
        super().__init__()
        if isinstance(channels, int):
            stage_channels = [int(channels)] * int(n_stages)
        else:
            stage_channels = [int(value) for value in channels]
        if len(stage_channels) != int(n_stages) or any(value <= 0 for value in stage_channels):
            raise ValueError("channels must provide one positive width per stage")
        self.stage_channels = tuple(stage_channels)
        # Keep the historical attribute for callers that inspect uniform controllers.
        self.channels = stage_channels[0] if len(set(stage_channels)) == 1 else self.stage_channels
        self.n_stages = int(n_stages)
        self.hidden_norm = hidden_norm
        self.output_norm = output_norm
        self.uses_batch_norm = False
        self.weight_sizes = tuple(value * value for value in self.stage_channels)
        self.bias_sizes = self.stage_channels
        out_dim = sum(weight + bias for weight, bias in zip(self.weight_sizes, self.bias_sizes))

        layers: list[nn.Module] = []
        in_dim = embedding_dim
        for index, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if hidden_norm == "layer_norm":
                layers.append(nn.LayerNorm(hidden_dim))
            elif hidden_norm == "batch_norm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif hidden_norm != "none":
                raise ValueError(f"Unsupported controller hidden norm: {hidden_norm}")
            if index >= 1 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if output_norm == "batch_norm_no_affine":
            layers.append(nn.BatchNorm1d(out_dim, affine=False))
        elif output_norm == "layer_norm":
            layers.append(nn.LayerNorm(out_dim))
        elif output_norm != "none":
            raise ValueError(f"Unsupported controller output norm: {output_norm}")
        self.net = nn.Sequential(*layers)

    def forward(self, embedding):
        raw = self.net(embedding)
        bsz = raw.shape[0]
        weight_count = sum(self.weight_sizes)
        weight_values = raw[:, :weight_count]
        bias_values = raw[:, weight_count:]
        results = []
        weight_offset = 0
        bias_offset = 0
        for channels, weight_size, bias_size in zip(
            self.stage_channels, self.weight_sizes, self.bias_sizes
        ):
            weight = weight_values[:, weight_offset : weight_offset + weight_size].reshape(
                bsz, channels, channels
            )
            bias = bias_values[:, bias_offset : bias_offset + bias_size].reshape(bsz, channels)
            results.append((weight, bias))
            weight_offset += weight_size
            bias_offset += bias_size
        return results


class GPoCM(nn.Module):
    def forward(self, h, weight, bias):
        if h.ndim != 4:
            raise ValueError("h must have shape [B, C, T, F]")
        if weight.shape != (h.shape[0], h.shape[1], h.shape[1]):
            raise ValueError("weight must have shape [B, C, C]")
        if bias.shape != (h.shape[0], h.shape[1]):
            raise ValueError("bias must have shape [B, C]")
        activation = torch.einsum("bcj,bjtf->bctf", weight, h) + bias[:, :, None, None]
        return torch.sigmoid(activation) * h

