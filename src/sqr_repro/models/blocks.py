from __future__ import annotations

import torch
import torch.nn as nn

from .conditioning import GPoCM


class TFCBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, layers: int, kernel_size: tuple[int, int]):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, channels, kernel_size=kernel_size, padding=pad, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels += channels

    def forward(self, x):
        features = [x]
        out = x
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return out


class TDFBlock(nn.Module):
    def __init__(self, channels: int, freq_bins: int, bottleneck_factor: int, min_units: int, bias: bool):
        super().__init__()
        hidden = max(freq_bins // bottleneck_factor, min_units)
        self.net = nn.Sequential(
            nn.Linear(freq_bins, hidden, bias=bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, freq_bins, bias=bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ControlledTfcTdfBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        freq_bins: int,
        tfc_layers: int,
        kernel_size: tuple[int, int],
        tdf_bottleneck_factor: int,
        tdf_min_units: int,
        tdf_bias: bool,
    ):
        super().__init__()
        self.tfc = TFCBlock(in_channels, channels, tfc_layers, kernel_size)
        self.gpocm = GPoCM()
        self.tdf = TDFBlock(channels, freq_bins, tdf_bottleneck_factor, tdf_min_units, tdf_bias)

    def forward(self, x, params=None):
        h = self.tfc(x)
        if params is not None:
            weight, bias = params
            h = self.gpocm(h, weight, bias)
        return h + self.tdf(h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

