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


class Growth24ProjectedTFCBlock(nn.Module):
    """Dense TFC with an explicit growth rate and projected block width.

    The Dense layers add ``growth_rate`` channels each.  Their concatenated
    input and outputs are projected with a 1x1 convolution to ``out_channels``.
    This is an explicit candidate architecture; the projection is not asserted
    to be part of the published paper implementation.
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        out_channels: int,
        layers: int,
        kernel_size: tuple[int, int],
    ):
        super().__init__()
        if min(in_channels, growth_rate, out_channels, layers) <= 0:
            raise ValueError("TFC widths and layer count must be positive")
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.in_channels = int(in_channels)
        self.growth_rate = int(growth_rate)
        self.out_channels = int(out_channels)
        self.layers = nn.ModuleList()
        for index in range(layers):
            layer_in = self.in_channels + index * self.growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        layer_in,
                        self.growth_rate,
                        kernel_size=kernel_size,
                        padding=pad,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.growth_rate),
                    nn.ReLU(inplace=True),
                )
            )
        dense_channels = self.in_channels + layers * self.growth_rate
        self.projection = nn.Sequential(
            nn.Conv2d(dense_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.projection(torch.cat(features, dim=1))


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


class Growth24ProjectedControlledTfcTdfBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        growth_rate: int,
        freq_bins: int,
        tfc_layers: int,
        kernel_size: tuple[int, int],
        tdf_bottleneck_factor: int,
        tdf_min_units: int,
        tdf_bias: bool,
    ):
        super().__init__()
        self.tfc = Growth24ProjectedTFCBlock(
            in_channels, growth_rate, out_channels, tfc_layers, kernel_size
        )
        self.gpocm = GPoCM()
        self.tdf = TDFBlock(
            out_channels, freq_bins, tdf_bottleneck_factor, tdf_min_units, tdf_bias
        )

    def forward(self, x, params=None):
        h = self.tfc(x)
        if params is not None:
            weight, bias = params
            h = self.gpocm(h, weight, bias)
        return h + self.tdf(h)


class Downsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = channels if out_channels is None else int(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = channels if out_channels is None else int(out_channels)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

