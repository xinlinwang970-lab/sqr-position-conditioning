from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for audio transforms. Install a compatible torch build first.")
    return torch


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 44100
    channels: int = 2
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 1024
    num_frames: int = 128
    center: bool = True
    pad_mode: str = "reflect"
    window_periodic: bool = True
    normalized: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "AudioConfig":
        return cls(
            sample_rate=int(data["sample_rate"]),
            channels=int(data["channels"]),
            n_fft=int(data["n_fft"]),
            win_length=int(data["win_length"]),
            hop_length=int(data["hop_length"]),
            num_frames=int(data["num_frames"]),
            center=bool(data["center"]),
            pad_mode=str(data["pad_mode"]),
            window_periodic=bool(data["window_periodic"]),
            normalized=bool(data["normalized"]),
        )

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def segment_samples(self) -> int:
        return (self.num_frames - 1) * self.hop_length


class AudioTransform:
    def __init__(self, config: AudioConfig):
        require_torch()
        if config.channels != 2:
            raise ValueError("AudioTransform only supports stereo waveforms")
        self.config = config
        self._window_cache: dict[tuple[str, int], torch.Tensor] = {}

    def _window(self, device, dtype):
        key = (str(device), torch.float32)
        if key not in self._window_cache:
            self._window_cache[key] = torch.hann_window(
                self.config.win_length,
                periodic=self.config.window_periodic,
                device=device,
                dtype=torch.float32,
            )
        return self._window_cache[key].to(dtype=torch.float32)

    def stft(self, waveform):
        if waveform.ndim != 3 or waveform.shape[1] != 2:
            raise ValueError("waveform must have shape [B, 2, L]")
        x = waveform.float()
        original_length = x.shape[-1]
        x = self._pad_short_for_reflect(x)
        batch = x.shape[0]
        flat = x.reshape(batch * 2, x.shape[-1])
        complex_spec = torch.stft(
            flat,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window(x.device, x.dtype),
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            normalized=self.config.normalized,
            onesided=True,
            return_complex=True,
        )
        complex_spec = complex_spec.reshape(batch, 2, self.config.frequency_bins, -1)
        packed = pack_complex(complex_spec)
        expected_frames = 1 + original_length // self.config.hop_length
        if packed.shape[2] != expected_frames:
            packed = packed[:, :, :expected_frames, :]
        return packed

    def istft(self, packed_spec, length: int):
        complex_spec = unpack_complex(packed_spec)
        batch = complex_spec.shape[0]
        flat = complex_spec.reshape(batch * 2, complex_spec.shape[2], complex_spec.shape[3])
        wave = torch.istft(
            flat,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window(packed_spec.device, packed_spec.dtype),
            center=self.config.center,
            normalized=self.config.normalized,
            onesided=True,
            length=length,
        )
        return wave.reshape(batch, 2, length)

    def _pad_short_for_reflect(self, waveform):
        if not self.config.center or self.config.pad_mode != "reflect":
            return waveform
        minimum = self.config.n_fft // 2 + 1
        if waveform.shape[-1] >= minimum:
            return waveform
        pad = minimum - waveform.shape[-1]
        return F.pad(waveform, (0, pad), mode="replicate")


def pack_complex(complex_spec):
    if complex_spec.ndim != 4 or complex_spec.shape[1] != 2:
        raise ValueError("complex_spec must have shape [B, 2, F, T]")
    real = complex_spec.real
    imag = complex_spec.imag
    packed = torch.stack([real[:, 0], real[:, 1], imag[:, 0], imag[:, 1]], dim=1)
    return packed.permute(0, 1, 3, 2).contiguous()


def unpack_complex(packed_spec):
    if packed_spec.ndim != 4 or packed_spec.shape[1] != 4:
        raise ValueError("packed_spec must have shape [B, 4, T, F]")
    spec = packed_spec.permute(0, 1, 3, 2).contiguous()
    left = torch.complex(spec[:, 0], spec[:, 2])
    right = torch.complex(spec[:, 1], spec[:, 3])
    return torch.stack([left, right], dim=1)

