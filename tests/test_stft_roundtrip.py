import pytest

torch = pytest.importorskip("torch")

from sqr_repro.audio import AudioConfig, AudioTransform


def test_stft_roundtrip_preserves_length_and_channels():
    cfg = AudioConfig(sample_rate=8000, n_fft=128, win_length=128, hop_length=64, num_frames=8)
    transform = AudioTransform(cfg)
    t = torch.arange(cfg.segment_samples, dtype=torch.float32) / cfg.sample_rate
    left = torch.sin(2 * torch.pi * 220 * t)
    right = torch.sin(2 * torch.pi * 330 * t)
    waveform = torch.stack([left, right], dim=0).unsqueeze(0)
    spec = transform.stft(waveform)
    restored = transform.istft(spec, length=waveform.shape[-1])
    rel = torch.linalg.vector_norm(restored - waveform) / torch.linalg.vector_norm(waveform)
    assert spec.shape == (1, 4, 8, 65)
    assert restored.shape == waveform.shape
    assert rel.item() <= 1e-5
    assert not torch.allclose(restored[:, 0], restored[:, 1])

