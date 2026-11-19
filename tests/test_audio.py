import pytest

torch = pytest.importorskip("torch")

from sqr_repro.audio import AudioConfig, AudioTransform, pack_complex, unpack_complex


def test_pack_unpack_roundtrip():
    complex_spec = torch.randn(2, 2, 9, 5, dtype=torch.complex64)
    packed = pack_complex(complex_spec)
    restored = unpack_complex(packed)
    assert packed.shape == (2, 4, 5, 9)
    assert torch.allclose(restored, complex_spec)


def test_stft_istft_roundtrip_debug():
    cfg = AudioConfig(sample_rate=8000, n_fft=128, win_length=128, hop_length=64, num_frames=8)
    transform = AudioTransform(cfg)
    waveform = torch.randn(2, 2, cfg.segment_samples)
    spec = transform.stft(waveform)
    restored = transform.istft(spec, length=waveform.shape[-1])
    relative = torch.linalg.vector_norm(restored - waveform) / torch.linalg.vector_norm(waveform)
    assert spec.shape == (2, 4, 8, 65)
    assert relative.item() < 1e-5

