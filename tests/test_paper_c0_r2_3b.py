from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sqr_repro.audio import AudioConfig, AudioTransform
from sqr_repro.checkpoint import capture_rng_state
from sqr_repro.paper_c0.metrics import PAPER_SDR_PROTOCOL, aggregate_track_metrics, evaluate_waveform_variants, waveform_sdr
from sqr_repro.paper_c0.waveform import PaperC0WaveformError, infer_checkpoint_wav, infer_full_waveform, infer_wav_file
from sqr_repro.paper_c0.trainer import PaperC0SyntheticTrainer
from test_paper_c0_trainer import make_training_fixture


class IdentitySpectralModel(torch.nn.Module):
    def forward(self, packed, timbre, position):
        return packed


def waveform_config(sample_rate=44100, n_fft=2048, hop_length=1024, num_frames=8):
    return {
        "audio": {
            "sample_rate": sample_rate,
            "channels": 2,
            "n_fft": n_fft,
            "win_length": n_fft,
            "hop_length": hop_length,
            "num_frames": num_frames,
            "center": True,
            "pad_mode": "reflect",
            "window_periodic": True,
            "normalized": False,
        }
    }


def make_stereo(length):
    time = torch.arange(length, dtype=torch.float32) / 1000.0
    return torch.stack([torch.sin(time), torch.cos(time * 0.7)], dim=0)


def test_full_waveform_identity_chunked_and_single_chunk_roundtrip():
    config = waveform_config()
    model = IdentitySpectralModel()
    mixture = make_stereo(9000)
    single, single_meta = infer_full_waveform(model, mixture, config, 0, 1, chunk_samples=7168, overlap_samples=0)
    chunked, chunk_meta = infer_full_waveform(model, mixture, config, 0, 1, chunk_samples=7168, overlap_samples=3072)
    assert single.shape == mixture.shape == chunked.shape
    assert torch.allclose(single, mixture, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(chunked, single, atol=2.0e-5, rtol=2.0e-5)
    assert single_meta["sample_rate"] == 44100
    assert chunk_meta["overlap_add_assumption"] == "ENGINEERING_ASSUMPTION"
    assert chunk_meta["chunk_starts"][-1] + 7168 >= mixture.shape[-1]


def test_full_waveform_short_tail_silence_and_channel_identity(tmp_path):
    config = waveform_config()
    model = IdentitySpectralModel()
    short = make_stereo(1000)
    output, metadata = infer_full_waveform(model, short, config, 1, 2, chunk_samples=7168, overlap_samples=0)
    assert output.shape == short.shape
    assert torch.allclose(output, short, atol=2.0e-5, rtol=2.0e-5)
    path = tmp_path / "mixture.wav"
    out = tmp_path / "target.wav"
    sf.write(path, short.T.numpy(), 44100, subtype="FLOAT")
    written = infer_wav_file(model, path, out, config, 1, 2, chunk_samples=7168)
    assert written["input_samples"] == 1000
    assert Path(out).exists() and Path(str(out) + ".json").exists()
    read, rate = sf.read(out, dtype="float32", always_2d=True)
    assert rate == 44100 and read.shape == (1000, 2)


def test_waveform_metrics_are_explicit_and_baselines_explainable():
    target = np.asarray([[1.0, -1.0], [0.5, -0.5]], dtype=np.float64)
    sources = [target, -target, np.zeros_like(target)]
    result = evaluate_waveform_variants(target, target, sources)
    assert result["mae"] == 0.0
    assert result["oracle"]["mae"] == 0.0
    assert result["zero"]["mae"] > 0.0
    assert result["scaled_estimate"]["sdr"] == waveform_sdr(0.5 * target, target)
    assert result["paper_sdr_protocol"] == PAPER_SDR_PROTOCOL
    aggregate = aggregate_track_metrics([{"mae": result["mae"], "sdr": result["sdr"]}])
    assert aggregate["sdr_protocol"] == PAPER_SDR_PROTOCOL


def test_waveform_inference_rejects_invalid_overlap_and_nonfinite():
    config = waveform_config()
    model = IdentitySpectralModel()
    with torch.no_grad():
        bad = make_stereo(9000)
        bad[0, 0] = float("nan")
    try:
        infer_full_waveform(model, bad, config, 0, 0, chunk_samples=7168)
    except PaperC0WaveformError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite waveform must be rejected")
    try:
        infer_full_waveform(model, make_stereo(9000), config, 0, 0, chunk_samples=7168, overlap_samples=1)
    except PaperC0WaveformError as exc:
        assert "hop-aligned" in str(exc)
    else:
        raise AssertionError("non-aligned overlap must be rejected")


def test_checkpoint_waveform_entrypoint_loads_without_update(tmp_path):
    config, dataset = make_training_fixture(tmp_path / "checkpoint_fixture")
    trainer = PaperC0SyntheticTrainer(config, dataset)
    checkpoint = tmp_path / "model.pt"
    trainer.save(checkpoint)
    mixture = make_stereo(5000)
    mixture_path = tmp_path / "mixture.wav"
    output_path = tmp_path / "output.wav"
    sf.write(mixture_path, mixture.T.numpy(), 8000, subtype="FLOAT")
    metadata = infer_checkpoint_wav(checkpoint, mixture_path, output_path, config, 0, 0, chunk_samples=3968)
    assert metadata["optimizer_updates"] == 0
    assert metadata["checkpoint_sha256"]
    assert metadata["architecture_identity"]


def test_waveform_evaluation_restores_all_rng_families_on_model_failure(monkeypatch):
    import random

    import sqr_repro.paper_c0.waveform as waveform_module

    config = waveform_config()
    model = IdentitySpectralModel()
    mixture = make_stereo(9000)
    before = capture_rng_state()
    monkeypatch.setattr(waveform_module.AudioTransform, "stft", lambda self, value: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    try:
        infer_full_waveform(model, mixture, config, 0, 0, chunk_samples=7168)
    except RuntimeError:
        pass
    else:
        raise AssertionError("injected waveform failure must propagate")
    after = capture_rng_state()
    assert torch.equal(before["torch_cpu"], after["torch_cpu"])
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert before["python"] == after["python"]
