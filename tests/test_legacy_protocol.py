"""Characterize the numerical limitations retained for historical result fidelity."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sqr_repro.audio import AudioConfig, AudioTransform
from sqr_repro.config import load_config
from sqr_repro.models.separator import Separator

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reported_metric_is_gain_dependent_waveform_sdr():
    module = load_script("evaluate_checkpoint")
    reference = torch.ones(2, 200)
    assert module.compute_sdr(reference * 0.5, reference).item() == pytest.approx(6.0206, abs=1e-4)
    assert module.compute_sdr(torch.zeros_like(reference), reference).item() == pytest.approx(0.0, abs=1e-6)


def test_legacy_hann_ola_keeps_zero_first_sample_and_short_padding():
    module = load_script("evaluate_checkpoint")
    class Identity(torch.nn.Module):
        def forward(self, spec, timbre, position):
            return spec
    cfg = AudioConfig(sample_rate=8000, n_fft=32, win_length=32, hop_length=16, num_frames=8)
    wave = torch.ones(2, 20)
    result = module.inference_full_length(Identity(), wave, AudioTransform(cfg), 0, 3, "cpu", 8, 16)
    assert result.shape[-1] == 112
    assert result[0, 0].item() == 0.0


def test_e4_identity_position_still_penalizes_independent_dropout(monkeypatch):
    module = load_script("train_PureSwap_E4_1000ep")
    config = load_config(REPO / "tests/fixtures/audit_math_tiny.yaml")
    config["model"].update(condition_route="mid_dual_paper", control_dropout=0.5)
    torch.manual_seed(2026)
    model = Separator(config)
    timbre, position = torch.tensor([0, 1, 2, 0]), torch.tensor([3, 1, 0, 2])
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))
    assert module.compute_position_invariance_reg(model.train(), timbre, position).item() > 0
    assert module.compute_position_invariance_reg(model.eval(), timbre, position).item() == 0


def test_manual_checkpoint_wrappers_update_batchnorm_during_recompute():
    module = load_script("train_PureSwap_E1cosine-P15_1000ep")
    config = load_config(REPO / "tests/fixtures/audit_math_tiny.yaml")
    config["model"].update(condition_route="mid_dual_paper", control_dropout=0.1)
    model = Separator(config).train()
    module.apply_activation_checkpointing(model)
    batchnorm = next(child for child in model.encoders[0].modules() if isinstance(child, torch.nn.BatchNorm2d))
    before = batchnorm.num_batches_tracked.item()
    output = model(torch.randn(2, 4, 8, 33), torch.tensor([0, 1]), torch.tensor([3, 1]))
    assert batchnorm.num_batches_tracked.item() == before + 1
    output.square().mean().backward()
    assert batchnorm.num_batches_tracked.item() == before + 2
