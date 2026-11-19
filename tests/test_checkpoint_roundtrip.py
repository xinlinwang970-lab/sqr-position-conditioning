from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sqr_repro.checkpoint import load_checkpoint, save_checkpoint
from sqr_repro.config import load_config
from sqr_repro.models.separator import Separator


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
ARTIFACTS = ROOT / "test_artifacts"


def test_untrained_checkpoint_roundtrip_outputs_match():
    torch.manual_seed(2026)
    config = load_config(FIXTURES / "audit_math_tiny.yaml")
    model = Separator(config).eval()
    spec = torch.randn(2, 4, 8, 33)
    timbre = torch.tensor([0, 1], dtype=torch.long)
    position = torch.tensor([3, 1], dtype=torch.long)
    with torch.inference_mode():
        expected = model(spec, timbre, position)
    path = ARTIFACTS / "roundtrip.pt"
    save_checkpoint(path, model, config, {"stage": "stage_a_round1"})
    loaded = Separator(config).eval()
    checkpoint = load_checkpoint(path, loaded, expected_config=config)
    with torch.inference_mode():
        actual = loaded(spec, timbre, position)
    assert checkpoint["metadata"]["stage"] == "stage_a_round1"
    assert torch.allclose(actual, expected, atol=1e-6)


def test_checkpoint_config_mismatch_fails_before_loading():
    torch.manual_seed(2026)
    config = load_config(FIXTURES / "audit_math_tiny.yaml")
    model = Separator(config).eval()
    path = ARTIFACTS / "roundtrip_mismatch.pt"
    save_checkpoint(path, model, config, {"stage": "stage_a_round1b"})

    expected = load_config(FIXTURES / "audit_math_tiny.yaml")
    expected["model"]["embedding_dim"] = int(expected["model"]["embedding_dim"]) + 1

    with pytest.raises(ValueError, match="config does not match"):
        load_checkpoint(path, Separator(config), expected_config=expected)
