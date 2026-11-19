from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sqr_repro.config import load_config
from sqr_repro.models.separator import Separator


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_separator_rejects_invalid_inputs():
    config = load_config(FIXTURES / "audit_math_tiny.yaml")
    model = Separator(config)
    spec = torch.randn(2, 4, 8, 33)
    with pytest.raises(ValueError, match="torch.long"):
        model(spec, torch.tensor([0, 1]), torch.tensor([3.0, 1.0]))
    with pytest.raises(ValueError, match="timbre_id"):
        model(
            spec,
            torch.tensor([0, 3], dtype=torch.long),
            torch.tensor([3, 1], dtype=torch.long),
        )
    bad = spec.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        model(
            bad,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([3, 1], dtype=torch.long),
        )


def test_separator_core_parameters_receive_finite_gradients():
    torch.manual_seed(2026)
    config = load_config(FIXTURES / "audit_math_tiny.yaml")
    model = Separator(config)
    model.train()
    spec = torch.randn(2, 4, 8, 33)
    target = torch.randn_like(spec)
    out = model(
        spec,
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([3, 1], dtype=torch.long),
    )
    loss = (out - target).square().mean()
    loss.backward()
    checks = {
        "timbre_embedding": model.embeddings.timbre.weight.grad,
        "position_embedding": model.embeddings.position.weight.grad,
        "stem": model.stem[0].weight.grad,
        "encoder_tfc": model.encoders[0].tfc.layers[0][0].weight.grad,
        "head": model.head.weight.grad,
    }
    for name, grad in checks.items():
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum().item() > 0, name
