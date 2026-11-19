from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sqr_repro.config import load_config
from sqr_repro.models.conditioning import GPoCM
from sqr_repro.models.separator import Separator


ROOT = Path(__file__).resolve().parents[1]


def test_gpocm_matches_loop_and_zero_gate():
    h = torch.randn(2, 3, 4, 5, requires_grad=True)
    weight = torch.randn(2, 3, 3, requires_grad=True)
    bias = torch.randn(2, 3, requires_grad=True)
    gpocm = GPoCM()
    y = gpocm(h, weight, bias)
    expected = torch.empty_like(h)
    for b in range(2):
        for c in range(3):
            activation = bias[b, c]
            for j in range(3):
                activation = activation + weight[b, c, j] * h[b, j]
            expected[b, c] = torch.sigmoid(activation) * h[b, c]
    assert torch.allclose(y, expected, atol=1e-6)
    zero = gpocm(h, torch.zeros_like(weight), torch.zeros_like(bias))
    assert torch.allclose(zero, 0.5 * h)
    y.mean().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_separator_debug_shape_and_gradients():
    config = load_config(ROOT / "tests" / "fixtures" / "debug_cpu.yaml")
    model = Separator(config)
    model.train()
    spec = torch.randn(2, 4, 8, 65)
    timbre = torch.tensor([0, 1])
    position = torch.tensor([3, 1])
    out = model(spec, timbre, position)
    assert out.shape == spec.shape
    loss = out.square().mean()
    loss.backward()
    assert model.embeddings.timbre.weight.grad is not None
    assert model.embeddings.position.weight.grad is not None


def test_batch_one_train_mode_rejected_by_controller():
    config = load_config(ROOT / "tests" / "fixtures" / "debug_cpu.yaml")
    model = Separator(config)
    model.train()
    spec = torch.randn(1, 4, 8, 65)
    with pytest.raises(ValueError, match="BatchNorm1d|more than 1 value per channel"):
        model(spec, torch.tensor([1]), torch.tensor([1]))

