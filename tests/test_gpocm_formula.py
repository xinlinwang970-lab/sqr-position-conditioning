import pytest

torch = pytest.importorskip("torch")

from sqr_repro.models.conditioning import GPoCM


def test_gpocm_formula_matches_direct_einsum():
    torch.manual_seed(2026)
    h = torch.randn(2, 3, 4, 5, requires_grad=True)
    weight = torch.randn(2, 3, 3, requires_grad=True)
    bias = torch.randn(2, 3, requires_grad=True)
    actual = GPoCM()(h, weight, bias)
    expected = torch.sigmoid(torch.einsum("bcj,bjtf->bctf", weight, h) + bias[:, :, None, None]) * h
    assert torch.max(torch.abs(actual - expected)).item() <= 1e-6
    actual.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert bias.grad is not None and torch.isfinite(bias.grad).all()


def test_gpocm_rejects_bad_shapes():
    h = torch.randn(2, 3, 4, 5)
    gpocm = GPoCM()
    with pytest.raises(ValueError, match="weight"):
        gpocm(h, torch.randn(2, 3), torch.randn(2, 3))
    with pytest.raises(ValueError, match="bias"):
        gpocm(h, torch.randn(2, 3, 3), torch.randn(2, 4))

