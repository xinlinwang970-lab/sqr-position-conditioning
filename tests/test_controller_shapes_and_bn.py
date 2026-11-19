import pytest

torch = pytest.importorskip("torch")

from sqr_repro.models.conditioning import ConditionParameterGenerator


def test_controller_returns_stage_parameter_shapes():
    controller = ConditionParameterGenerator(
        embedding_dim=8,
        channels=4,
        n_stages=3,
        hidden_dims=[8, 16],
        dropout=0.0,
        hidden_norm="none",
        output_norm="none",
    )
    params = controller(torch.randn(2, 8))
    assert len(params) == 3
    for weight, bias in params:
        assert weight.shape == (2, 4, 4)
        assert bias.shape == (2, 4)
        assert torch.isfinite(weight).all()
        assert torch.isfinite(bias).all()


def test_controller_batchnorm_guard_covers_hidden_and_output_norm():
    hidden_bn = ConditionParameterGenerator(8, 4, 1, [8], 0.0, "batch_norm", "none")
    output_bn = ConditionParameterGenerator(
        8, 4, 1, [8], 0.0, "none", "batch_norm_no_affine"
    )
    hidden_bn.train()
    output_bn.train()
    with pytest.raises(ValueError, match="BatchNorm1d|more than 1 value"):
        hidden_bn(torch.randn(1, 8))
    with pytest.raises(ValueError, match="BatchNorm1d|more than 1 value"):
        output_bn(torch.randn(1, 8))


def test_controller_batchnorm_train_batch_two_and_eval_batch_one():
    controller = ConditionParameterGenerator(
        8, 4, 1, [8], 0.0, "batch_norm", "batch_norm_no_affine"
    )
    controller.train()
    params = controller(torch.randn(2, 8))
    loss = sum(weight.square().mean() + bias.square().mean() for weight, bias in params)
    loss.backward()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in controller.parameters()
    )
    controller.eval()
    params = controller(torch.randn(1, 8))
    assert params[0][0].shape == (1, 4, 4)
