import pytest

pytest.importorskip("torch")

import torch

from sqr_repro.losses import packed_complex_mse, packed_complex_mse_with_common_consistency


def _batch():
    mixture = torch.tensor(
        [
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
            [[[2.0]], [[1.0]], [[0.0]], [[1.0]]],
            [[[2.0]], [[1.0]], [[0.0]], [[1.0]]],
        ]
    )
    target = torch.tensor(
        [
            [[[0.7]], [[1.0]], [[2.0]], [[2.0]]],
            [[[0.3]], [[1.0]], [[1.0]], [[2.0]]],
            [[[1.5]], [[0.4]], [[0.0]], [[0.5]]],
            [[[0.5]], [[0.6]], [[0.0]], [[0.5]]],
        ],
        requires_grad=True,
    )
    prediction = target.detach().clone()
    prediction[0] += 0.1
    prediction[1] -= 0.1
    prediction[2] += 0.1
    prediction[3] += 0.1
    prediction.requires_grad_()
    return prediction, target, mixture


def test_lambda_zero_matches_existing_loss_and_preserves_pairing():
    prediction, target, mixture = _batch()
    expected = packed_complex_mse(prediction, target)
    actual, common = packed_complex_mse_with_common_consistency(
        prediction,
        target,
        mixture,
        ["m0", "m0", "m1", "m1"],
        ["m0_q0", "m0_q1", "m1_q0", "m1_q1"],
        0.0,
    )
    assert actual.item() == pytest.approx(expected.item(), abs=1.0e-12)
    expected.backward(retain_graph=True)
    expected_grad = prediction.grad.detach().clone()
    prediction.grad.zero_()
    actual.backward()
    assert torch.allclose(prediction.grad, expected_grad, atol=1.0e-7, rtol=0.0)
    assert common.item() >= 0.0


def test_common_term_distinguishes_symmetric_and_common_mode_errors():
    prediction, target, mixture = _batch()
    symmetric = target.detach().clone()
    symmetric[0] += 0.2
    symmetric[1] -= 0.2
    _, symmetric_common = packed_complex_mse_with_common_consistency(
        symmetric, target, mixture, ["m0", "m0", "m1", "m1"], ["a", "b", "c", "d"], 1.0
    )
    common = target.detach().clone()
    common[0] += 0.2
    common[1] += 0.2
    _, common_common = packed_complex_mse_with_common_consistency(
        common, target, mixture, ["m0", "m0", "m1", "m1"], ["a", "b", "c", "d"], 1.0
    )
    assert symmetric_common.item() < common_common.item()


def test_common_term_rejects_cross_mixture_or_incomplete_pairs():
    prediction, target, mixture = _batch()
    with pytest.raises(ValueError, match="expected exactly two"):
        packed_complex_mse_with_common_consistency(
            prediction, target, mixture[[0, 1, 2, 3]], ["m0", "m0", "m0", "m1"], ["a", "b", "c", "d"], 1.0
        )
