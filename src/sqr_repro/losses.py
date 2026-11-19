from __future__ import annotations


def packed_complex_mse(prediction, target):
    return ((prediction.float() - target.float()) ** 2).mean()


def packed_complex_mse_with_common_consistency(
    prediction,
    target,
    mixture_spec,
    mixture_ids,
    query_ids,
    common_loss_lambda: float = 0.0,
):
    """Return separation loss plus an optional same-mixture common term.

    The common term is formed only from legal query pairs belonging to the same
    mixture. It is intentionally a small, explicit helper rather than a second
    training framework. At lambda=0 the returned separation term is exactly the
    existing packed complex MSE and the common term is diagnostic-only.
    """
    import torch

    if prediction.shape != target.shape or prediction.shape != mixture_spec.shape:
        raise ValueError(
            f"loss shape mismatch: prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}, mixture={tuple(mixture_spec.shape)}"
        )
    if prediction.ndim != 4:
        raise ValueError("packed complex loss expects [B, 4, T, F]")
    if len(mixture_ids) != prediction.shape[0] or len(query_ids) != prediction.shape[0]:
        raise ValueError("mixture_ids and query_ids must match batch size")
    if common_loss_lambda < 0.0:
        raise ValueError("common_loss_lambda must be non-negative")

    separation_loss = packed_complex_mse(prediction, target)
    pair_indices = {}
    for index, mixture_id in enumerate(mixture_ids):
        pair_indices.setdefault(str(mixture_id), []).append(index)
    common_terms = []
    for mixture_id, indices in sorted(pair_indices.items()):
        if len(indices) != 2:
            raise ValueError(f"{mixture_id}: expected exactly two legal queries, got {len(indices)}")
        if query_ids[indices[0]] == query_ids[indices[1]]:
            raise ValueError(f"{mixture_id}: duplicate query ids in common-consistency pair")
        predicted_common = (prediction[indices[0]] + prediction[indices[1]]) / 2.0
        target_common = mixture_spec[indices[0]] / 2.0
        if not torch.allclose(mixture_spec[indices[0]], mixture_spec[indices[1]], atol=0.0, rtol=0.0):
            raise ValueError(f"{mixture_id}: paired queries do not share identical mixture input")
        common_terms.append(((predicted_common.float() - target_common.float()) ** 2).mean())
    common_loss = torch.stack(common_terms).mean() if common_terms else separation_loss.new_zeros(())
    return separation_loss + float(common_loss_lambda) * common_loss, common_loss

