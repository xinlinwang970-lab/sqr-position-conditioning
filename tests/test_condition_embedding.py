import pytest

torch = pytest.importorskip("torch")

from sqr_repro.models.conditioning import ConditionEmbeddings


def test_condition_embedding_mean_and_independence():
    embeddings = ConditionEmbeddings(8)
    timbre = torch.tensor([0, 0, 1], dtype=torch.long)
    position = torch.tensor([1, 2, 1], dtype=torch.long)
    e_t, e_m, e_p = embeddings(timbre, position)
    assert torch.allclose(e_m, 0.5 * (e_t + e_p))
    assert torch.allclose(e_t[0], e_t[1])
    assert not torch.allclose(e_p[0], e_p[1])
    assert torch.allclose(e_p[0], e_p[2])
    assert not torch.allclose(e_t[0], e_t[2])


def test_condition_embedding_out_of_range_is_readable():
    embeddings = ConditionEmbeddings(8)
    with pytest.raises(IndexError):
        embeddings(torch.tensor([3], dtype=torch.long), torch.tensor([1], dtype=torch.long))

