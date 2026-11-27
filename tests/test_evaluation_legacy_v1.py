import pytest


torch = pytest.importorskip("torch")

from sqr_repro.evaluation import (  # noqa: E402
    compute_wavesdr,
    inference_full_length_legacy_v1,
    summarize_wavesdr,
)


class _EchoModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def __call__(self, spec, timbre, position):
        return spec


class _IdentityTransform:
    def stft(self, waveform):
        return waveform.unsqueeze(-1)

    def istft(self, spec, length):
        return spec[..., 0][..., :length]


def _historical_reference(
    model,
    mixture,
    transform,
    timbre_id,
    position_id,
    device,
    num_frames,
    hop_length,
    overlap_ratio=0.5,
    *,
    pad_time_to_multiple_of_8=True,
):
    channels, total_samples = mixture.shape
    segment_samples = (num_frames - 1) * hop_length
    hop_samples = int(segment_samples * (1 - overlap_ratio))
    if total_samples < segment_samples:
        mixture = torch.nn.functional.pad(
            mixture, (0, segment_samples - total_samples)
        )
        total_samples = mixture.shape[-1]
    output = torch.zeros(channels, total_samples, dtype=torch.float32)
    weight = torch.zeros(total_samples, dtype=torch.float32)
    window = torch.hann_window(segment_samples, periodic=True)
    starts = list(
        range(0, max(total_samples - segment_samples + 1, 1), hop_samples)
    )
    if not starts or starts[-1] + segment_samples < total_samples:
        starts.append(max(total_samples - segment_samples, 0))
    for start in starts:
        end = min(start + segment_samples, total_samples)
        chunk = mixture[:, start:end]
        if chunk.shape[-1] < segment_samples:
            chunk = torch.nn.functional.pad(
                chunk, (0, segment_samples - chunk.shape[-1])
            )
        spec = transform.stft(chunk.unsqueeze(0).to(device))
        if pad_time_to_multiple_of_8:
            time_frames = spec.shape[2]
            pad_frames = (8 - time_frames % 8) % 8
            if pad_frames:
                spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_frames))
        timbre = torch.tensor([timbre_id], dtype=torch.long, device=device)
        position = torch.tensor([position_id], dtype=torch.long, device=device)
        with torch.no_grad():
            estimate = model(spec, timbre, position)
        wave = transform.istft(estimate, length=segment_samples)[0].cpu()
        blend = window[: end - start]
        output[:, start:end] += wave[:, : end - start] * blend
        weight[start:end] += blend
    return (output / torch.clamp(weight, min=1e-8))[:, :total_samples]


@pytest.mark.parametrize("length", [5, 17])
@pytest.mark.parametrize("pad_time", [False, True])
def test_legacy_v1_inference_matches_frozen_historical_reference(length, pad_time):
    mixture = torch.arange(2 * length, dtype=torch.float32).reshape(2, length) / 10
    expected = _historical_reference(
        _EchoModel(),
        mixture,
        _IdentityTransform(),
        0,
        3,
        "cpu",
        4,
        2,
        pad_time_to_multiple_of_8=pad_time,
    )
    actual = inference_full_length_legacy_v1(
        _EchoModel(),
        mixture,
        _IdentityTransform(),
        0,
        3,
        "cpu",
        4,
        2,
        pad_time_to_multiple_of_8=pad_time,
    )
    assert torch.equal(actual, expected)


def test_wavesdr_and_summary_keep_historical_field_contract():
    reference = torch.tensor([1.0, 2.0])
    estimate = torch.tensor([1.0, 1.0])
    expected = 10 * torch.log10(torch.tensor(5.0 / (1.0 + 1.0e-8)))
    assert torch.equal(compute_wavesdr(estimate, reference), expected)

    rows = [
        {"layout_id": "type1", "sdr": -1.0},
        {"layout_id": "type1", "sdr": 1.0},
        {"layout_id": "type2", "sdr": 2.0},
        {"layout_id": "type2", "sdr": 4.0},
    ]
    assert summarize_wavesdr(rows) == {
        "mean_sdr": 1.5,
        "median_sdr": 1.5,
        "type1_median": 0.0,
        "type2_median": 3.0,
        "n": 4,
    }
