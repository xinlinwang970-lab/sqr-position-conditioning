from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sqr_repro.config import load_config
from sqr_repro.models.separator import Separator


ROOT = Path(__file__).resolve().parents[1]


class SpyController(torch.nn.Module):
    def __init__(self, channels, n_stages):
        super().__init__()
        self.channels = channels
        self.n_stages = n_stages
        self.seen = None

    def forward(self, embedding):
        self.seen = embedding.detach().clone()
        batch = embedding.shape[0]
        return [
            (
                torch.zeros(batch, self.channels, self.channels, device=embedding.device),
                torch.zeros(batch, self.channels, device=embedding.device),
            )
            for _ in range(self.n_stages)
        ]


def test_split_stage_routes_e_m_to_all_controllers():
    # Legacy behavior: e_m (mean of e_t and e_p) is broadcast to encoder/middle/decoder.
    config = load_config(ROOT / "tests" / "fixtures" / "audit_math_tiny.yaml")
    model = Separator(config).eval()
    c = config["model"]["base_channels"]
    model.encoder_controller = SpyController(c, 3)
    model.middle_controller = SpyController(c, 1)
    model.decoder_controller = SpyController(c, 3)
    spec = torch.randn(2, 4, 8, 33)
    timbre = torch.tensor([0, 1], dtype=torch.long)
    position = torch.tensor([3, 1], dtype=torch.long)
    _, debug = model(spec, timbre, position, return_debug=True)
    _, e_m, _ = model.embeddings(timbre, position)
    assert torch.allclose(model.encoder_controller.seen, e_m)
    assert torch.allclose(model.middle_controller.seen, e_m)
    assert torch.allclose(model.decoder_controller.seen, e_m)
    assert debug["condition_route"] == "split_stage"
    assert debug["output_shape"] == [2, 4, 8, 33]


def test_split_stage_paper_routes_timbre_to_encoder_position_to_decoder():
    # Paper §4.2: encoder<-e_t, middle<-e_m (=0.5*(e_t+e_p)), decoder<-e_p
    config = load_config(ROOT / "tests" / "fixtures" / "audit_math_tiny.yaml")
    config["model"]["condition_route"] = "split_stage_paper"
    model = Separator(config).eval()
    c = config["model"]["base_channels"]
    model.encoder_controller = SpyController(c, 3)
    model.middle_controller = SpyController(c, 1)
    model.decoder_controller = SpyController(c, 3)
    spec = torch.randn(2, 4, 8, 33)
    timbre = torch.tensor([0, 1], dtype=torch.long)
    position = torch.tensor([3, 1], dtype=torch.long)
    _, debug = model(spec, timbre, position, return_debug=True)
    e_t, e_m, e_p = model.embeddings(timbre, position)
    assert torch.allclose(model.encoder_controller.seen, e_t)
    assert torch.allclose(model.middle_controller.seen, e_m)
    assert torch.allclose(model.decoder_controller.seen, e_p)
    assert debug["condition_route"] == "split_stage_paper"
    assert debug["output_shape"] == [2, 4, 8, 33]


def test_mean_decoder_leaves_encoder_and_middle_unconditioned():
    config = load_config(ROOT / "tests" / "fixtures" / "audit_math_tiny.yaml")
    config["model"]["condition_route"] = "mean_decoder"
    model = Separator(config).eval()
    c = config["model"]["base_channels"]
    model.decoder_controller = SpyController(c, 3)
    spec = torch.randn(2, 4, 8, 33)
    timbre = torch.tensor([0, 1], dtype=torch.long)
    position = torch.tensor([3, 1], dtype=torch.long)
    _, debug = model(spec, timbre, position, return_debug=True)
    _, e_m, _ = model.embeddings(timbre, position)
    assert torch.allclose(model.decoder_controller.seen, e_m)
    assert debug["condition_parameter_shapes"]["encoder"] == [None, None, None]
    assert debug["condition_parameter_shapes"]["middle"] == [None]

