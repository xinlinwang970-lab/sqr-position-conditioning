import gc
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sqr_repro.config import ConfigError, load_config, validate_config
from sqr_repro.models.separator import Separator
from sqr_repro.paper_c0.model_factory import ModelFactoryError, build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_w24_reference_candidate_is_explicit_and_counted():
    config = load_config(CONFIGS / "paper_route_w24_v1.yaml")
    assert config["model"]["base_channels"] == 24
    assert config["model"]["tfc_growth_rate"] == 24
    assert config["model"]["condition_route"] == "split_stage_paper"

    model = Separator(config)
    assert model.count_parameters()["total"] == 2_241_292
    assert model.stage_channels == (24, 24, 24, 24)
    assert model.projected_growth24 is False


def test_projected_b_candidate_count_shapes_route_and_gradients():
    config = load_config(CONFIGS / "growth24_projected_b_v1.yaml")
    count_model = Separator(config)
    assert count_model.count_parameters()["total"] == 46_017_316
    del count_model
    gc.collect()

    config["audio"].update(
        {"n_fft": 64, "win_length": 64, "hop_length": 16, "num_frames": 8}
    )
    model = Separator(config)
    assert model.stage_channels == (96, 144, 144, 240)
    assert model.tfc_growth_rate == 24

    # Dense layers add 24 channels, while the projection sets each block's output width.
    enc_tfc = model.encoders[0].tfc
    assert [layer[0].in_channels for layer in enc_tfc.layers] == [96, 120, 144, 168, 192]
    assert all(layer[0].out_channels == 24 for layer in enc_tfc.layers)
    assert enc_tfc.projection[0].in_channels == 216
    assert enc_tfc.projection[0].out_channels == 96
    dec_tfc = model.decoders[0].tfc
    assert [layer[0].in_channels for layer in dec_tfc.layers] == [288, 312, 336, 360, 384]
    assert dec_tfc.projection[0].in_channels == 408
    assert dec_tfc.projection[0].out_channels == 144

    model.train()
    spec = torch.randn(1, 4, 8, 33)
    timbre = torch.tensor([1], dtype=torch.long)
    position = torch.tensor([2], dtype=torch.long)
    output, debug = model(spec, timbre, position, return_debug=True)
    assert output.shape == spec.shape
    assert torch.isfinite(output).all()
    assert [item["weight"][1] for item in debug["condition_parameter_shapes"]["encoder"]] == [96, 144, 144]
    assert debug["condition_parameter_shapes"]["middle"][0]["weight"] == [1, 240, 240]
    assert [item["weight"][1] for item in debug["condition_parameter_shapes"]["decoder"]] == [144, 144, 96]

    output.square().mean().backward()
    gradients = [
        model.embeddings.timbre.weight.grad,
        model.embeddings.position.weight.grad,
        model.encoders[0].tfc.layers[0][0].weight.grad,
        model.encoders[0].tfc.projection[0].weight.grad,
        model.encoder_controller.net[-1].weight.grad,
        model.middle_controller.net[-1].weight.grad,
        model.decoder_controller.net[-1].weight.grad,
        model.head.weight.grad,
    ]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)


def test_projected_candidate_rejects_unregistered_width_schedule():
    config = deepcopy(load_config(CONFIGS / "growth24_projected_b_v1.yaml"))
    config["model"]["stage_channels"] = [96, 144, 168, 240]
    with pytest.raises(ConfigError, match="fixes stage_channels"):
        validate_config(config)


def test_candidate_configs_lock_parameter_affecting_model_settings():
    config = deepcopy(load_config(CONFIGS / "growth24_projected_b_v1.yaml"))
    config["model"]["tfc_layers"] = 4
    with pytest.raises(ConfigError, match="fixes its audited model settings"):
        validate_config(config)


def test_growth24_candidates_are_not_admitted_as_paper_reproductions():
    config = deepcopy(load_config(CONFIGS / "paper_route_w24_v1.yaml"))
    config["experiment"]["run_mode"] = "paper_reproduction"
    with pytest.raises(ModelFactoryError, match="architecture_status=RESOLVED"):
        build_model(config)


def test_existing_c124_variants_keep_their_architecture_and_counts():
    mid_dual = load_config(CONFIGS / "PureSwap_E1cosine-P15_1000ep.yaml")
    mid_dual_model = Separator(mid_dual)
    assert mid_dual_model.route == "mid_dual_paper"
    assert mid_dual_model.tfc_growth_rate == 124
    assert mid_dual_model.count_parameters()["total"] == 30_441_288
    del mid_dual_model
    gc.collect()

    author_release = load_config(CONFIGS / "author_release_12ep_warmup_cosine.yaml")
    author_release_model = Separator(author_release)
    assert author_release_model.route == "split_stage_paper"
    assert author_release_model.tfc_growth_rate == 124
    assert author_release_model.count_parameters()["total"] == 46_463_992
