from pathlib import Path

import pytest
import yaml

from sqr_repro.config import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
ARTIFACTS = ROOT / "test_artifacts"


def test_debug_config_derives_segment_samples():
    config = load_config(FIXTURES / "debug_cpu.yaml")
    assert config["audio"]["segment_samples"] == 7 * 64


def test_unknown_key_is_rejected():
    ARTIFACTS.mkdir(exist_ok=True)
    config = yaml.safe_load((FIXTURES / "debug_cpu.yaml").read_text(encoding="utf-8"))
    config["model"]["surprise"] = True
    path = ARTIFACTS / "bad_unknown_key.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(path)


def test_batchnorm_training_batch_one_is_rejected():
    ARTIFACTS.mkdir(exist_ok=True)
    config = yaml.safe_load((FIXTURES / "reimpl_w24.yaml").read_text(encoding="utf-8"))
    config["training"]["batch_size"] = 1
    path = ARTIFACTS / "bad_batch.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="BatchNorm1d"):
        load_config(path)


def test_all_checked_in_configs_declare_exact_legacy_v1_evaluation():
    paths = sorted((ROOT / "configs").glob("*.yaml")) + sorted(FIXTURES.glob("*.yaml"))
    assert paths
    for path in paths:
        inference = load_config(path)["inference"]
        assert inference["evaluation_protocol"] == "legacy_v1", path
        assert inference["overlap_ratio_requested"] == 0.5, path
        assert inference["align_stride_to_stft_hop"] is False, path
        assert inference["blend_window"] == "hann", path
        assert inference["blend_weight_floor"] == 1.0e-8, path
        assert inference["keep_original_length"] is True, path


def test_legacy_v1_rejects_nonhistorical_inference_settings(tmp_path):
    config = yaml.safe_load(
        (FIXTURES / "debug_cpu.yaml").read_text(encoding="utf-8")
    )
    config["inference"]["align_stride_to_stft_hop"] = True
    path = tmp_path / "bad_legacy_inference.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="legacy_v1"):
        load_config(path)
