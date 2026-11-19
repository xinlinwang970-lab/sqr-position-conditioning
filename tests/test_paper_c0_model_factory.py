import copy
import hashlib
import json
from pathlib import Path

import pytest

from sqr_repro.config import load_config
from sqr_repro.paper_c0.model_factory import ModelFactoryError, build_model


CONFIG = str(Path(__file__).resolve().parent / "fixtures/paper_c0_candidate_w24.yaml")


def candidate_config():
    config = load_config(CONFIG)
    config["experiment"]["run_mode"] = "w24_engineering_pilot"
    config["model"]["architecture_version"] = "c0_w24_candidate_v1"
    return config


def test_w24_factory_is_explicitly_engineering_only():
    model, report = build_model(candidate_config())
    assert report["run_mode"] == "w24_engineering_pilot"
    assert report["parameters"]["total"] == 2_241_292
    assert report["paper_gate"] == "NOT_APPLICABLE"
    assert report["optimizer_created"] is False
    assert model.count_parameters()["total"] == 2_241_292


def test_synthetic_smoke_is_a_distinct_mode():
    config = candidate_config()
    config["experiment"]["run_mode"] = "synthetic_smoke"
    _, report = build_model(config)
    assert report["run_mode"] == "synthetic_smoke"


def test_paper_mode_fails_closed_without_resolved_architecture_evidence():
    config = candidate_config()
    config["experiment"]["run_mode"] = "paper_reproduction"
    config["experiment"]["architecture_status"] = "UNRESOLVED"
    with pytest.raises(ModelFactoryError, match="RESOLVED"):
        build_model(config)


def test_paper_mode_never_accepts_w24_by_name_change_only():
    config = candidate_config()
    config["experiment"]["run_mode"] = "paper_reproduction"
    config["experiment"]["architecture_status"] = "RESOLVED"
    config["experiment"]["architecture_evidence_path"] = "does-not-exist.json"
    with pytest.raises(ModelFactoryError, match="unavailable"):
        build_model(config)


def resolved_paper_like_config(tmp_path):
    config = candidate_config()
    config["experiment"].update({
        "run_mode": "paper_reproduction",
        "architecture_status": "RESOLVED",
    })
    evidence_path = tmp_path / "evidence.json"
    config["experiment"]["architecture_evidence_path"] = str(evidence_path)
    evidence = {
        "schema_version": 1,
        "status": "RESOLVED",
        "architecture_id": "c0_w24_candidate_v1",
        "architecture_version": "c0_w24_candidate_v1",
        "implementation": "sqr_repro.models.separator:Separator",
        "config_sha256": "0" * 64,
        "key_specifications": {"n_blocks": 7},
        "parameter_audit": {"entries": [{"name": "fake", "parameters": 46_000_000}]},
        "source_hashes": {"src/sqr_repro/models/separator.py": "0" * 64},
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return config, evidence_path


def test_fake_resolved_evidence_is_rejected(tmp_path):
    config, _ = resolved_paper_like_config(tmp_path)
    with pytest.raises(ModelFactoryError):
        build_model(config)


def test_unrelated_fake_46m_audit_is_rejected(tmp_path):
    config, path = resolved_paper_like_config(tmp_path)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["architecture_id"] = "fake_46m"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ModelFactoryError, match="registry"):
        build_model(config)


def test_modified_source_hash_is_rejected(tmp_path):
    import sqr_repro.paper_c0.model_factory as factory

    config, path = resolved_paper_like_config(tmp_path)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["architecture_id"] = "test_paper"
    evidence["architecture_version"] = "test_paper_v1"
    config["model"]["architecture_version"] = "test_paper_v1"
    factory.ARCHITECTURE_REGISTRY["test_paper"] = {
        "architecture_version": "test_paper_v1",
        "implementation": "sqr_repro.models.separator:Separator",
        "paper_allowed": True,
    }
    path.write_text(json.dumps(evidence), encoding="utf-8")
    try:
        with pytest.raises(ModelFactoryError, match="source hash mismatch"):
            build_model(config)
    finally:
        del factory.ARCHITECTURE_REGISTRY["test_paper"]


def test_non_mapping_audit_and_hashes_are_rejected(tmp_path):
    config, path = resolved_paper_like_config(tmp_path)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["architecture_id"] = "unknown_paper"
    evidence["parameter_audit"] = "non-empty"
    evidence["source_hashes"] = "non-empty"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ModelFactoryError, match="registry"):
        build_model(config)


def test_mode_override_cannot_downgrade_paper_config():
    config = candidate_config()
    config["experiment"]["run_mode"] = "paper_reproduction"
    with pytest.raises(ModelFactoryError, match="override"):
        build_model(config, "synthetic_smoke")


def test_mode_override_cannot_upgrade_pilot_config():
    config = candidate_config()
    with pytest.raises(ModelFactoryError, match="override"):
        build_model(config, "paper_reproduction")


def test_paper_mode_rejects_score_encoder():
    config = candidate_config()
    config["experiment"].update({"run_mode": "paper_reproduction", "architecture_status": "UNRESOLVED"})
    config["score"] = {"enabled": True, "num_pitches": 128, "hidden_channels": 32, "injection": "pre_middle_add"}
    with pytest.raises(ModelFactoryError, match="RESOLVED"):
        build_model(config)


def test_model_requirements_return_isolated_copy():
    from sqr_repro.paper_c0.model_factory import model_requirements

    requirements = model_requirements()
    requirements["kernel_size"][0] = 99
    assert model_requirements()["kernel_size"] == [3, 3]


def test_unknown_run_mode_is_rejected():
    config = candidate_config()
    with pytest.raises(ModelFactoryError, match="run_mode"):
        build_model(config, "not_a_mode")
