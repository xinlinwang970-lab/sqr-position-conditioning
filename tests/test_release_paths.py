import json
from pathlib import Path

import pytest

from sqr_repro.release_paths import prepare_training, validate_inputs


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/PureSwap_E1cosine-P15_1000ep.yaml"


def test_release_defaults_resolve_from_repo_when_cwd_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config, args = prepare_training(CONFIG, repo_root=REPO, argv=[])
    assert Path(config["data"]["root"]) == REPO / "data"
    assert args.train_manifest == REPO / "manifests/sqr_paper_split_train.jsonl"
    assert Path(config["paths"]["run_root"]).is_absolute()


def test_explicit_root_and_split_paths_override_defaults(tmp_path):
    train = tmp_path / "custom.jsonl"
    config, args = prepare_training(CONFIG, repo_root=REPO, argv=[
        "--data-root", str(tmp_path / "audio"), "--train-manifest", str(train),
        "--test-data-root", str(tmp_path / "real"), "--run-root", str(tmp_path / "runs"),
    ])
    assert args.train_manifest == train
    assert args.test_data_root == tmp_path / "real"
    assert config["data"]["root"] == str(tmp_path / "audio")
    assert config["data"]["manifest"] == str(train)


def test_custom_dataset_requires_explicit_roots_and_manifests():
    with pytest.raises(SystemExit):
        prepare_training(CONFIG, repo_root=REPO, custom=True, argv=[])


def test_preflight_rejects_wrong_train_count_before_training(tmp_path):
    config, args = prepare_training(CONFIG, repo_root=REPO, argv=[])
    args.train_manifest = tmp_path / "train.jsonl"
    args.train_manifest.write_text(json.dumps({"item_id": "one"}) + "\n")
    with pytest.raises(ValueError, match="59"):
        validate_inputs(config, args, expected_train_records=59)
