import copy
import json
import hashlib
from pathlib import Path

import torch

from sqr_repro.fixtures import make_fixtures
from sqr_repro.paper_c0.dataset import PaperC0WavDataset
from sqr_repro.paper_c0.trainer import PaperC0SyntheticTrainer, PaperC0TrainerError
from sqr_repro.config import load_config


def make_training_fixture(tmp_path: Path):
    root = tmp_path / "train"
    generated = make_fixtures(root, seed=31, sample_rate=8000, seconds=0.6)
    record = json.loads(generated.read_text(encoding="utf-8"))
    record["item_id"] = "trainer-fixture"
    record["composition_id"] = "trainer-work"
    record["performance_id"] = "trainer-take"
    generated.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = load_config(str(Path(__file__).resolve().parent / "fixtures/paper_c0_candidate_w24.yaml"))
    config["experiment"]["run_mode"] = "synthetic_smoke"
    config["audio"].update({"sample_rate": 8000, "n_fft": 256, "win_length": 256, "hop_length": 128, "num_frames": 32})
    config["training"].update({"batch_size": 2, "gradient_accumulation_steps": 1, "precision": "fp32", "foreach": False})
    dataset = PaperC0WavDataset(generated, root, "train", crop_samples=3968, crop_seed=123, require_fixture=True)
    return config, dataset


def flatten_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_digest(value):
    if isinstance(value, torch.Tensor):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if isinstance(value, dict):
        return {key: state_digest(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [state_digest(item) for item in value]
    return value


def test_synthetic_trainer_updates_and_writes_full_checkpoint(tmp_path):
    config, dataset = make_training_fixture(tmp_path)
    trainer = PaperC0SyntheticTrainer(config, dataset)
    rows = trainer.train_updates(2)
    assert len(rows) == 2
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in rows)
    assert trainer.completed_updates == 2
    assert rows[0]["batch_indices"] == [0, 1]
    assert rows[1]["batch_indices"] == [2, 3]
    assert rows[0]["batch_indices"] != rows[1]["batch_indices"]
    assert len(rows[0]["query_ids"]) == 2
    assert len(rows[0]["crop_offsets"]) == 2
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert {"model_state_dict", "optimizer_state_dict", "rng_state", "sampler_state", "completed_updates"}.issubset(payload)


def test_ten_plus_ten_resume_matches_continuous_twenty(tmp_path):
    config, dataset = make_training_fixture(tmp_path)
    torch.manual_seed(55)
    continuous = PaperC0SyntheticTrainer(config, dataset)
    continuous_rows = continuous.train_updates(20)
    continuous_state = flatten_state(continuous.model)
    continuous_optimizer = continuous.optimizer.state_dict()

    config2, dataset2 = make_training_fixture(tmp_path / "branch")
    torch.manual_seed(55)
    resumed = PaperC0SyntheticTrainer(config2, dataset2)
    resumed.train_updates(10)
    checkpoint = tmp_path / "resume.pt"
    resumed.save(checkpoint)
    restored_config, restored_dataset = make_training_fixture(tmp_path / "branch")
    torch.manual_seed(999)
    restored = PaperC0SyntheticTrainer(restored_config, restored_dataset)
    restored.load(checkpoint)
    restored_rows = restored.train_updates(10)
    restored_state = flatten_state(restored.model)
    for key in continuous_state:
        assert torch.equal(continuous_state[key], restored_state[key]), key
    assert continuous.optimizer.state_dict()["state"].keys() == restored.optimizer.state_dict()["state"].keys()
    assert continuous_rows[10:] == restored_rows
    assert state_digest(continuous_optimizer) == state_digest(restored.optimizer.state_dict())
    assert continuous.sampler.state_dict() == restored.sampler.state_dict()
    assert continuous.completed_updates == restored.completed_updates == 20
    assert continuous.attempted_updates == restored.attempted_updates == 20
    assert continuous_rows[9]["completed_updates"] == 10
    assert restored_rows[-1]["completed_updates"] == 20


def test_trainer_rejects_non_synthetic_mode(tmp_path):
    config, dataset = make_training_fixture(tmp_path)
    config["experiment"]["run_mode"] = "w24_engineering_pilot"
    try:
        PaperC0SyntheticTrainer(config, dataset)
    except PaperC0TrainerError as exc:
        assert "synthetic" in str(exc)
    else:
        raise AssertionError("synthetic trainer must reject pilot mode")


def test_digest_covers_audio_identity_and_training_contract(tmp_path):
    config, dataset = make_training_fixture(tmp_path)
    trainer = PaperC0SyntheticTrainer(config, dataset)
    original = trainer.dataset_digest
    dataset.records[0]["mixture_hash"] = "changed"
    changed_audio = PaperC0SyntheticTrainer(config, dataset).dataset_digest
    assert changed_audio != original
    dataset.records[0]["mixture_hash"] = ""
    dataset.records[0]["position_ids"]["viola"] += 1
    changed_position = PaperC0SyntheticTrainer(config, dataset).dataset_digest
    assert changed_position != original
    dataset.records[0]["position_ids"]["viola"] -= 1
    dataset.crop_samples += 1
    changed_crop = PaperC0SyntheticTrainer(config, dataset).dataset_digest
    assert changed_crop != original


def test_checkpoint_rejects_wrong_dataset_without_mutating_state(tmp_path):
    config, dataset = make_training_fixture(tmp_path / "source")
    trainer = PaperC0SyntheticTrainer(config, dataset)
    trainer.train_updates(1)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    wrong_config, wrong_dataset = make_training_fixture(tmp_path / "wrong")
    wrong = PaperC0SyntheticTrainer(wrong_config, wrong_dataset)
    before = flatten_state(wrong.model)
    before_sampler = wrong.sampler.state_dict().copy()
    try:
        wrong.load(path)
    except PaperC0TrainerError as exc:
        assert "dataset digest" in str(exc)
    else:
        raise AssertionError("wrong dataset must be rejected")
    assert all(torch.equal(before[key], wrong.model.state_dict()[key].cpu()) for key in before)
    assert wrong.sampler.state_dict() == before_sampler
    assert wrong.completed_updates == 0


def test_checkpoint_rejects_bad_sampler_architecture_and_rng_atomically(tmp_path):
    config, dataset = make_training_fixture(tmp_path)
    trainer = PaperC0SyntheticTrainer(config, dataset)
    trainer.train_updates(1)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    original = torch.load(path, map_location="cpu", weights_only=False)
    for mutation, message in (
        (lambda payload: payload["sampler_state"].update(cursor=99), "sampler"),
        (lambda payload: payload["metadata"].update(architecture_identity="wrong"), "architecture"),
        (lambda payload: payload.update(rng_state={"bad": True}), "RNG"),
    ):
        payload = copy.deepcopy(original)
        mutation(payload)
        bad_path = tmp_path / f"bad_{message}.pt"
        torch.save(payload, bad_path)
        candidate = PaperC0SyntheticTrainer(config, dataset)
        before = flatten_state(candidate.model)
        try:
            candidate.load(bad_path)
        except PaperC0TrainerError as exc:
            assert message.lower() in str(exc).lower()
        else:
            raise AssertionError(f"bad {message} must be rejected")
        assert all(torch.equal(before[key], candidate.model.state_dict()[key].cpu()) for key in before)
        assert candidate.sampler.state_dict()["cursor"] == 0


def test_failed_update_is_fail_stopped_and_does_not_commit(tmp_path, monkeypatch):
    config, dataset = make_training_fixture(tmp_path)
    trainer = PaperC0SyntheticTrainer(config, dataset)
    monkeypatch.setattr(trainer, "_batch", lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    try:
        trainer.update()
    except RuntimeError as exc:
        assert "fixture failure" in str(exc)
    else:
        raise AssertionError("injected update failure must propagate")
    assert trainer.attempted_updates == 1
    assert trainer.completed_updates == 0
    assert trainer.sampler.cursor == 0
    assert trainer.failed is True
    try:
        trainer.update()
    except PaperC0TrainerError as exc:
        assert "fail-stopped" in str(exc)
    else:
        raise AssertionError("failed trainer must not silently retry")


def test_attempted_budget_is_bounded(tmp_path):
    config, dataset = make_training_fixture(tmp_path)
    config["training"]["max_attempted_updates"] = 1
    trainer = PaperC0SyntheticTrainer(config, dataset)
    trainer.update()
    try:
        trainer.update()
    except PaperC0TrainerError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("attempt budget must stop the trainer")
    assert trainer.attempted_updates == trainer.completed_updates == 1
