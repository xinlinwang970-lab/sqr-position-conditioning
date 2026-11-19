import torch

import sqr_repro.paper_c0.evaluator as evaluator_module
from sqr_repro.paper_c0.evaluator import PaperC0EvaluatorError, evaluate_model
from sqr_repro.paper_c0.trainer import PaperC0SyntheticTrainer
from test_paper_c0_trainer import make_training_fixture, flatten_state


def make_eval_fixture(tmp_path):
    config, dataset = make_training_fixture(tmp_path)
    dataset.all_targets = True
    trainer = PaperC0SyntheticTrainer(config, dataset)
    return config, dataset, trainer


def test_paper_c0_evaluator_reports_baselines_and_zero_updates(tmp_path):
    config, dataset, trainer = make_eval_fixture(tmp_path)
    before = flatten_state(trainer.model)
    result = evaluate_model(trainer.model, dataset, config, device="cpu", batch_size=1)
    assert result["optimizer_updates"] == 0
    assert result["summary"]["query_count"] == len(dataset)
    assert result["summary"]["blind_R"] >= 0.0
    assert result["summary"]["zero_R"] == 1.0
    assert result["model_state_hash_before"] == result["model_state_hash_after"]
    assert result["model_training_mode_restored"] is True
    assert all(torch.equal(before[key], trainer.model.state_dict()[key].cpu()) for key in before)


def test_paper_c0_evaluator_requires_all_targets_and_rejects_nonfinite(tmp_path):
    config, dataset, trainer = make_eval_fixture(tmp_path)
    dataset.all_targets = False
    try:
        evaluate_model(trainer.model, dataset, config)
    except PaperC0EvaluatorError as exc:
        assert "all_targets" in str(exc)
    else:
        raise AssertionError("evaluator must require full target diagnostics")


def test_paper_c0_evaluator_restores_mode_and_rng_after_exception(tmp_path, monkeypatch):
    config, dataset, trainer = make_eval_fixture(tmp_path)
    trainer.model.train(False)
    original = flatten_state(trainer.model)
    original_rng = torch.get_rng_state().clone()
    monkeypatch.setattr(evaluator_module.AudioTransform, "stft", lambda self, value: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    try:
        evaluate_model(trainer.model, dataset, config)
    except RuntimeError as exc:
        assert "fixture failure" in str(exc)
    else:
        raise AssertionError("injected evaluator failure must propagate")
    assert trainer.model.training is False
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert all(torch.equal(original[key], trainer.model.state_dict()[key].cpu()) for key in original)
