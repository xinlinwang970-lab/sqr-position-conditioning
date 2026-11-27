import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
AUTHOR_ENTRYPOINT = "train_author_release_12ep_warmup_cosine.py"
LOCAL_EVALUATION_FUNCTIONS = {
    "compute_sdr",
    "inference_full_length",
    "evaluate_full_length",
    "summarize",
}


def _entrypoints():
    return sorted(SCRIPTS.glob("train_*.py")) + [
        SCRIPTS / "evaluate_checkpoint.py"
    ]


def test_all_evaluation_entrypoints_use_the_shared_legacy_v1_module():
    entrypoints = _entrypoints()
    assert len(entrypoints) == 33

    for path in entrypoints:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "sqr_repro.evaluation"
        ]
        assert len(imports) == 1, path
        imported_names = {alias.name for alias in imports[0].names}
        assert "evaluate_full_length_legacy_v1" in imported_names, path


def test_entrypoints_do_not_copy_the_evaluation_algorithm():
    for path in _entrypoints():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        local_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in LOCAL_EVALUATION_FUNCTIONS
        }
        expected = (
            {"evaluate_full_length"} if path.name == AUTHOR_ENTRYPOINT else set()
        )
        assert local_names == expected, path
