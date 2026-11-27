"""Model construction and strict Paper C0 run-mode boundaries."""

from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from typing import Any

from ..models.separator import Separator


RUN_MODES = {"synthetic_smoke", "w24_engineering_pilot", "paper_reproduction"}
PAPER_REQUIREMENTS = {
    "n_blocks": 7,
    "tfc_layers": 5,
    "kernel_size": [3, 3],
    "tfc_growth_rate": 24,
    "num_timbres": 3,
    "num_positions": 4,
    "embedding_dim": 32,
    "condition_route": "split_stage_paper",
    "modulation": "gpocm",
    "output_mode": "complex_mapping",
}

# Only audited implementations may enter paper_reproduction.  The current
# candidate is deliberately registered as non-paper; no Paper C0 architecture
# is registered until the unresolved 46M structure has source-backed evidence.
ARCHITECTURE_REGISTRY = {
    "c0_w24_candidate_v1": {
        "architecture_version": "c0_w24_candidate_v1",
        "implementation": "sqr_repro.models.separator:Separator",
        "paper_allowed": False,
    },
    "paper_route_w24_v1": {
        "architecture_version": "paper_route_w24_v1",
        "implementation": "sqr_repro.models.separator:Separator",
        "paper_allowed": False,
    },
    "growth24_projected_a_v1": {
        "architecture_version": "growth24_projected_a_v1",
        "implementation": "sqr_repro.models.separator:Separator",
        "paper_allowed": False,
    },
    "growth24_projected_b_v1": {
        "architecture_version": "growth24_projected_b_v1",
        "implementation": "sqr_repro.models.separator:Separator",
        "paper_allowed": False,
    },
    "growth24_projected_c_v1": {
        "architecture_version": "growth24_projected_c_v1",
        "implementation": "sqr_repro.models.separator:Separator",
        "paper_allowed": False,
    },
}
EVIDENCE_SCHEMA_VERSION = 1


class ModelFactoryError(ValueError):
    pass


def model_requirements() -> dict[str, Any]:
    return copy.deepcopy(PAPER_REQUIREMENTS)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_architecture_evidence(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    evidence_path = experiment.get("architecture_evidence_path")
    if not evidence_path:
        raise ModelFactoryError("paper_reproduction requires architecture_evidence_path")
    path = Path(evidence_path)
    if not path.is_file():
        raise ModelFactoryError(f"architecture evidence is unavailable: {path}")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelFactoryError(f"architecture evidence cannot be read: {path}") from exc
    required = {
        "schema_version", "status", "architecture_id", "architecture_version",
        "implementation", "config_sha256", "key_specifications",
        "parameter_audit", "source_hashes",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ModelFactoryError("architecture evidence schema mismatch")
    if evidence["schema_version"] != EVIDENCE_SCHEMA_VERSION or evidence["status"] != "RESOLVED":
        raise ModelFactoryError("architecture evidence is not a resolved schema-v1 record")
    registry = ARCHITECTURE_REGISTRY.get(evidence["architecture_id"])
    if registry is None:
        raise ModelFactoryError("architecture_id is not in the audited registry")
    if not registry["paper_allowed"]:
        raise ModelFactoryError("architecture_id is not approved for paper_reproduction")
    if evidence["architecture_version"] != registry["architecture_version"]:
        raise ModelFactoryError("architecture_version does not match registry")
    if config.get("model", {}).get("architecture_version") != evidence["architecture_version"]:
        raise ModelFactoryError("evidence architecture identity does not match config")
    if evidence["implementation"] != registry["implementation"]:
        raise ModelFactoryError("implementation does not match registry")
    if not isinstance(evidence["key_specifications"], dict) or any(
        value == "UNKNOWN" for value in evidence["key_specifications"].values()
    ):
        raise ModelFactoryError("key specifications contain UNKNOWN or invalid data")
    parameter_audit = evidence["parameter_audit"]
    if not isinstance(parameter_audit, dict) or not isinstance(parameter_audit.get("entries"), list):
        raise ModelFactoryError("parameter_audit must contain an entries list")
    source_hashes = evidence["source_hashes"]
    if not isinstance(source_hashes, dict) or not source_hashes or any(
        not isinstance(path_name, str) or not isinstance(value, str) or len(value) != 64
        for path_name, value in source_hashes.items()
    ):
        raise ModelFactoryError("source_hashes must be a non-empty path-to-sha256 mapping")
    for path_name, expected_hash in source_hashes.items():
        source_path = Path(path_name)
        if not source_path.is_absolute():
            candidates = [Path.cwd() / source_path, path.parent / source_path]
            source_path = next((candidate for candidate in candidates if candidate.is_file()), source_path)
        if not source_path.is_file() or _hash_file(source_path) != expected_hash:
            raise ModelFactoryError(f"source hash mismatch: {path_name}")
    return {"path": str(path.resolve()), "sha256": _hash_file(path), "evidence": evidence}


def _check_paper_requirements(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    mismatches = {}
    for key, expected in PAPER_REQUIREMENTS.items():
        actual = model.get(key)
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ModelFactoryError(f"paper architecture requirements mismatch: {mismatches}")
    score = config.get("score") or {}
    if score.get("enabled", False):
        raise ModelFactoryError("paper_reproduction must not enable Score Encoder")
    return _load_architecture_evidence(config)


def _validate_bound_evidence(config: dict[str, Any], model: Separator, loaded: dict[str, Any]) -> None:
    evidence = loaded["evidence"]
    if evidence["config_sha256"] != _config_sha256(config):
        raise ModelFactoryError("architecture evidence config hash mismatch")
    actual_entries = [
        {"name": name, "shape": list(parameter.shape), "parameters": int(parameter.numel())}
        for name, parameter in model.named_parameters()
    ]
    audit = evidence["parameter_audit"]
    if audit.get("total") != sum(entry["parameters"] for entry in actual_entries):
        raise ModelFactoryError("architecture parameter total mismatch")
    if audit.get("entries") != actual_entries:
        raise ModelFactoryError("architecture parameter audit does not match model")


def build_model(config: dict[str, Any], run_mode: str | None = None) -> tuple[Separator, dict[str, Any]]:
    """Build a separator and return an auditable construction report.

    This function does not create an optimizer and does not run forward/backward.
    """
    declared_mode = config.get("experiment", {}).get("run_mode")
    if run_mode is not None and declared_mode is not None and run_mode != declared_mode:
        raise ModelFactoryError("explicit run_mode override disagrees with config run_mode")
    mode = run_mode or declared_mode
    if mode not in RUN_MODES:
        raise ModelFactoryError(f"run_mode must be one of {sorted(RUN_MODES)}")
    evidence = None
    if mode == "paper_reproduction":
        if config.get("experiment", {}).get("architecture_status") != "RESOLVED":
            raise ModelFactoryError("paper_reproduction requires architecture_status=RESOLVED")
        evidence = _check_paper_requirements(config)
    model = Separator(config)
    if mode == "paper_reproduction":
        _validate_bound_evidence(config, model, evidence)
    report = {
        "run_mode": mode,
        "architecture_version": config["model"].get("architecture_version", "unspecified"),
        "parameters": model.count_parameters(),
        "optimizer_created": False,
        "backward_called": False,
        "paper_gate": "PASS" if mode == "paper_reproduction" else "NOT_APPLICABLE",
        "architecture_evidence": evidence,
    }
    return model, report
