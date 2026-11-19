"""Target-machine smoke checks; never starts real SQR training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from ..audio import AudioConfig, AudioTransform
from ..checkpoint import capture_rng_state, save_training_checkpoint, load_training_checkpoint
from ..config import load_config
from ..losses import packed_complex_mse
from .dataset import PaperC0WavDataset
from .model_factory import build_model
from .portable import run_portable_preflight


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_check(device: str = "cuda") -> dict[str, Any]:
    result = {"python": sys.version, "torch": torch.__version__, "device_requested": device, "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        result.update({
            "gpu": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "torch_arch_list": list(torch.cuda.get_arch_list()),
            "sm_120_visible": "sm_120" in torch.cuda.get_arch_list(),
        })
    else:
        result.update({"gpu": None, "capability": None, "torch_arch_list": [], "sm_120_visible": False})
    return result


def verify_source_hashes(path: str | Path) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    checks = {}
    for file_name, expected in bundle.items():
        file_path = Path(file_name)
        checks[file_name] = {"expected": expected, "actual": _file_hash(file_path) if file_path.is_file() else None, "match": file_path.is_file() and _file_hash(file_path) == expected}
    return {"all_match": all(item["match"] for item in checks.values()), "checks": checks}


def run_target_smoke(
    config_path: str | Path,
    out_path: str | Path,
    device: str = "cuda",
    source_hashes_path: str | Path | None = None,
    fixture_manifest: str | Path | None = None,
    fixture_root: str | Path | None = None,
    real_manifest: str | Path | None = None,
    real_root: str | Path | None = None,
    run_pytest: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    result: dict[str, Any] = {"status": "TARGET_SMOKE_INCOMPLETE", "optimizer_updates": 0, "environment": environment_check(device)}
    if device == "cuda" and not torch.cuda.is_available():
        result["environment_error"] = "CUDA unavailable"
    if source_hashes_path:
        result["source_hashes"] = verify_source_hashes(source_hashes_path)
    if run_pytest:
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, check=False)
        result["pytest"] = {"returncode": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]}
    if fixture_manifest and fixture_root:
        data = PaperC0WavDataset(fixture_manifest, fixture_root, "train", crop_samples=int(config["audio"]["segment_samples"]), crop_seed=2026, require_fixture=True)
        model, report = build_model(config, "w24_engineering_pilot")
        model.to(device).train()
        transform = AudioTransform(AudioConfig.from_dict(config["audio"]))
        batch = data[0]
        mixture = batch["mixture"].unsqueeze(0).to(device)
        target = batch["target"].unsqueeze(0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]), foreach=False)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        optimizer.zero_grad(set_to_none=True)
        prediction = model(transform.stft(mixture), batch["timbre_id"].reshape(1).to(device), batch["position_id"].reshape(1).to(device))
        loss = packed_complex_mse(prediction, transform.stft(target))
        loss.backward()
        optimizer.step()
        changed = any(not torch.equal(before[key], value) for key, value in model.state_dict().items())
        result["w24_forward_backward"] = {"pass": bool(torch.isfinite(loss).item() and changed), "loss": float(loss.detach().cpu()), "model_report": report}
        if torch.cuda.is_available() and device == "cuda":
            result["memory"] = {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
        checkpoint = Path(out_path).with_suffix(".smoke.pt")
        save_training_checkpoint(checkpoint, model, optimizer, config, 1, capture_rng_state(), {"dataset_length": len(data), "batch_size": 1, "cursor": 1}, metadata={"architecture_identity": report["architecture_version"]}, attempted_updates=1, max_attempted_updates=1)
        restored, _ = build_model(config, "w24_engineering_pilot")
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=float(config["training"]["learning_rate"]), foreach=False)
        load_training_checkpoint(checkpoint, restored, restored_optimizer, expected_config=config, map_location=device)
        result["checkpoint_roundtrip"] = {"pass": True, "path": str(checkpoint)}
    if real_manifest and real_root:
        result["real_io_preflight"] = run_portable_preflight(config_path, real_manifest, real_root, split="train", require_audio=True)
        result["real_io_update_count"] = 0
    result["status"] = "TARGET_SMOKE_PASS" if result.get("w24_forward_backward", {}).get("pass", True) and result.get("source_hashes", {"all_match": True}).get("all_match", True) else "TARGET_SMOKE_FAIL"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
