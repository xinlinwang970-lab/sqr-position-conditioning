from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .fixtures import make_fixtures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sqr_repro")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--out", required=True)

    fixtures = sub.add_parser("make-fixtures")
    fixtures.add_argument("--out", required=True)
    fixtures.add_argument("--seed", type=int, default=2026)

    inspect_model = sub.add_parser("inspect-model")
    inspect_model.add_argument("--config", required=True)
    inspect_model.add_argument("--out", default="docs/architecture_report.md")

    paper_c0_preflight = sub.add_parser("paper-c0-preflight")
    paper_c0_preflight.add_argument("--config", required=True)
    paper_c0_preflight.add_argument("--manifest", required=True)
    paper_c0_preflight.add_argument("--data-root", required=True)
    paper_c0_preflight.add_argument("--split", default="train")
    paper_c0_preflight.add_argument("--require-audio", action="store_true")
    paper_c0_preflight.add_argument("--resolved-config-out")
    paper_c0_preflight.add_argument("--out", required=True)

    target_smoke = sub.add_parser("paper-c0-target-smoke")
    target_smoke.add_argument("--config", required=True)
    target_smoke.add_argument("--out", required=True)
    target_smoke.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    target_smoke.add_argument("--source-hashes")
    target_smoke.add_argument("--fixture-manifest")
    target_smoke.add_argument("--fixture-root")
    target_smoke.add_argument("--real-manifest")
    target_smoke.add_argument("--real-root")
    target_smoke.add_argument("--run-pytest", action="store_true")

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    smoke.add_argument("--updates", type=int, default=2)

    round2_toy = sub.add_parser("round2-toy")
    round2_toy.add_argument("--config", required=True)
    round2_toy.add_argument("--out", required=True)
    round2_toy.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    round2_toy.add_argument("--updates", type=int, default=None)

    round2_coverage = sub.add_parser("round2-coverage")
    round2_coverage.add_argument("--config", required=True)
    round2_coverage.add_argument("--out", required=True)
    round2_coverage.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    round2_coverage.add_argument("--updates", type=int, default=None)

    round2_s144_resume = sub.add_parser("round2-s144-resume")
    round2_s144_resume.add_argument("--parent-checkpoint", required=True)
    round2_s144_resume.add_argument("--out", required=True)
    round2_s144_resume.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    round2_s144_resume.add_argument("--target-updates", type=int, default=18000)
    round2_s144_resume.add_argument("--milestones", type=int, nargs="+", default=[6000, 12000, 18000])
    round2_s144_resume.add_argument("--max-additional-updates", type=int, default=16000)

    scene_diagnostic = sub.add_parser("round2-scene-diagnostic")
    scene_diagnostic.add_argument("--config", required=True)
    scene_diagnostic.add_argument("--run-dir", required=True)
    scene_diagnostic.add_argument("--out", required=True)
    scene_diagnostic.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    scene_diagnostic.add_argument("--checkpoints", type=int, nargs="+", default=[6000, 12000, 18000])

    projection = sub.add_parser("round2-projection-eval")
    projection.add_argument("--config", required=True)
    projection.add_argument("--run-dir", required=True)
    projection.add_argument("--out", required=True)
    projection.add_argument("--checkpoint-update", type=int, default=18000)
    projection.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    common_ablation = sub.add_parser("round2-common-loss-ablation")
    common_ablation.add_argument("--parent-checkpoint", required=True)
    common_ablation.add_argument("--out", required=True)
    common_ablation.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    common_ablation.add_argument("--updates", type=int, default=2000)
    common_ablation.add_argument("--milestones", type=int, nargs="+", default=[0, 500, 1000, 1500, 2000])

    bn_cf = sub.add_parser("round2-bn-counterfactual")
    bn_cf.add_argument("--config", required=True)
    bn_cf.add_argument("--parent-run", required=True)
    bn_cf.add_argument("--b0-run", required=True)
    bn_cf.add_argument("--b1-run", required=True)
    bn_cf.add_argument("--out", required=True)
    bn_cf.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    p2_score = sub.add_parser("p2-score-route")
    p2_score.add_argument("--config", required=True)
    p2_score.add_argument("--out", required=True)
    p2_score.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p2_score.add_argument("--updates", type=int, default=None)

    p2b = sub.add_parser("p2b-score-coverage")
    p2b.add_argument("--config", required=True)
    p2b.add_argument("--out", required=True)
    p2b.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p2b.add_argument("--updates", type=int, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            return cmd_preflight(args)
        if args.command == "make-fixtures":
            return cmd_make_fixtures(args)
        if args.command == "inspect-model":
            return cmd_inspect_model(args)
        if args.command == "paper-c0-preflight":
            return cmd_paper_c0_preflight(args)
        if args.command == "paper-c0-target-smoke":
            return cmd_paper_c0_target_smoke(args)
        if args.command == "smoke":
            return cmd_smoke(args)
        if args.command == "round2-toy":
            return cmd_round2_toy(args)
        if args.command == "round2-coverage":
            return cmd_round2_coverage(args)
        if args.command == "round2-s144-resume":
            return cmd_round2_s144_resume(args)
        if args.command == "round2-scene-diagnostic":
            return cmd_round2_scene_diagnostic(args)
        if args.command == "round2-projection-eval":
            return cmd_round2_projection_eval(args)
        if args.command == "round2-common-loss-ablation":
            return cmd_round2_common_loss_ablation(args)
        if args.command == "round2-bn-counterfactual":
            return cmd_round2_bn_counterfactual(args)
        if args.command == "p2-score-route":
            return cmd_p2_score_route(args)
        if args.command == "p2b-score-coverage":
            return cmd_p2b_score_coverage(args)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print(f"ERROR: Missing Python dependency: {missing}", file=sys.stderr)
        return 2
    except (ConfigError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


def cmd_preflight(args) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": _torch_info(),
        "nvidia_smi": _command_json(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"]),
        "git": _command_json(["git", "rev-parse", "--short", "HEAD"]),
    }
    out.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(env, indent=2, ensure_ascii=False))
    return 0


def cmd_make_fixtures(args) -> int:
    manifest = make_fixtures(args.out, seed=args.seed)
    print(f"Wrote fixture manifest: {manifest}")
    return 0


def cmd_inspect_model(args) -> int:
    import torch

    from .models.separator import Separator
    from .profiling import model_shape_report

    config = load_config(args.config)
    model = Separator(config)
    report = model_shape_report(model, config, device="cpu")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Architecture Report",
        "",
        f"- config: `{args.config}`",
        f"- torch: `{torch.__version__}`",
        f"- condition route: `{report['condition_route']}`",
        f"- input shape: `{report['input_shape']}`",
        f"- output shape: `{report['output_shape']}`",
        f"- total parameters: `{report['parameter_count']['total']}`",
        f"- trainable parameters: `{report['parameter_count']['trainable']}`",
        "",
        "## Parameters By Module",
        "",
        *[
            f"- `{name}`: `{count}`"
            for name, count in sorted(report["parameter_count"].get("by_module", {}).items())
        ],
        "",
        "This report is generated from a synthetic shape pass, not from real SQR data.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cmd_paper_c0_preflight(args) -> int:
    from .paper_c0.portable import run_portable_preflight

    result = run_portable_preflight(
        args.config,
        args.manifest,
        args.data_root,
        split=args.split,
        require_audio=args.require_audio,
        resolved_config_out=args.resolved_config_out,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_paper_c0_target_smoke(args) -> int:
    from .paper_c0.target_smoke import run_target_smoke

    result = run_target_smoke(
        args.config,
        args.out,
        device=args.device,
        source_hashes_path=args.source_hashes,
        fixture_manifest=args.fixture_manifest,
        fixture_root=args.fixture_root,
        real_manifest=args.real_manifest,
        real_root=args.real_root,
        run_pytest=args.run_pytest,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] == "TARGET_SMOKE_PASS" else 2
    return 0


def cmd_smoke(args) -> int:
    from .trainer import run_smoke

    config = load_config(args.config)
    result = run_smoke(config, device=args.device, updates=args.updates)
    print(json.dumps(result, indent=2))
    return 0


def cmd_round2_toy(args) -> int:
    from .trainer import run_bounded_toy_training

    config = load_config(args.config)
    result = run_bounded_toy_training(config, out_dir=args.out, device=args.device, attempted_updates=args.updates)
    print(json.dumps(result, indent=2))
    return 0


def cmd_round2_coverage(args) -> int:
    from .coverage import run_data_coverage_comparison

    config = load_config(args.config)
    result = run_data_coverage_comparison(config, out_dir=args.out, device=args.device, attempted_updates=args.updates)
    print(json.dumps(result, indent=2))
    return 0


def cmd_round2_s144_resume(args) -> int:
    from .resume import run_s144_resume_diagnostic

    result = run_s144_resume_diagnostic(
        parent_checkpoint=args.parent_checkpoint,
        out_dir=args.out,
        device=args.device,
        target_updates=args.target_updates,
        milestones=tuple(args.milestones),
        max_additional_updates=args.max_additional_updates,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_round2_scene_diagnostic(args) -> int:
    from .scene_diagnostic import run_scene_level_diagnostic

    result = run_scene_level_diagnostic(
        config_path=args.config,
        run_dir=args.run_dir,
        out_dir=args.out,
        device=args.device,
        checkpoints=tuple(args.checkpoints),
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_round2_projection_eval(args) -> int:
    from .projection_eval import run_projection_evaluator

    result = run_projection_evaluator(
        config_path=args.config,
        run_dir=args.run_dir,
        out_dir=args.out,
        checkpoint_update=args.checkpoint_update,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_round2_common_loss_ablation(args) -> int:
    from .common_loss_ablation import run_common_loss_ablation

    result = run_common_loss_ablation(
        parent_checkpoint=args.parent_checkpoint,
        out_dir=args.out,
        device=args.device,
        attempted_updates=args.updates,
        milestones=tuple(args.milestones),
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_round2_bn_counterfactual(args) -> int:
    from .bn_counterfactual import run_bn_counterfactual

    result = run_bn_counterfactual(
        config_path=args.config,
        source_runs={"parent18000": args.parent_run, "B0_20000": args.b0_run, "B1_20000": args.b1_run},
        out_dir=args.out,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_p2_score_route(args) -> int:
    from .score_route import run_score_route

    config = load_config(args.config)
    result = run_score_route(config, out_dir=args.out, device=args.device, attempted_updates=args.updates)
    print(json.dumps(result, indent=2))
    return 0


def cmd_p2b_score_coverage(args) -> int:
    from .score_coverage_runner import run_p2b

    config = load_config(args.config)
    result = run_p2b(config, out_dir=args.out, device=args.device, attempted_updates=args.updates)
    print(json.dumps(result, indent=2))
    return 0


def _torch_info() -> dict:
    try:
        import torch
    except ModuleNotFoundError as exc:
        return {"available": False, "error": str(exc)}
    info = {
        "available": True,
        "version": torch.__version__,
        "file": getattr(torch, "__file__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info["device"] = prop.name
        info["device_memory_gib"] = round(prop.total_memory / (1024**3), 3)
    return info


def _command_json(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


if __name__ == "__main__":
    raise SystemExit(main())

