from __future__ import annotations


def count_parameters(model) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    by_module = {name: sum(param.numel() for param in module.parameters()) for name, module in model.named_children()}
    return {"total": total, "trainable": trainable, "by_module": by_module}


def model_shape_report(model, config: dict, device: str = "cpu") -> dict:
    import torch

    model = model.to(device)
    model.eval()
    batch = int(config["training"]["batch_size"])
    frames = int(config["audio"]["num_frames"])
    freq = int(config["audio"]["n_fft"]) // 2 + 1
    spec = torch.randn(batch, 4, frames, freq, device=device)
    timbre = torch.tensor([0, 1] * ((batch + 1) // 2), device=device)[:batch]
    position = torch.tensor([3, 1] * ((batch + 1) // 2), device=device)[:batch]
    score_config = config.get("score") or {}
    score_roll = None
    if score_config.get("enabled", False):
        score_roll = torch.randn(
            batch,
            int(score_config.get("num_pitches", 128)),
            frames,
            device=device,
        )
    with torch.inference_mode():
        out = model(spec, timbre.long(), position.long(), score_roll=score_roll)
    return {
        "input_shape": list(spec.shape),
        "output_shape": list(out.shape),
        "parameter_count": count_parameters(model),
        "condition_route": config["model"]["condition_route"],
    }

