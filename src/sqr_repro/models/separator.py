from __future__ import annotations

import torch
import torch.nn as nn

from ..config import input_frequency_bins
from .blocks import ControlledTfcTdfBlock, Downsample, Upsample
from .conditioning import ConditionEmbeddings, ConditionParameterGenerator
from .score_encoder import ScoreEncoder


class Separator(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        audio = config["audio"]
        model = config["model"]
        self.config = config
        self.channels = int(model["base_channels"])
        self.tfc_growth_rate = int(model["tfc_growth_rate"])
        if self.channels != self.tfc_growth_rate:
            raise ValueError(
                "base_channels and tfc_growth_rate are split for audit clarity, but must remain equal until "
                "a LaSAFT-compatible projection rule is confirmed"
            )
        self.route = str(model["condition_route"])
        self.num_timbres = int(model.get("num_timbres", 3))
        self.num_positions = int(model.get("num_positions", 4))
        score_config = config.get("score") or {}
        self.score_enabled = bool(score_config.get("enabled", False))
        self.input_freq = input_frequency_bins(config)
        self.stem_freq = self.input_freq - 1
        if self.stem_freq % 8 != 0:
            raise ValueError("F-1 must be divisible by 8")

        c = self.channels
        kernel = tuple(int(v) for v in model["kernel_size"])
        tfc_layers = int(model["tfc_layers"])
        tdf_factor = int(model["tdf_bottleneck_factor"])
        tdf_min = int(model["tdf_min_units"])
        tdf_bias = bool(model["tdf_bias"])

        self.stem = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=(1, 2), padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        enc_freqs = [self.stem_freq, self.stem_freq // 2, self.stem_freq // 4]
        dec_freqs = [self.stem_freq // 4, self.stem_freq // 2, self.stem_freq]
        self.encoders = nn.ModuleList(
            [
                ControlledTfcTdfBlock(c, c, freq, tfc_layers, kernel, tdf_factor, tdf_min, tdf_bias)
                for freq in enc_freqs
            ]
        )
        self.downs = nn.ModuleList([Downsample(c) for _ in range(3)])
        self.middle = ControlledTfcTdfBlock(
            c, c, self.stem_freq // 8, tfc_layers, kernel, tdf_factor, tdf_min, tdf_bias
        )
        self.ups = nn.ModuleList([Upsample(c) for _ in range(3)])
        self.decoders = nn.ModuleList(
            [
                ControlledTfcTdfBlock(2 * c, c, freq, tfc_layers, kernel, tdf_factor, tdf_min, tdf_bias)
                for freq in dec_freqs
            ]
        )
        self.head = nn.Conv2d(c, 4, kernel_size=(1, 2), padding=(0, 1))

        self.score_encoder = None
        if self.score_enabled:
            if score_config.get("injection", "pre_middle_add") != "pre_middle_add":
                raise ValueError("Only score injection=pre_middle_add is implemented")
            self.score_encoder = ScoreEncoder(
                int(score_config.get("num_pitches", 128)),
                int(score_config.get("hidden_channels", 32)),
                c,
            )

        self.embeddings = ConditionEmbeddings(int(model["embedding_dim"]), self.num_timbres, self.num_positions)
        hidden_dims = [int(v) for v in model["control_hidden_dims"]]
        dropout = float(model["control_dropout"])
        hidden_norm = str(model["control_hidden_norm"])
        output_norm = str(model["control_output_norm"])

        self.encoder_controller = None
        self.middle_controller = None
        self.decoder_controller = None
        if self.route in {"split_stage", "split_stage_paper"}:
            self.encoder_controller = ConditionParameterGenerator(
                int(model["embedding_dim"]), c, 3, hidden_dims, dropout, hidden_norm, output_norm
            )
            self.middle_controller = ConditionParameterGenerator(
                int(model["embedding_dim"]), c, 1, hidden_dims, dropout, hidden_norm, output_norm
            )
            self.decoder_controller = ConditionParameterGenerator(
                int(model["embedding_dim"]), c, 3, hidden_dims, dropout, hidden_norm, output_norm
            )
        elif self.route == "mean_decoder":
            self.decoder_controller = ConditionParameterGenerator(
                int(model["embedding_dim"]), c, 3, hidden_dims, dropout, hidden_norm, output_norm
            )
        elif self.route == "mid_dual_paper":
            # Paper Fig.2 + §3.2 original: single mixed embedding, decoder-only injection.
            self.decoder_controller = ConditionParameterGenerator(
                int(model["embedding_dim"]), c, 3, hidden_dims, dropout, hidden_norm, output_norm
            )
        else:
            raise ValueError(f"Unknown condition route: {self.route}")

    def forward(self, spec, timbre_id, position_id, score_roll=None, return_debug: bool = False):
        self._validate_input(spec, timbre_id, position_id, score_roll)
        enc_params, middle_params, dec_params = self._condition_params(timbre_id, position_id)
        debug = self._init_debug(spec, enc_params, middle_params, dec_params) if return_debug else None
        h = self.stem(spec)
        if debug is not None:
            debug["stem_shape"] = list(h.shape)
        skips = []
        for index, block in enumerate(self.encoders):
            h = block(h, enc_params[index])
            skips.append(h)
            if debug is not None:
                debug["encoder_shapes"].append({"block": list(h.shape), "skip": list(skips[-1].shape)})
            h = self.downs[index](h)
            if debug is not None:
                debug["encoder_shapes"][-1]["down"] = list(h.shape)
        score_debug = {"score_enabled": self.score_enabled, "score_input_shape": None, "score_encoded_shape": None, "score_projected_shape": None, "score_encoded_feature": None, "score_projected_feature": None, "middle_shape_before_score": list(h.shape), "middle_shape_after_score": list(h.shape), "score_projection_zero_initialized": None}
        if self.score_enabled:
            score_debug["score_input_shape"] = list(score_roll.shape)
            score_encoded = self.score_encoder.encode(score_roll)
            score_feature = self.score_encoder.project(score_encoded)
            if debug is not None:
                debug["score_encoded_feature"] = score_encoded.detach().clone()
                debug["score_projected_feature"] = score_feature.detach().clone()
            score_debug["score_encoded_shape"] = list(score_encoded.shape)
            expected = (h.shape[0], h.shape[1], h.shape[2])
            if tuple(score_feature.shape) != expected:
                raise ValueError(f"Score feature shape {tuple(score_feature.shape)} does not match middle [B,C,T] {expected}")
            score_debug["score_projected_shape"] = list(score_feature.shape)
            score_feature = score_feature.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1])
            h = h + score_feature
            score_debug["middle_shape_after_score"] = list(h.shape)
            score_debug["score_projection_zero_initialized"] = self.score_encoder.projection_is_zero()
        if debug is not None:
            debug.update(score_debug)
        h = self.middle(h, middle_params[0] if middle_params else None)
        if debug is not None:
            debug["middle_shape"] = list(h.shape)
        for index, block in enumerate(self.decoders):
            h = self.ups[index](h)
            skip = skips[-(index + 1)]
            if h.shape[-2:] != skip.shape[-2:]:
                raise ValueError(f"Skip shape mismatch: up {h.shape}, skip {skip.shape}")
            h = torch.cat([h, skip], dim=1)
            h = block(h, dec_params[index])
            if debug is not None:
                debug["decoder_shapes"].append({"up": list(skip.shape), "block": list(h.shape)})
        out = self.head(h)
        self._validate_output(spec, out)
        if return_debug:
            debug["output_shape"] = list(out.shape)
            return out, debug
        return out

    def count_parameters(self) -> dict[str, int]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        by_module = {name: sum(param.numel() for param in module.parameters()) for name, module in self.named_children()}
        return {"total": total, "trainable": trainable, "by_module": by_module}

    def _condition_params(self, timbre_id, position_id):
        e_t, e_m, e_p = self.embeddings(timbre_id, position_id)
        if self.route == "split_stage":
            return (
                self.encoder_controller(e_m),
                self.middle_controller(e_m),
                self.decoder_controller(e_m),
            )
        if self.route == "split_stage_paper":
            # Paper §4.2 best config: encoder<-e_t, middle<-0.5*(e_t+e_p), decoder<-e_p
            return (
                self.encoder_controller(e_t),
                self.middle_controller(e_m),
                self.decoder_controller(e_p),
            )
        if self.route == "mid_dual_paper":
            # Paper Fig.2 + §3.2: single mixed embedding e_m = 0.5*(e_t+e_p) -> decoder ONLY.
            # Encoder and middle run UNCONDITIONED (no GPoCM gates applied).
            return (
                [None, None, None],
                [None],
                self.decoder_controller(e_m),
            )
        return ([None, None, None], [None], self.decoder_controller(e_m))

    def _validate_input(self, spec, timbre_id, position_id, score_roll=None) -> None:
        if spec.ndim != 4 or spec.shape[1] != 4:
            raise ValueError("spec must have shape [B, 4, T, F]")
        if not torch.is_floating_point(spec):
            raise ValueError("spec must be a floating point tensor")
        if not torch.isfinite(spec).all():
            raise ValueError("spec contains NaN or Inf")
        if spec.shape[-1] != self.input_freq:
            raise ValueError(f"Expected F={self.input_freq}, got {spec.shape[-1]}")
        if spec.shape[-2] % 8 != 0:
            raise ValueError("T must be divisible by 8")
        if timbre_id.shape != position_id.shape or timbre_id.shape[0] != spec.shape[0]:
            raise ValueError("timbre_id and position_id must have shape [B]")
        if timbre_id.dtype != torch.long or position_id.dtype != torch.long:
            raise ValueError("timbre_id and position_id must be torch.long")
        if timbre_id.device != spec.device or position_id.device != spec.device:
            raise ValueError("spec, timbre_id, and position_id must be on the same device")
        if timbre_id.numel() and (int(timbre_id.min()) < 0 or int(timbre_id.max()) >= self.num_timbres):
            raise ValueError(f"timbre_id values must be in [0, {self.num_timbres - 1}]")
        if position_id.numel() and (int(position_id.min()) < 0 or int(position_id.max()) >= self.num_positions):
            raise ValueError(f"position_id values must be in [0, {self.num_positions - 1}]")
        if not self.score_enabled and score_roll is not None:
            raise ValueError("score_roll must be None when score.enabled=false")
        if self.score_enabled:
            if score_roll is None:
                raise ValueError("score_roll is required when score.enabled=true")
            if score_roll.ndim != 3 or score_roll.shape[0] != spec.shape[0] or score_roll.shape[1] != int((self.config.get("score") or {}).get("num_pitches", 128)) or score_roll.shape[2] != spec.shape[2]:
                raise ValueError("score_roll must have shape [B, num_pitches, T] matching spec")
            if score_roll.device != spec.device:
                raise ValueError("score_roll and spec must be on the same device")

    def _validate_output(self, spec, out) -> None:
        if out.shape != spec.shape:
            raise ValueError(f"Separator output shape {tuple(out.shape)} does not match input {tuple(spec.shape)}")
        if not torch.isfinite(out).all():
            raise ValueError("Separator output contains NaN or Inf")

    def _init_debug(self, spec, enc_params, middle_params, dec_params) -> dict:
        def shapes(params):
            out = []
            for item in params:
                if item is None:
                    out.append(None)
                    continue
                weight, bias = item
                out.append({"weight": list(weight.shape), "bias": list(bias.shape)})
            return out

        return {
            "input_shape": list(spec.shape),
            "stem_shape": None,
            "encoder_shapes": [],
            "middle_shape": None,
            "decoder_shapes": [],
            "condition_route": self.route,
            "score_enabled": self.score_enabled,
            "score_input_shape": None,
            "score_encoded_shape": None,
            "score_projected_shape": None,
            "score_encoded_feature": None,
            "score_projected_feature": None,
            "middle_shape_before_score": None,
            "middle_shape_after_score": None,
            "score_projection_zero_initialized": None,
            "condition_parameter_shapes": {
                "encoder": shapes(enc_params),
                "middle": shapes(middle_params),
                "decoder": shapes(dec_params),
            },
        }

