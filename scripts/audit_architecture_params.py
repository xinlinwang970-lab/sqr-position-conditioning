#!/usr/bin/env python3
"""Print a reproducible parameter-count summary for a model config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqr_repro.config import load_config
from sqr_repro.models.separator import Separator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a model YAML config")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    model = Separator(config)
    model_config = config["model"]
    report = {
        "config": str(config_path),
        "architecture_version": model.architecture_version,
        "condition_route": model.route,
        "stage_channels": list(model.stage_channels),
        "tfc_growth_rate": model.tfc_growth_rate,
        "tfc_layers_per_block": int(model_config["tfc_layers"]),
        "parameters": model.count_parameters(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
