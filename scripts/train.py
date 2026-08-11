"""Run a W&B-lineaged Dummy or Candidate A validation experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from flight_delay.modeling.tracking import TrackingError
from flight_delay.modeling.training import run_training_experiment, training_result_dict


def _mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TrackingError(f"cannot read configuration {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TrackingError(f"configuration {path} must contain a mapping")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a validation-only Brief 02 experiment.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        base = _mapping(args.config)
        tracking = base["tracking"]
        result = run_training_experiment(
            experiment=_mapping(args.experiment),
            entity=os.getenv("WANDB_ENTITY", ""),
            project=os.getenv("WANDB_PROJECT", str(tracking["project"])),
            mode=args.wandb_mode or os.getenv("WANDB_MODE", str(tracking["mode"])),
        )
    except (KeyError, TypeError, ValueError, TrackingError) as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(training_result_dict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
