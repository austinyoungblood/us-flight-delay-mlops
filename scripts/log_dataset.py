"""Publish the prepared BTS splits as a versioned W&B dataset artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from flight_delay.modeling.tracking import TrackingError, publish_dataset_artifact


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TrackingError(f"cannot read configuration {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TrackingError("configuration root must be a mapping")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log the prepared BTS dataset to W&B.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args.config)
        data = config["data"]
        tracking = config["tracking"]
        result = publish_dataset_artifact(
            source_manifest_path=Path(data["source_manifest"]),
            processed_manifest_path=Path(data["processed_manifest"]),
            processed_directory=Path(data["processed_directory"]),
            entity=os.getenv("WANDB_ENTITY", ""),
            project=os.getenv("WANDB_PROJECT", str(tracking["project"])),
            mode=args.wandb_mode or os.getenv("WANDB_MODE", str(tracking["mode"])),
            artifact_name=str(tracking["dataset_artifact_name"]),
        )
    except (KeyError, TypeError, TrackingError) as error:
        print(f"dataset publication failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
