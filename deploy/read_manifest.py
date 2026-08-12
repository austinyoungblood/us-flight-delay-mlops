#!/usr/bin/env python3
"""Validate the deployment manifest and expose non-secret fields to shell scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.deployment import load_and_validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("image", "port", "env-names"))
    parser.add_argument("component", choices=("api", "traveler", "monitor"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "deploy/deployment_manifest.json")
    args = parser.parse_args()
    manifest = load_and_validate_manifest(
        args.manifest,
        release_decision_path=ROOT / "release/release_decision.json",
        selection_lock_path=ROOT / "release/selection_lock.json",
        repository_root=ROOT,
    )
    if args.command == "image":
        print(manifest["images"][args.component]["reference"])
    elif args.command == "port":
        print(manifest["runtime"]["ports"][args.component])
    else:
        print(json.dumps(sorted(manifest["environment_variable_names"][args.component])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
