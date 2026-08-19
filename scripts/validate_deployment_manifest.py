"""Validate the frozen deployment manifest against committed release evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.deployment import DeploymentManifestError, load_and_validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("deploy/deployment_manifest.json"))
    parser.add_argument(
        "--release-decision", type=Path, default=Path("release/release_decision.json")
    )
    parser.add_argument("--selection-lock", type=Path, default=Path("release/selection_lock.json"))
    args = parser.parse_args()
    try:
        manifest = load_and_validate_manifest(
            args.manifest,
            release_decision_path=args.release_decision,
            selection_lock_path=args.selection_lock,
            repository_root=Path.cwd(),
        )
    except DeploymentManifestError as error:
        print(json.dumps({"status": "invalid", "detail": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "valid",
                "deployment_git_sha": manifest["deployment_git_sha"],
                "model": {
                    "alias": manifest["model"]["serving_alias"],
                    "version": manifest["model"]["registry_version"],
                    "digest": manifest["model"]["registry_digest"],
                },
                "images": {name: value["reference"] for name, value in manifest["images"].items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
