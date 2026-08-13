#!/usr/bin/env python3
"""Expose validated non-secret manifest fields to minimal deployment hosts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATTERN = re.compile(r"^ghcr\.io/[a-z0-9_.-]+/[a-z0-9_.-]+@sha256:[0-9a-f]{64}$")


def load_host_manifest(path: Path) -> dict:
    """Validate fields consumed by host shell scripts using only the Python standard library."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"manifest read failed: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise SystemExit("manifest read failed: unsupported schema")
    deployment_sha = manifest.get("deployment_git_sha")
    if not isinstance(deployment_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", deployment_sha):
        raise SystemExit("manifest read failed: invalid deployment SHA")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("image", "port", "env-names"))
    parser.add_argument("component", choices=("api", "traveler", "monitor"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "deploy/deployment_manifest.json")
    args = parser.parse_args()
    manifest = load_host_manifest(args.manifest)
    try:
        image = manifest["images"][args.component]
        port = manifest["runtime"]["ports"][args.component]
        names = manifest["environment_variable_names"][args.component]
    except (KeyError, TypeError) as error:
        raise SystemExit("manifest read failed: component fields are incomplete") from error
    reference = image.get("reference") if isinstance(image, dict) else None
    if not isinstance(reference, str) or not IMAGE_PATTERN.fullmatch(reference):
        raise SystemExit("manifest read failed: image is not digest-pinned")
    if image.get("source_git_sha") != manifest["deployment_git_sha"]:
        raise SystemExit("manifest read failed: image source SHA mismatch")
    if not isinstance(port, int) or port <= 0:
        raise SystemExit("manifest read failed: component port is invalid")
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
            for name in names
        )
        or len(names) != len(set(names))
    ):
        raise SystemExit("manifest read failed: environment names are invalid")
    if args.command == "image":
        print(reference)
    elif args.command == "port":
        print(port)
    else:
        print(json.dumps(sorted(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
