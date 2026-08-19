"""Validate final-evidence coverage and, optionally, captured files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.deployment.evidence import (
    EvidenceValidationError,
    load_evidence_manifest,
    missing_required_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evidence/evidence_manifest.json"))
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()
    try:
        result = load_evidence_manifest(args.manifest, require_files=args.require_files)
    except EvidenceValidationError as error:
        print(json.dumps({"status": "invalid", "detail": str(error)}, sort_keys=True))
        return 2
    status_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    for capture in result["captures"]:
        status = capture["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        mode = capture["evidence_mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    missing = missing_required_evidence(result)
    print(
        json.dumps(
            {
                "status": "valid" if not missing else "incomplete",
                "evidence_modes": mode_counts,
                "evidence_statuses": status_counts,
                "required_evidence": {
                    "complete": not missing,
                    "missing": missing,
                    "present": len([item for item in result["captures"] if item["required"]])
                    - len(missing),
                    "total": len([item for item in result["captures"] if item["required"]]),
                },
            },
            sort_keys=True,
        )
    )
    if missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
