#!/usr/bin/env python3
"""Plan by default; explicitly export read-only source W&B evidence for v3 recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from flight_delay.modeling.v3.protocol import load_and_validate_v3_protocol
from flight_delay.modeling.v3.recovery import (
    SOURCE_GROUP,
    freeze_source_evidence,
    recovery_directory,
    wandb_source_runs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument(
        "--apply", action="store_true", help="contact W&B read-only and freeze JSON"
    )
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    directory = recovery_directory(root, arguments.recovery_id)
    if not arguments.apply:
        report = {
            "mode": "dry-run/preflight",
            "source_group": SOURCE_GROUP,
            "network_contacted": False,
            "original_runs_mutated": False,
            "output": str(directory / "source_evidence.json"),
        }
    else:
        protocol, _lock, _sha = load_and_validate_v3_protocol(
            root / "configs/v3_experiment_protocol.yaml",
            lock_path=root / "experiments/v3/protocol_lock.json",
            repository_root=root,
        )
        runs = wandb_source_runs(
            entity=os.environ.get("WANDB_ENTITY", "").strip(),
            project=os.environ.get("WANDB_PROJECT", "").strip(),
        )
        payload, digest = freeze_source_evidence(
            root, protocol=protocol, recovery_id=arguments.recovery_id, tracking_runs=runs
        )
        report = {
            "mode": "read-only-source-export",
            "run_count": len(payload["source_tracking_runs"]),
            "source_evidence_sha256": digest,
            "original_runs_mutated": False,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
