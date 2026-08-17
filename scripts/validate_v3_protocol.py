#!/usr/bin/env python3
"""Validate the immutable pre-training v3 protocol without model imports or network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.modeling.v3.protocol import (
    CANDIDATE_IDENTITY_IDS,
    V3ProtocolError,
    load_and_validate_v3_protocol,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/v3_experiment_protocol.yaml")
    )
    parser.add_argument("--lock", type=Path, default=Path("experiments/v3/protocol_lock.json"))
    arguments = parser.parse_args(argv)
    try:
        protocol, lock, protocol_sha = load_and_validate_v3_protocol(
            arguments.protocol,
            lock_path=arguments.lock,
            repository_root=Path.cwd(),
        )
    except V3ProtocolError as error:
        print(json.dumps({"status": "invalid", "detail": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "valid",
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha,
                "base_git_sha": lock["base_git_sha"],
                "candidate_identities": list(CANDIDATE_IDENTITY_IDS),
                "candidate_identity_count": protocol["candidate_identities"]["total"],
                "feature_count": protocol["feature_contract"]["total_feature_count"],
                "native_categorical_count": protocol["feature_contract"][
                    "native_categorical_count"
                ],
                "finalist_count": protocol["finalists"]["total"],
                "training_started": lock["training_started"],
                "december_opened": lock["december_opened"],
                "production_registry_version": lock["production_registry_version"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
