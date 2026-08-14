#!/usr/bin/env python3
"""Validate the immutable pre-training v2 protocol without model imports or network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.modeling.v2.protocol import V2ProtocolError, load_and_validate_v2_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/v2_experiment_protocol.yaml")
    )
    parser.add_argument("--lock", type=Path, default=Path("experiments/v2/protocol_lock.json"))
    arguments = parser.parse_args(argv)
    try:
        protocol, lock, protocol_sha = load_and_validate_v2_protocol(
            arguments.protocol,
            lock_path=arguments.lock,
            repository_root=Path.cwd(),
        )
    except V2ProtocolError as error:
        print(json.dumps({"status": "invalid", "detail": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "valid",
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha,
                "base_git_sha": lock["base_git_sha"],
                "lightgbm_candidate_count": protocol["lightgbm_search"]["candidate_count"],
                "catboost_candidate_count": protocol["catboost_search"]["candidate_count"],
                "feature_count": protocol["feature_contract"]["total_feature_count"],
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
