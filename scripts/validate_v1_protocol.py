"""Validate the immutable, pre-training v1 experiment protocol without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.modeling.v1_protocol import V1ProtocolError, load_and_validate_v1_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/v1_experiment_protocol.yaml")
    )
    parser.add_argument("--lock", type=Path, default=Path("experiments/v1/protocol_lock.json"))
    args = parser.parse_args()
    try:
        protocol, lock, protocol_sha256 = load_and_validate_v1_protocol(
            args.protocol,
            lock_path=args.lock,
            repository_root=Path.cwd(),
        )
    except V1ProtocolError as error:
        print(json.dumps({"status": "invalid", "detail": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "valid",
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "base_git_sha": lock["base_git_sha"],
                "candidate_count": protocol["catboost_search"]["candidate_count"],
                "fold_count": protocol["rolling_origin"]["fold_count"],
                "training_started": lock["training_started"],
                "production_registry_version": lock["incumbent_registry_version"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
