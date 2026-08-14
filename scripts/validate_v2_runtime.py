#!/usr/bin/env python3
"""Instantiate every pinned v2 constructor without fitting or accessing data."""

from __future__ import annotations

import json
from pathlib import Path

from flight_delay.modeling.v2.models import require_versions, validate_constructor_contract
from flight_delay.modeling.v2.protocol import load_and_validate_v2_protocol


def main() -> int:
    root = Path.cwd()
    protocol, _lock, protocol_sha = load_and_validate_v2_protocol(
        root / "configs/v2_experiment_protocol.yaml",
        lock_path=root / "experiments/v2/protocol_lock.json",
        repository_root=root,
    )
    versions = require_versions()
    constructors = validate_constructor_contract(protocol)
    print(
        json.dumps(
            {
                "status": "valid",
                "protocol_sha256": protocol_sha,
                "versions": versions,
                "constructor_count": len(constructors),
                "model_fit_performed": False,
                "data_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
