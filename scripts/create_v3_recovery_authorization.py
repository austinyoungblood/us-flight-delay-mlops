#!/usr/bin/env python3
"""Create the immutable authorization required for an applied governed v3 recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.modeling.v3.protocol import load_and_validate_v3_protocol
from flight_delay.modeling.v3.recovery import create_authorization


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument("--selector-test-command", required=True)
    parser.add_argument("--selector-test-result", required=True)
    parser.add_argument("--benchmark-command", required=True)
    parser.add_argument("--benchmark-result", required=True)
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.apply:
        report = {
            "mode": "dry-run/preflight",
            "authorization_written": False,
            "requires_termination_record": True,
            "requires_source_evidence": True,
        }
    else:
        root = Path(__file__).resolve().parents[1]
        protocol, _lock, _sha = load_and_validate_v3_protocol(
            root / "configs/v3_experiment_protocol.yaml",
            lock_path=root / "experiments/v3/protocol_lock.json",
            repository_root=root,
        )
        report = create_authorization(
            root,
            protocol=protocol,
            recovery_id=arguments.recovery_id,
            corrected_selector_test_evidence={
                "command": arguments.selector_test_command,
                "result": arguments.selector_test_result,
            },
            corrected_selector_benchmark_evidence={
                "command": arguments.benchmark_command,
                "result": arguments.benchmark_result,
            },
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
