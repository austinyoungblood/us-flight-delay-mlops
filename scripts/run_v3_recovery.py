#!/usr/bin/env python3
"""Preflight by default; explicitly apply the governed v3 recovery after authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.modeling.v3.recovery import recovery_preflight, run_recovery_apply


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--tracking", choices=("disabled", "online"), default="disabled")
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if arguments.apply:
        report = run_recovery_apply(
            root, recovery_id=arguments.recovery_id, tracking=arguments.tracking
        )
    else:
        if arguments.tracking != "disabled":
            raise SystemExit("recovery dry-run does not permit online tracking")
        report = recovery_preflight(root, recovery_id=arguments.recovery_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
