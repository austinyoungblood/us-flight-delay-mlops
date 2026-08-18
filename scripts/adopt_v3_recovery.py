#!/usr/bin/env python3
"""Preflight or explicitly adopt completed recovery evidence into canonical v3 paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.modeling.v3.recovery import adopt_recovery, adoption_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    report = (
        adopt_recovery(root, recovery_id=arguments.recovery_id)
        if arguments.apply
        else adoption_preflight(root, recovery_id=arguments.recovery_id)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
