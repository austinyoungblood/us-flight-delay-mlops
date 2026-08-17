#!/usr/bin/env python3
"""Separate one-time December 2025 qualification for a frozen v3 November winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.modeling.v3.execution import preflight, run_december_apply


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="open December exactly once")
    parser.add_argument(
        "--tracking", choices=("disabled", "online"), default="disabled", help="tracking mode"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if arguments.apply:
        report = run_december_apply(root, tracking=arguments.tracking)
    else:
        if arguments.tracking != "disabled":
            raise SystemExit("dry-run does not permit online tracking")
        report = preflight(root, stage="qualification")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
