#!/usr/bin/env python3
"""Preflight by default; explicitly apply governed v2 development only after merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.modeling.v2.execution import preflight, run_development_apply


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="run governed development once")
    parser.add_argument(
        "--tracking", choices=("disabled", "online"), default="disabled", help="tracking mode"
    )
    parser.add_argument("--output", type=Path, help="optional local dry-run report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if arguments.apply:
        if arguments.output is not None:
            raise SystemExit("--output is dry-run-only")
        report = run_development_apply(root, tracking=arguments.tracking)
    else:
        if arguments.tracking != "disabled":
            raise SystemExit("dry-run does not permit online tracking")
        report = preflight(root, stage="development")
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
