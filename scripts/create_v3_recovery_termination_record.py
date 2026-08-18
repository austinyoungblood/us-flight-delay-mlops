#!/usr/bin/env python3
"""Plan or freeze an operator handoff record after the original v3 process has terminated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_delay.modeling.v3.recovery import create_termination_record, recovery_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-log", required=True, type=Path)
    parser.add_argument("--original-pid", type=int)
    parser.add_argument("--wrapper-exit-status", required=True, type=int)
    parser.add_argument("--termination-mechanism", required=True)
    parser.add_argument("--termination-reason", required=True)
    parser.add_argument("--confirm-original-terminated", action="store_true")
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if not arguments.apply:
        report = {
            "mode": "dry-run/preflight",
            "process_signaled_or_inspected": False,
            "source_files_opened": False,
            "output": str(
                recovery_directory(root, arguments.recovery_id) / "termination_record.json"
            ),
        }
    else:
        report = create_termination_record(
            root,
            recovery_id=arguments.recovery_id,
            source_root=arguments.source_root,
            source_log=arguments.source_log,
            original_pid=arguments.original_pid,
            wrapper_exit_status=arguments.wrapper_exit_status,
            termination_mechanism=arguments.termination_mechanism,
            termination_reason=arguments.termination_reason,
            original_execution_terminated=arguments.confirm_original_terminated,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
