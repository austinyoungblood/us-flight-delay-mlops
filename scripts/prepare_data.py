"""Reserved entry point for dataset preparation orchestration."""

import sys


def main() -> int:
    """Explain the current scope without reading or writing a dataset."""

    print(
        "Dataset orchestration is intentionally deferred. Brief 01 provides only tested, "
        "in-memory preprocessing primitives.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
