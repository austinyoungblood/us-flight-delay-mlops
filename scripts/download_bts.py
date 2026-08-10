"""Reserved entry point for the reviewed BTS download increment."""

import sys


def main() -> int:
    """Explain the current scope without making a network request."""

    print(
        "BTS downloading is intentionally not implemented in Brief 01. "
        "The next reviewed increment will add official URLs and SHA-256 manifests.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
