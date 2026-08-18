#!/usr/bin/env python3
"""Measure the exact threshold sweep on a large deterministic synthetic vector."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from flight_delay.modeling.v1_selection import select_v1_threshold

MINIMUM_ROWS = 250_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=MINIMUM_ROWS)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.rows < MINIMUM_ROWS:
        raise SystemExit(f"--rows must be at least {MINIMUM_ROWS}")

    rng = np.random.default_rng(arguments.seed)
    scores = (np.arange(arguments.rows, dtype=float) + 0.5) / arguments.rows
    rng.shuffle(scores)
    labels = (rng.random(arguments.rows) < 0.20).astype(int)
    labels[0], labels[1] = 0, 1

    started = time.perf_counter()
    result = select_v1_threshold(
        labels,
        scores,
        recall_min=0.60,
        precision_min=0.30,
        predicted_positive_rate_max=0.50,
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "rows": arguments.rows,
                "seed": arguments.seed,
                "selected_threshold": result.selected_threshold,
                "threshold_rows": len(result.threshold_table),
                "unique_scores": int(np.unique(scores).size),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
