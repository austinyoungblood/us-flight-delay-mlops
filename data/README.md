# Data policy

No raw or processed BTS data is committed to this repository. Future download tooling will use
official monthly Reporting Carrier On-Time Performance archives and record SHA-256 checksums in a
dataset manifest. Local `raw/`, `interim/`, and `processed/` directories are ignored by Git.

Brief 01 uses small in-memory DataFrames in tests and performs no downloads.
