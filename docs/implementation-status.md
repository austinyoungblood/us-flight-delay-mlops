# Implementation status

Status reflects completed Brief 01-05 work, including the governed `staging` release; it is not
final-project completion.

## Completed in Brief 01

- Python 3.11 package metadata and src-layout monorepo scaffold
- Typed prediction, response, feedback, route reliability, and health contracts
- Central immutable model-feature allowlist, forbidden set, and dedicated leakage exception
- Pure BTS column normalization, CRS time parsing, eligibility filtering, and target construction
- Scheduled time/route features, chronological partitions, and deterministic monthly sampling
- Carrier-route and all-carrier historical reliability aggregation with minimum-support flag
- Hermetic unit/integration tests, Ruff/pytest/coverage configuration, and pull-request CI
- Health-only FastAPI skeleton and two non-model-loading Streamlit placeholders
- Three Dockerfiles, local Compose wiring, architecture documentation, and data policy

## Completed in Brief 02

- Git checkpoint `f105bd1b08921b89df56afde89e881d290057730` on `main` and isolated
  `feat/data-artifact-baselines` branch
- Sequential, retrying, atomic, idempotent official BTS downloader with ZIP/member/checksum checks
- Seventeen Reporting Carrier archives (501,916,656 bytes) and canonical source manifest
- Leakage-safe deterministic Parquet preparation with exact chronological boundaries and canonical
  processed manifest (1,275,000 sampled rows)
- W&B 0.28.1 dataset artifact
  [`flight-delay-bts-sampled:v0`](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/dataset/flight-delay-bts-sampled/v0),
  digest `2ecdb5a6a60b23ed1ee1d603fb976516`
- Online [Dummy](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/g0cnsglm)
  and [Candidate A](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/7mt7qz71)
  validation-only runs consuming the exact dataset artifact
- Complete seven-file model artifacts `flight-delay-model:v0` (Dummy) and `v1` (Candidate A), with
  aggregate training baselines, validation metrics/plots, load time, and bounded latency
- Read-only W&B audit proving both runs finished, used dataset `v0`, and contain no test/final-test
  metric keys; metadata records no test evaluation and no Registry promotion
- Hermetic fake-W&B/model/data tests and branch-aware coverage enforcement at or above 80%

## Completed in Brief 05

- Accepted and merged the controlled Brief 04 stop, then committed a revised course-aligned release
  policy before final-test access
- Reconstructed only R3 sigmoid with the locked threshold and reproduced Brief 04 metrics within a
  documented `1e-9` floating-point tolerance
- Built and hashed the exact eight-file bundle plus a display-only route asset covering 6,879,484
  eligible completed 2025 flights
- Logged source artifact `flight-delay-r3-sigmoid-release:v0` and linked the identical bytes to the
  restricted Registry collection `wandb-registry-Model/us-flight-arrival-delay-15m:v0:staging`
- Clean-downloaded `staging` and verified every locked hash before opening the final test
- Evaluated January-May 2026 exactly once in W&B run
  [w4te9tla](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/w4te9tla)
- Retained `staging` because Brier skill, prior log-loss, probability-gap, and ECE gates failed;
  emitted a durable one-use marker and `release/release_decision.json` with `serving_alias=staging`

## Partially complete

- Brief 04: independent calibration metrics, six fixed rolling-origin bases, and six calibrated
  November finalists are complete. Every finalist failed AP and F1, so December qualification and
  all downstream release work correctly did not start.
- Brief 03: exact time partitions, calibrated Candidate A, bounded Candidate B tuning, validation
  evaluation, thresholds, model artifacts, and W&B audit are complete. No candidate passed every
  mandatory validation gate, so selection/freeze, Registry, and final-test tasks did not start.
- Data pipeline and one-time final-test evaluation are complete; the final test is consumed.
- FastAPI: only `/health` exists; model, cache, persistence, prediction, retrieval, and feedback do not.
- UIs: processes can render an honest placeholder; interactive workflows are not implemented.
- CI: workflow definition exists; repository-hosted pull-request evidence and branch protection remain.

## Not started

- DynamoDB table, serialization, seed events, persistence adapters, and monitoring queries
- Prediction/reliability/model-info/retrieval/feedback endpoints and TTL inference cache
- Production user interface, monitoring metrics/drift dashboard, AWS deployment, and teardown runbook
- Public GitHub/W&B setup, EC2 deployment, live validation, screenshots, and rubric evidence
