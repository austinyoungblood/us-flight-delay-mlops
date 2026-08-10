# Implementation status

Status reflects the repository scaffold only and must not be read as final-project completion.

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

## Partially complete

- Data pipeline: tested primitives exist; download, manifests, file I/O, and orchestration do not.
- FastAPI: only `/health` exists; model, cache, persistence, prediction, retrieval, and feedback do not.
- UIs: processes can render an honest placeholder; interactive workflows are not implemented.
- CI: workflow definition exists; repository-hosted pull-request evidence and branch protection remain.

## Not started

- Official BTS downloader, SHA-256 dataset manifest, processed splits, and W&B dataset artifact
- Dummy baseline or Candidate A/B training, evaluation, threshold selection, and model bundle
- W&B experiment tracking, Registry collection, staging/production aliases, and public evidence
- DynamoDB table, serialization, seed events, persistence adapters, and monitoring queries
- Prediction/reliability/model-info/retrieval/feedback endpoints and TTL inference cache
- Production user interface, monitoring metrics/drift dashboard, AWS deployment, and teardown runbook
- Public GitHub/W&B setup, EC2 deployment, live validation, screenshots, and rubric evidence
