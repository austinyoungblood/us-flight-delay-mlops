# Implementation status

This record summarizes the repository's completed implementation stages before live AWS deployment.
It does not itself constitute cloud-deployment evidence.

## Repository foundation

- Python 3.11 package metadata and src-layout monorepo scaffold
- Typed prediction, response, feedback, route reliability, and health contracts
- Central immutable model-feature allowlist, forbidden set, and dedicated leakage exception
- Pure BTS column normalization, CRS time parsing, eligibility filtering, and target construction
- Scheduled time/route features, chronological partitions, and deterministic monthly sampling
- Carrier-route and all-carrier historical reliability aggregation with minimum-support flag
- Hermetic unit/integration tests, Ruff/pytest/coverage configuration, and pull-request CI
- Health-only FastAPI skeleton and two non-model-loading Streamlit placeholders
- Three Dockerfiles, local Compose wiring, architecture documentation, and data policy

## Data and baseline modeling

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

## Governed release and final test

- Accepted the controlled remediation stop, then committed a revised course-aligned release
  policy before final-test access
- Reconstructed only R3 sigmoid with the locked threshold and reproduced the remediation metrics within a
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

## API and DynamoDB persistence

- Revalidated the governed release locally and against live W&B, reproduced all three accepted image digests,
  merged it into `main` at `3bc9208c06f125f7d741699f7f89b441d1295dc3`, and created clean
  `feat/api-dynamodb` from that SHA
- Implemented the fail-closed Registry runtime: release-decision parsing, alias/version/digest/source
  anti-drift checks, clean external cache, selection-lock and all-file hash verification, leakage
  validation, asset loading, deterministic canary and exact serving metadata
- Real loader verified `staging` `v0`, digest `865ddd18f6debd44f24a79fc71739f2a`, bundle digest
  `2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572`, 19 features and
  20,112 route rows
- Implemented lifespan dependency initialization, bounded thread-safe TTL inference cache, structured
  ready/degraded health, model info, prediction, route reliability, retrieval and revisioned feedback
- Implemented recursive Decimal-safe serialization, conditional event/model/feedback persistence,
  strongly consistent reads and exact active-model baseline/identity metadata
- Added an idempotent, non-destructive DynamoDB provisioner for PAY_PER_REQUEST table
  `flight-delay-events` and GSI `event-date-created-at-index`
- Proved 119 hermetic tests at 80.67% branch coverage plus ready/degraded API container behavior; all
  three non-root Python 3.11 images build

## Traveler and monitoring applications

- Revalidated and merged the API and persistence work to `main` at
  `11216f2f77e279e3a82ae326a33327b44e84de02`,
  then created `feat/streamlit-monitoring` from that exact checkpoint
- Added the development-only DynamoDB endpoint path, pinned DynamoDB Local Compose service, exact
  idempotent table initialization, dummy local credentials, and overridable host ports
- Replaced the traveler placeholder with a typed FastAPI-only client and complete scheduled-flight,
  route-context, provenance, degraded-state, persistence, and feedback workflow
- Implemented the DynamoDB-only monitoring repository with fully paginated per-day GSI queries,
  31-day bound, filters, model metadata, inspection, and feedback adjudication
- Implemented deterministic operational, latency, distribution, PSI, Jensen-Shannon, target-indicator,
  and feedback performance metrics with honest unavailable states
- Replaced the monitor placeholder with filtered operational/model/feedback views and added a
  deterministic, dry-run-default, explicitly labeled, batch-cleanable local demo workflow
- Proved the real W&B staging v0 runtime with DynamoDB Local: ready health, two unique persisted events
  across a cache hit, retrieval, route context, feedback revision, monitoring query, and both UI health
  endpoints
- Passed 143 tests at 80.87% branch coverage and rebuilt all three non-root Python 3.11 images; no AWS
  Academy session or AWS service call occurred

## Deployment preflight and publication

- Revalidated all traveler and monitoring acceptance evidence and merged it to `main` at
  `02dccc4e3bd862b65df2e15b0de01215c24ca528`
- Added strict deployment-manifest and evidence-manifest validators, including exact committed
  release identity, digest-only GHCR references, environment-name allowlists, and safe filenames
- Added supported-host bootstrap plus API/traveler/monitor deployment scripts that validate mode-0600
  host env files, reject AWS credential/endpoint variables, replace only named containers, and prove
  health; all offer no-network dry-run validation
- Added a typed local/live smoke sequence for exact staging identity, unique persisted events, cache,
  retrieval, feedback revision, UI health, optional direct table verification, and explicitly gated
  labeled demo data
- Froze the three-host security group/environment topology, copy-ready live command sheet, complete
  evidence matrix, and phase-gated four-hour runbook with recovery/abort criteria
- Continued the hard zero-AWS boundary: no Learner Lab activation, credential validation, Console
  execution, AWS API call, or AWS resource interaction occurred
- Published the audited reachable history to the public repository, protected `main`, opened draft
  PR #1, and proved its complete `validate` job green in GitHub Actions
- Published three source-labeled public images by immutable GHCR digest, proved all three exact
  references anonymously pullable, materialized the strict deployment manifest, and passed the
  exact-digest local rehearsal with two persisted predictions, cache hit, feedback revision, direct
  DynamoDB Local verification, both UI health checks, and 30 demo events

## Serving-alias compliance

- Revalidated and normally merged deployment-preflight PR #1 at
  `521bb39bad46fbde328e9b386b39aebb3eb7a622`, then created
  `feat/production-promotion-automation` from that clean checkpoint
- Added `production` to the existing immutable Registry `v0` without uploading model bytes; retained
  `staging` and verified both aliases resolve digest `865ddd18f6debd44f24a79fc71739f2a`
  and bundle digest `2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572`
- Changed the release control plane to serve `production` while preserving the historical failed
  final-test gates and fixed threshold; added machine-readable `deployment_purpose=academic_demo`
  and `internal_production_gate_passed=false`
- Made API, traveler, monitor, manifest, smoke, and current deployment documentation expose the
  academic-demo/internal-gate disclosure independently of the alias name

## Automated selection and promotion

- Added versioned policy-as-code for exact Registry/project/dataset/schema/protocol scope, operational
  gates, deterministic multi-metric ranking, incumbent behavior, and dry-run/apply semantics
- Added a pure selector with strict candidate parsing, non-finite/missing metadata rejection,
  validation-only input enforcement, explicit outcomes, deterministic tie-breaking, and no W&B
  dependency
- Added an adapter that enumerates exact Registry versions, verifies immutable artifacts, performs
  idempotent preconditioned alias mutation through supported W&B Registry linking, and re-queries the
  exact postcondition; tests use an in-memory implementation
- Added sanitized JSON decision audits and a manual-only GitHub Actions workflow that validates the
  policy/tests, selects, optionally applies, verifies, logs a W&B audit run, and uploads the decision
- Proved a real W&B dry-run against `production:v0`: `retain_current`, no mutation, exact digest
  verified; the selector used development-validation metadata only

## Live AWS deployment

- Completed the reviewed live deployment in `us-east-1` on three separate `t3.small` EC2 hosts:
  `flight-api`, `flight-user-ui`, and `flight-monitor`
- Validated the `flight-delay-events` DynamoDB table, `PAY_PER_REQUEST` billing, String `pk`, and
  `event-date-created-at-index` GSI with `event_date`/`created_at` keys and `ALL` projection
- Served the unchanged Registry `production:v0` release for the course deployment while preserving
  `deployment_purpose=academic_demo` and `internal_production_gate_passed=false`
- Validated end-to-end prediction persistence, retrieval, revisioned feedback, Traveler-to-API
  communication, and the Monitor's direct DynamoDB query path
- Captured required AWS Console and live application evidence for host health, security groups,
  instance profiles, DynamoDB, API documentation, Traveler prediction/feedback, and Monitor views
- Preserved the immutable image digests and deployment manifest that describe the validated live
  deployment; subsequent source-only changes are not represented as deployed

## Historical model-development stop points

- Model remediation: independent calibration metrics, six fixed rolling-origin bases, and six calibrated
  November finalists are complete. Every finalist failed AP and F1, so December qualification and
  downstream work in that workflow correctly did not start.
- Model selection: exact time partitions, calibrated Candidate A, bounded Candidate B tuning, validation
  evaluation, thresholds, model artifacts, and W&B audit are complete. No candidate passed every
  mandatory validation gate, so that workflow stopped before selection/freeze or test access.
- Data pipeline and one-time final-test evaluation are complete; the final test is consumed.
- CI: repository-hosted pull-request evidence is green and `main` branch protection requires the
  `validate` context, one approving review, and resolved conversations.

## Current repository validation

- 291 tests pass with 85.94% branch coverage; CI enforces an 82% minimum
- Coverage includes failure and edge behavior for API clients/contracts, persistence, monitoring,
  deployment/evidence validation, promotion policy/metadata, and governed monitoring traffic
- Ruff lint and formatting checks, deployment/evidence manifest validation, and deployment shell
  syntax checks pass without contacting W&B or AWS
