# Submission readiness

This checklist is a current navigation layer for the repository's implementation, governance, and
evidence. Historical stage reports preserve the state that existed when each stage was reviewed;
the README, this checklist, and the v1/v2/v3 result reports describe the final project state.

## Architecture

- [x] Three role-separated services are implemented: FastAPI owns model loading and prediction
  persistence, the Traveler UI calls FastAPI only, and the Monitor reads DynamoDB directly.
- [x] Model serving is fail-closed on the committed release decision, exact Registry identity,
  bundle hashes, leakage-safe feature schema, and DynamoDB readiness.
- [x] [Architecture](architecture.md) documents the validated three-host topology and ownership
  boundaries; [deployment manifest](../deploy/deployment_manifest.json) freezes application images
  by digest.

## Deployed services

- [x] The validated AWS evidence covers separate `flight-api`, `flight-user-ui`, and
  `flight-monitor` `t3.small` hosts in `us-east-1`.
- [x] The deployment exercised API readiness/model identity, Traveler prediction and feedback,
  DynamoDB persistence, and direct Monitor queries.
- [x] Evidence describes a time-bounded academic demonstration, not a continuously available public
  endpoint; no public IP or ephemeral hostname is committed.

## Governed v0 identity

- [x] The deployed incumbent remains W&B Registry `production:v0`, digest
  `865ddd18f6debd44f24a79fc71739f2a`.
- [x] Bundle SHA-256 is
  `2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572`; classification threshold
  is `0.1840285229739868`.
- [x] Runtime metadata discloses `deployment_purpose=academic_demo` and
  `internal_production_gate_passed=false`; the alias is not a claim of enterprise production
  readiness.
- [x] V0 remains deployed because v1, v2, and v3 all stopped at the locked November threshold
  eligibility frontier without creating a winner.

## Governed experiments

- [x] [V1 result](v1-model-experiment-result.md): CatBoost challenger completed with a governed
  stop; all six finalists returned `no_eligible_threshold`; December remained unopened.
- [x] [V2 result](v2-model-experiment-result.md): prior-month historical propensity features plus
  LightGBM/CatBoost improved rolling ranking, but all 12 November finalists returned
  `no_eligible_threshold`; December remained unopened.
- [x] [V3 result](v3-model-experiment-result.md): expanded 2024–2025 history, seasonal/temporal
  features, two boosting families, calibration, and ensembles produced 15 finalists; all returned
  `no_eligible_threshold`, so downstream gates were not evaluated and v0 was retained.
- [x] The governed v3 recovery preserved the original `status=started` marker, froze and hashed the
  incident evidence, applied a mathematically equivalent exact-selector correction, reconstructed
  completed advancement from immutable tracking evidence, reevaluated all finalists from scratch,
  and adopted the byte-identical `governed_stop` decision.
- [x] Compact public evidence is committed at
  [`experiments/v1/development_result.json`](../experiments/v1/development_result.json) and
  [`experiments/v2/development_result.json`](../experiments/v2/development_result.json), plus the
  v3 recovery result at
  [`experiments/v3/development_result.json`](../experiments/v3/development_result.json).
- [x] Raw decisions, model files, historical-state payloads, threshold tables, Parquet data, and
  W&B local files remain excluded from Git.

## CI/CD

- [x] Pull requests run Ruff lint, Ruff format verification, branch-inclusive pytest coverage,
  v1/v2/v3/deployment/evidence validators, deployment-shell syntax checks, and all three container
  builds.
- [x] Container checks prove the optional v1/v2/v3 modeling packages are absent from the runtime
  images.
- [x] Promotion is a separate manual, policy-checked workflow; experiment completion cannot mutate
  the Registry or deploy a model.
- [x] Branch protection requires the `validate` context, one approval, and resolved conversations.

## Monitoring

- [x] The Monitor reports operational volume/status, latency/cache behavior, model identity,
  prediction/target indicators, input drift, feedback metrics, and individual record inspection.
- [x] Synthetic monitoring traffic is API-only, deterministic, rate/count bounded, dry-run by
  default, and persisted with `traffic_source=synthetic_load_test` when explicitly applied.
- [x] The ignored August 15, 2026 batch audit records 150 planned, 150 successful, zero failed, and
  both audit and persistence validation passed. Its source files remain local:
  `artifacts/monitoring-load/2026-08-15-batch-150.json` (SHA-256
  `30adc1841ea31dc263407eea6e96f347cfb41964d8ae8b816e035b8350ac9bb3`) and
  `2026-08-15-batch-150.success` (SHA-256
  `303d8558dee360c33c6d4fe44e74ec66ecf9ab81a86365220c6f16381a0cc6aa`).
- [x] No successful August 16 or 17 monitoring batch is claimed; the available files for those
  dates are timeout logs only.

## Testing and coverage

- [x] The complete hermetic suite contains 802 tests, passes at 86.65% branch-inclusive coverage,
  and enforces an 86% minimum.
- [x] Tests cover feature leakage, data contracts, modeling governance, API/UI contracts,
  persistence, monitoring, deployment validation, protocol drift, offline isolation, and failure
  paths.
- [x] Test and validation commands require no AWS, Registry, W&B mutation, BTS training run, or
  historical-test access.

## Security and governance

- [x] `.env`, credentials, keys, raw/processed data, models, experiment artifacts, and W&B state are
  ignored.
- [x] Traveler has no AWS role; API and Monitor use the Academy `LabRole` only for their bounded
  DynamoDB responsibilities.
- [x] Security-group evidence preserves the Traveler-to-API private path and limits public UI/SSH
  access; the API is not intentionally exposed to the world.
- [x] Evidence manifest redaction instructions cover account identifiers, addresses, DNS values,
  operator CIDRs, and secrets before public submission.
- [x] V1/v2 development did not reopen the consumed January–May 2026 test, open December, alter the
  threshold, mutate `production:v0`, deploy, or contact AWS. The actual governed v3 execution and
  recovery likewise left the historical test, Registry, deployment, AWS, and production v0
  untouched and stopped before December qualification.
- [x] V3 discloses its separate pre-run boundary precisely: December 2025 was transiently
  materialized once during implementation testing, no model was scored against it and no decision
  used it, and the test was replaced with synthetic data before governed execution.

## Known limitations

- The AWS deployment evidence is a time-bounded Academy demonstration and does not promise a live
  endpoint after the lab session.
- The original captured monitoring views contain one organic end-to-end example; those drift and
  feedback values prove system wiring, not statistical production performance.
- The August 15 batch is explicitly synthetic load, not organic traveler behavior, and contains no
  fabricated feedback.
- V1, v2, and v3 all stopped at November threshold eligibility. V3's additional history,
  seasonality, weighting, calibration, and ensembles still could not produce a simultaneous
  `P >= .30 / R >= .60 / PPR <= .50` operating point, so precision/recall generalization remains an
  open modeling problem.
- The W&B project and artifact page shells were checked for public reachability, but nested views may
  still depend on W&B's current anonymous-access behavior.

## Evidence paths

| Evidence | Path |
| --- | --- |
| Project overview and usage | [`README.md`](../README.md) |
| Architecture | [`docs/architecture.md`](architecture.md) |
| Implementation history | [`docs/implementation-status.md`](implementation-status.md) |
| V0 release decision | [`release/release_decision.json`](../release/release_decision.json) |
| Deployment manifest | [`deploy/deployment_manifest.json`](../deploy/deployment_manifest.json) |
| Evidence manifest | [`evidence/evidence_manifest.json`](../evidence/evidence_manifest.json) |
| Evidence checklist | [`docs/final-evidence-checklist.md`](final-evidence-checklist.md) |
| Curated AWS/application captures | [`aws/screenshots/`](../aws/screenshots) |
| W&B/GitHub visibility audit | [`docs/public-deliverables.md`](public-deliverables.md) |
| V1 report / compact JSON | [`docs/v1-model-experiment-result.md`](v1-model-experiment-result.md) / [`experiments/v1/development_result.json`](../experiments/v1/development_result.json) |
| V2 report / compact JSON | [`docs/v2-model-experiment-result.md`](v2-model-experiment-result.md) / [`experiments/v2/development_result.json`](../experiments/v2/development_result.json) |
| V3 report / compact JSON | [`docs/v3-model-experiment-result.md`](v3-model-experiment-result.md) / [`experiments/v3/development_result.json`](../experiments/v3/development_result.json) |
