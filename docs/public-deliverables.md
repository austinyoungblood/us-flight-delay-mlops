# Public deliverables and visibility status

This page separates the final repository state from time-bounded public-access checks. It does not
claim that external endpoints or nested W&B views remain continuously available.

## Public project surfaces

The public source repository is
[`austinyoungblood/us-flight-delay-mlops`](https://github.com/austinyoungblood/us-flight-delay-mlops).
The final hermetic baseline contains 809 passing tests at 86.69% branch-inclusive coverage, with an
86% minimum enforced by pull-request CI. CI also runs Ruff, formatting, all v1/v2/v3 offline
validators, deployment/evidence validation, shell syntax checks, three service-image builds, and
runtime dependency-isolation probes.

An unauthenticated visibility audit on 2026-08-12 returned HTTP 200 for these public W&B page shells:

- [W&B project](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops)
- [dataset artifact v0](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/dataset/flight-delay-bts-sampled/v0)
- [one-time final-test run](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/w4te9tla)
- [Registry collection](https://wandb.ai/austin-youngblood-university-of-denver/registry/model?selectionPath=austin-youngblood-university-of-denver/wandb-registry-Model/us-flight-arrival-delay-15m)

That historical check proved page-shell reachability at the time, not perpetual availability or
anonymous rendering of every nested panel. The final scrub did not contact W&B or revalidate the
network claims.

## Frozen model and application identity

The Registry collection is `wandb-registry-Model/us-flight-arrival-delay-15m`. The served release is
alias `production`, version `v0`, digest `865ddd18f6debd44f24a79fc71739f2a`, with bundle SHA-256
`2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572` and threshold
`0.1840285229739868`. The alias is scoped to `deployment_purpose=academic_demo`; the immutable
metadata still reports `internal_production_gate_passed=false`.

The latest provenance-enabled application release was published from source revision
`ce10f1a123bbe21eb75ca31b2681caf90ccda731`:

- API: `ghcr.io/austinyoungblood/us-flight-delay-mlops-api@sha256:8c70e59c1cd24be98be5e47fd318464d7bae95aaf1be44608af3b33adacbca0e`
- Traveler: `ghcr.io/austinyoungblood/us-flight-delay-mlops-traveler@sha256:06d36b32304b9f7711d0b224fa9d7f049a8875b761dffd907c17a73f3eebef94`
- Monitor: `ghcr.io/austinyoungblood/us-flight-delay-mlops-monitor@sha256:97a4e6bb99358e8cfe6885581713fd2731f64ea97164ad6a4d64f6efb1c7277c`

Each digest passed an anonymous pull with an isolated Docker configuration at publication time. This
same source revision and image set is the single identity frozen in the
[deployment manifest](../deploy/deployment_manifest.json) and directly validated during the
time-bounded AWS deployment. Later monitoring audits prove application behavior and persistence;
they do not imply continuous endpoint availability. The [evidence index](../evidence/evidence_manifest.json)
records the corresponding time-bounded proof.

## Governed challenger evidence

All applied challenger tracks retained `production:v0`:

- [v1 result](v1-model-experiment-result.md): six November finalists, no eligible threshold.
- [v2 result](v2-model-experiment-result.md): 12 November finalists, no eligible threshold.
- [v3 result and recovery](v3-model-experiment-result.md): 15 November finalists, no eligible
  threshold after governed recovery.

Threshold ineligibility short-circuited each workflow before downstream gates. The reports do not
represent unexecuted downstream gates as failures. No challenger reopened the consumed historical
final test, advanced to December qualification, changed the threshold, or mutated the Registry.

## Final smoke and monitoring evidence

The sanitized final smoke summary records `status=passed`, two unique predictions, two application
retrievals, a cache hit on the second prediction, persisted feedback, and ready API, Traveler, and
Monitor services. It intentionally omits endpoints, account information, payloads, credentials, and
local paths. See [`final_live_smoke_summary.json`](../evidence/final_live_smoke_summary.json).

For UTC 2026-08-14 through 2026-08-19 with demo data excluded, the Monitor recorded 461 requests,
461 successes, zero errors, 34 cache hits (`7.4%`), and `28.353 ms` p95 latency. Source attribution
was 305 `synthetic_load_test`, 155 `legacy_unattributed`, and one `traveler_ui`.

The August 15 scheduled API-only batch and August 19 operator-invoked batch each completed 150/150
with zero failures and passed persistence validation. The synthetic total also includes five
provenance canaries; it is not 305 scheduled events or organic traveler behavior. The scheduled
August 16 and 17 attempts timed out before generation and remain documented availability failures.
See the [monitoring evidence report](monitoring-evidence.md) for audit hashes and interpretation.

## Evidence navigation

- [Submission readiness](submission-readiness.md)
- [Architecture](architecture.md)
- [Model card](model-card.md)
- [Final-test historical decision](final-test-report.md)
- [Deployment manifest](../deploy/deployment_manifest.json)
- [Evidence manifest](../evidence/evidence_manifest.json)
- [Final evidence checklist](final-evidence-checklist.md)
- [Curated screenshots](../aws/screenshots)
