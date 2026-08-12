# Brief 07 UI and monitoring status

> Historical Brief 07 report. Brief 09A later changed the unchanged `v0` serving alias to
> `production`. Both current UIs derive their persistent warning from governance metadata—not the
> alias—so they continue to disclose the failed internal production-quality gate.

## Delivered boundary

Brief 07 completes a local three-service workflow around the accepted staging backend. The traveler
application is an HTTP client of FastAPI only. The monitoring application is a DynamoDB client only.
The API remains the exclusive model loader and prediction writer. No AWS Academy session was started,
no AWS credentials were inspected, and no AWS service endpoint was called.

## Traveler application

- Typed `httpx` client for health, model information, prediction, route reliability, retrieval, and
  feedback, with separate 5-second connect and 15-second read timeouts.
- No automatic POST retries, preventing duplicate prediction or feedback events.
- Readiness/degraded behavior, persistent staging/academic warning, leakage-safe scheduled-flight
  form, optional historical route context, full prediction provenance, session-state preservation,
  and revisioned observed-outcome feedback.
- Errors are normalized for display without leaking request internals or credentials.

## Monitoring data plane and metrics

- Inclusive UTC date partitions with a hard 31-day bound and complete pagination of
  `event-date-created-at-index`; GSI results are explicitly eventually consistent.
- Carrier, route, request-status, and model-version filtering; strongly consistent single-prediction
  inspection; feedback correction; exact model metadata read or one bounded metadata-only scan.
- Request/success/error/cache measures; p50/p95/max total, inference, and persistence latency;
  probability/risk/model-version distributions; predicted-positive prevalence versus training
  prevalence; numeric PSI and categorical Jensen–Shannon divergence with epsilon smoothing; feedback
  coverage, accuracy, precision, recall, F1, and Brier score.
- Missing baselines, empty samples, and undefined denominators produce explicit unavailable states,
  never invented zeroes. The prevalence delta is labeled as an indicator, not measured accuracy.

## Demo-data safety

The seeder is deterministic and dry-run by default. Every item has `demo_data=true`, a required
`demo_batch_id`, and unmistakably synthetic provenance; it does not invoke the model. Writes are
conditional and cannot overwrite prediction IDs. Mutation is refused unless an explicit
`DYNAMODB_ENDPOINT_URL` is supplied. Cleanup requires an exact batch ID and conditionally deletes only
matching records. The dashboard warns when demo rows are present and can exclude them.

## Local evidence (2026-08-10 America/Denver)

- Python 3.11.14; Ruff lint and format checks passed.
- 143 tests passed at 80.87% branch coverage, including Streamlit AppTest empty, degraded, populated,
  filter, demo-exclusion, inspector, and feedback paths.
- API, traveler, and monitoring Python 3.11 images built. Compose provisioned the exact PAY_PER_REQUEST
  `flight-delay-events` table and ALL-projection GSI in `amazon/dynamodb-local:2.6.1`.
- Required tags were built as API `05dcd17cc786…`, traveler `8b6284f83cdb…`, and monitor
  `8b7cc6ccd510…`; all configure user `app`. File inspection found no `.env`, release bundle, or
  `model.joblib` in either UI image.
- Real W&B Registry staging v0 loaded and verified. `/health` returned ready with both dependencies;
  both Streamlit health endpoints returned HTTP 200.
- Two API calls generated unique persisted prediction IDs for the same request; the second was an
  inference-cache hit. Retrieval, route evidence, and feedback revision 1 all returned HTTP 200.
- The bounded monitoring query returned 14 events: 12 clearly labeled demo rows and two real local API
  predictions. It resolved model metadata v0 and calculated operational and feedback metrics.
- Demo dry run, seed, query, and exact-batch cleanup were exercised only against local port 18001 with
  dummy credentials. Alternate host ports were used because port 8000 was already occupied; Compose
  now supports non-breaking host-port overrides while retaining the documented defaults.

## Deferred boundary

There is no cloud deployment evidence in Brief 07. The exact next increment is **deployment preflight
and evidence runbook with no AWS calls; only after that review, activate one AWS Academy session for
DynamoDB recovery + three-EC2 deployment + end-to-end evidence capture**.

## Future deployment evidence checklist (not executed)

- Preflight records exact branch/commit, image digests, release-decision hashes, required ports,
  environment names, IAM assumptions, expected table/GSI contract, health commands, and teardown.
- After explicit review, record one Learner Lab session start/end and current credential expiry without
  ever writing credentials into evidence.
- Prove DynamoDB recovery/validation without schema replacement, then capture separate API, traveler,
  and monitor EC2 identity, process, port, and non-root container evidence.
- Capture API ready/degraded health, unique persisted predictions across a cache hit, feedback revision,
  bounded monitoring query, staging banners, and traveler/monitor screenshots.
- Capture table/item evidence with secrets and personal data redacted, then execute and record teardown
  before the Academy session expires.
