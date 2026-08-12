# Brief 06 API and DynamoDB status

> Historical Brief 06 report. Brief 09A later moved the unchanged Registry `v0` to the course-required
> `production` serving alias. The current API exposes `internal_production_gate_passed=false`,
> `deployment_purpose=academic_demo`, and the persistent academic-demonstration governance notice.

## Implementation summary

Brief 06 starts from merged Brief 05 SHA `3bc9208c06f125f7d741699f7f89b441d1295dc3` on
`feat/api-dynamodb`. It implements the release-decision-backed serving runtime, complete FastAPI
backend, required event persistence adapter, active-model metadata and idempotent DynamoDB
provisioner. It does not implement either Streamlit UI, monitoring calculations, demo seeding, EC2,
CloudWatch or deployment automation.

## Exact serving release

- Control plane: `release/release_decision.json`; `serving_alias=staging`.
- Registry: `wandb-registry-Model/us-flight-arrival-delay-15m:v0`.
- Registry/source digest: `865ddd18f6debd44f24a79fc71739f2a`.
- Bundle digest: `2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572`.
- Selection-lock SHA-256: `a730a25c34a9f259b3ca02eb92c4ad44c1e75f50fd52ce270a940e4a60142340`.
- Route asset SHA-256: `193cd6ee089bfdbbe754752ea0aaf05b3e197650eec3e35eaa046e627d4ee0cd`.
- Classification threshold: `0.1840285229739868`.
- Live load: all 19 model features, 20,112 route rows, and deterministic canary passed.
- Notice: this is a Registry staging release and is not approved for production use.

## API contract

Implemented routes are `GET /health`, `GET /model-info`, `POST /predict`,
`GET /route-reliability`, `GET /predictions/{id}`, and `POST /feedback/{id}`. The lifespan performs
all external initialization. Readiness requires both the verified Registry runtime and DynamoDB.
Cache entries contain only deterministic inference/route output; cache hits still get a unique UUID,
UTC timestamp and persisted prediction event. A DynamoDB failure never returns prediction success.

## DynamoDB contract and external status

The adapter targets PAY_PER_REQUEST table `flight-delay-events`, String key `pk`, and ALL-projection
GSI `event-date-created-at-index` (`event_date`, `created_at`). It uses finite Decimal conversion,
conditional prediction creation, immutable-identity model metadata updates, existing-record feedback
revision, and strongly consistent retrieval.

The safe provisioner dry-run passed. The real AWS Academy attempt stopped on `DescribeTable` with
`UnrecognizedClientException: The security token included in the request is invalid`. Therefore the
real table and end-to-end persistence/cache/feedback smoke remain blocked; no AWS mutation occurred.

## Validation evidence

- Python 3.11.14.
- Ruff and format checks passed.
- 119 tests passed; 80.67% branch coverage.
- Real W&B Registry load and canary passed.
- API container returned structured 503/degraded without credentials and 200/ready with injected
  in-memory dependencies.
- API, user UI and monitoring UI non-root Python 3.11 images built successfully.

## Next exact increment

**user Streamlit + separate DynamoDB-backed monitoring dashboard consuming this API/data plane**
