# Monitoring and provenance evidence

This document records the final validated monitoring state and preserves earlier implementation
evidence as explicitly historical context. The deployed Registry identity remained
`production:v0`, with `deployment_purpose=academic_demo` and
`internal_production_gate_passed=false` throughout.

## Final multi-day state

For UTC 2026-08-14 through 2026-08-19, with demo data excluded, the live Monitor reported:

| Metric | Evidence |
| --- | ---: |
| Total requests | 461 |
| Successful requests | 461 (`100%`) |
| Errors | 0 |
| Cache hits | 34 (`7.4%`) |
| Latency p95 | `28.353 ms` |
| `synthetic_load_test` | 305 |
| `legacy_unattributed` | 155 |
| `traveler_ui` | 1 |

A fresh Streamlit session filtered to `traffic_source=synthetic_load_test` returned 305 requests,
reducing the monitored population from 461 to 305. The 305 synthetic events comprise five
provenance-canary events plus the August 15 and August 19 batches of 150. They are not all scheduled
batches, do not represent organic traveler activity, and are not statistical model-performance
evidence.

## Batch provenance

Both applied batches sent leakage-safe scheduled-flight requests only through `POST /predict`, used
`traffic_source=synthetic_load_test`, created no feedback, and passed application persistence
validation.

| UTC batch date | Invocation | Planned | Successful | Failed | Audit SHA-256 | Success-sentinel SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-08-15 | Scheduled | 150 | 150 | 0 | `30adc1841ea31dc263407eea6e96f347cfb41964d8ae8b816e035b8350ac9bb3` | `303d8558dee360c33c6d4fe44e74ec66ecf9ab81a86365220c6f16381a0cc6aa` |
| 2026-08-19 | Operator-invoked | 150 | 150 | 0 | `4369f05df2eb816440a5bf18de9800ad388038fdfa0b0706a24e1d195b4433fd` | `1664dbe1ead21b78ded234a7b826e4a3afdbcc06dd12222b2cca1b407d604328` |

The August 19 audit additionally records 150 unique prediction IDs, `model_version=v0`, model
digest `865ddd18f6debd44f24a79fc71739f2a`, and load-generator source Git SHA
`aa681b2ecdf3ac09b58c97556d5fa44dabb55748`.

The scheduled August 16 and August 17 attempts timed out before traffic generation. They remain
availability failures; no successful batch or generated traffic is claimed for either date.

## Final live smoke

The final live smoke passed with API, Traveler, and Monitor ready. It created two unique predictions,
retrieved both through the application contract, observed an inference-cache hit on the second, and
persisted revisioned feedback. Application persistence and retrieval were verified; the smoke did
not claim a separate workstation-side DynamoDB SDK inspection.

The sanitized summary is
[`evidence/final_live_smoke_summary.json`](../evidence/final_live_smoke_summary.json). The raw ignored
smoke record is bound by SHA-256
`6a6af69da3b5b930fa6127387e7abf0de020f3095dc728f7ce8ab2d6ad988b54`.

## Monitoring data plane and metrics

The Monitor reads DynamoDB directly through its bounded data-plane client:

- Inclusive UTC date partitions with a hard 31-day bound and complete pagination of
  `event-date-created-at-index`; GSI results are explicitly eventually consistent.
- Carrier, route, request-status, model-version, and traffic-source filtering; strongly consistent
  single-prediction inspection; feedback correction; and exact model metadata lookup.
- Request, success, error, and cache measures; p50/p95/max total, inference, and persistence
  latency; probability, risk, and model-version distributions.
- Predicted-positive prevalence versus training prevalence; numeric PSI and categorical
  Jensen-Shannon divergence with epsilon smoothing.
- Feedback coverage, accuracy, precision, recall, F1, and Brier score.

Missing baselines, empty samples, and undefined denominators produce explicit unavailable states,
never invented zeroes. Drift and prevalence deltas are indicators, not measured accuracy.

## Traffic and demo-data safety

The monitoring-load generator is deterministic and dry-run by default. Applied execution requires
`--apply`, a credential-free HTTP(S) API origin, 1-500 requests, and no more than 10 requests per
second. It uses the API-only `flight_delay.load_testing` package and has no boto3, botocore, AWS
credential, AWS endpoint, direct DynamoDB, W&B mutation, feedback fabrication, or final-test
dependency.

The separate demo-data seeder is also deterministic and dry-run by default. Every demo item has
`demo_data=true` and a required batch ID. Writes require an explicit local DynamoDB endpoint and
cannot overwrite prediction IDs; cleanup deletes only exact matching batches. The dashboard exposes
and can exclude demo rows.

## Historical local implementation evidence

The initial local-runtime stage completed the same three-service ownership boundary before any live
deployment: the Traveler used FastAPI only, the Monitor used DynamoDB only, and the API remained the
exclusive model loader and prediction writer. At that historical checkpoint:

- 143 tests passed at 80.87% branch coverage.
- Three Python 3.11 images ran non-root against DynamoDB Local.
- Two API requests produced unique persisted IDs, the second request was a cache hit, and retrieval
  plus feedback returned HTTP 200.
- A bounded query returned 12 labeled demo records and two local API predictions.
- No AWS service was contacted during that local stage.

Those values are retained as implementation chronology, not the final test baseline or live
monitoring population. The final repository baseline is 809 tests at 86.69% branch-inclusive
coverage with an enforced 86% minimum.
