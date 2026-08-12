# Final evidence checklist

The canonical mapping is `evidence/evidence_manifest.json`. Capture in filename order and update
each entry from `pending-live-session` to `captured` with its exact source URL. Run:

```bash
python scripts/validate_evidence_manifest.py --require-files
```

## GitHub and W&B (before lab activation)

- `01_github_repository.png`: public home and rendered README in a private browser.
- `02_github_pr_ci.png`: real PR to main with Ruff, format, coverage/tests, and three builds green.
- `03_wandb_project.png`: public project overview.
- `04_wandb_experiments.png`: comparable candidate/final-test run evidence.
- `05_wandb_dataset.png`: dataset artifact name, version, digest, and lineage.
- `06_wandb_registry.png`: Registry collection with `v0`, digest, and explicit `production` alias
  (retain `staging` if shown).

## AWS Console (live session only)

- `07_aws_ec2_instances.png`: three separately named running instances and status checks.
- `08_aws_security_groups.png`: least-privilege inbound relationships.
- `09_aws_instance_profile.png`: role/profile attached to API and monitor, not traveler.
- `10_aws_dynamodb_schema.png`: table key, GSI, ACTIVE state, and on-demand billing.
- `11_aws_dynamodb_prediction.png`: representative prediction with feedback fields; hide unrelated
  account identifiers.
- `12_aws_status_metrics.png`: EC2 basic status/metrics or CloudWatch status evidence.

## Live application (live session only)

- `13_app_api_docs.png`: FastAPI `/docs` endpoints.
- `14_app_health.png`: ready model/database dependency health.
- `15_app_model_info.png`: exact `production:v0` identity, immutable digests,
  `internal_production_gate_passed=false`, and `deployment_purpose=academic_demo`.
- `16_app_traveler_prediction.png`: traveler result with academic-demo/internal-gate notice.
- `17_app_traveler_feedback.png`: persisted feedback confirmation.
- `18_app_monitor_operations.png`: non-zero volume, latency, cache rate, and status/distribution.
- `19_app_monitor_drift.png`: target drift plus a calculable input drift metric and demo warning.
- `20_app_monitor_feedback.png`: coverage, feedback metrics/inspector, `production:v0` identity, and
  academic-demo/internal-gate notice.

Also retain `21_live_smoke_summary.json` as machine-readable supplemental evidence. Screenshots
must never display the W&B token, AWS credentials, `.env` content, SSH keys, or terminal history
containing secrets.
