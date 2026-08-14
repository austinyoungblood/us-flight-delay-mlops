# Final evidence checklist

The canonical post-session mapping is `evidence/evidence_manifest.json`. It records whether each
criterion is required, whether it is verified by `public_url`, `screenshot`, or `supplemental`
evidence, and whether that evidence is available. It supports multiple files for one screenshot
criterion and flags captures that still require redaction before public release. Run:

```bash
python scripts/validate_evidence_manifest.py --require-files
```

## GitHub and W&B public URLs

These deliverables are verified through the externally inspectable URLs retained in the manifest;
screenshots are not rubric-required:

- public GitHub repository and Actions/PR history;
- W&B project overview and historical run;
- W&B dataset artifact and Registry collection.

## AWS Console

- `07_aws_ec2_instances.png`: three separately named running instances and status checks.
- `08a_aws_security_group_api.png`, `08b_aws_security_group_traveler.png`, and
  `08c_aws_security_group_monitor.png`: least-privilege inbound relationships.
- `09a_aws_iam_api.png`, `09b_aws_iam_traveler.png`, and `09c_aws_iam_monitor.png`: `LabRole` on API
  and Monitor and no role on Traveler.
- `10a_aws_dynamodb_table.png` and `10b_aws_dynamodb_gsi.png`: table key, GSI, ACTIVE state,
  projection, and on-demand billing.
- `11a_aws_dynamodb_prediction.png` and `11b_aws_dynamodb_feedback.png`: representative prediction
  and expanded feedback fields.
- `12_aws_ec2_status_checks.png`: API system, instance, and EBS status checks.

## Live application

- `13_app_api_docs.png`: FastAPI `/docs` endpoints.
- `16_app_traveler_prediction.png`: traveler result with academic-demo/internal-gate notice.
- `17_app_traveler_feedback.png`: persisted feedback confirmation.
- `18_app_monitor_operations.png`: non-zero volume, latency, cache rate, and status/distribution.
- `19_app_monitor_drift.png`: target drift plus a calculable input drift metric and demo warning.
- `20_app_monitor_feedback.png`: coverage, feedback metrics/inspector, `production:v0` identity, and
  academic-demo/internal-gate notice.

Dedicated `14_app_health.png` and `15_app_model_info.png` captures are documented as non-required
supplemental evidence. They were not captured and are not reported as rubric failures. The local
`21_live_smoke_summary.json` is also supplemental and must not substitute for required AWS or live
application screenshots.

Screenshots must never display the W&B token, AWS credentials, `.env` content, SSH keys, or terminal
history containing secrets. Apply every `redaction-required` instruction in the evidence manifest
before a public commit.
