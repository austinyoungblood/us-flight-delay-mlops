# Frozen deployment package

This directory is the reviewed, digest-only package used for the validated time-bounded deployment.
It performs no AWS provisioning. `deployment_manifest.json` remains the source of truth for Git,
W&B, image, port, host, and DynamoDB identities. The package records historical deployment proof;
it does not claim that a live endpoint remains available.

Validate the package locally without contacting AWS or W&B:

```bash
python scripts/validate_deployment_manifest.py
python scripts/validate_evidence_manifest.py --require-files
bash -n deploy/*.sh deploy/lib/*.sh
DEPLOY_DRY_RUN=1 deploy/bootstrap_host.sh
```

The complete historical Console/CLI procedure, gates, evidence sequence, recovery rules, and report
contract remain in
[`docs/aws-deployment-walkthrough.md`](../docs/aws-deployment-walkthrough.md).

If the package is reused during a separately authorized deployment, copy the relevant template to a
mode-0600 file outside Git and run only its matching script. The scripts pull by digest, replace only
their named project container, set `unless-stopped`, and fail closed on health errors.

```bash
sudo deploy/deploy_api.sh --env-file /opt/us-flight-delay-mlops/api.env
sudo deploy/deploy_traveler.sh --env-file /opt/us-flight-delay-mlops/traveler.env
sudo deploy/deploy_monitor.sh --env-file /opt/us-flight-delay-mlops/monitor.env
```

The shared deployment wrapper supplies `HOME=/tmp` as a fixed runtime argument only for the
non-root API container. Do not add `HOME` to `api.env` or to the deployment manifest's host
environment-variable contract. Traveler and Monitor receive no `HOME` override.

The frozen topology consumes exact Registry `production:v0` and must display the
academic-demo/internal-gate disclosure. Any future live use requires separate authorization and all
documented pre-activation gates; local validation does not authorize AWS access.
