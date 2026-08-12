# Frozen deployment package

This directory is the reviewed, digest-only package for one later AWS Academy session.
It performs no AWS provisioning. `deployment_manifest.json` is the source of truth for Git,
W&B, image, port, host, and DynamoDB identities.

Before AWS activation, validate locally:

```bash
python scripts/validate_deployment_manifest.py
python scripts/validate_evidence_manifest.py
bash -n deploy/*.sh deploy/lib/*.sh
DEPLOY_DRY_RUN=1 deploy/bootstrap_host.sh
```

On each eventual host, copy the relevant template to a mode-0600 file outside Git and run
only its matching script. The scripts pull by digest, replace only their named project
container, set `unless-stopped`, and fail closed on health errors.

```bash
deploy/deploy_api.sh --env-file /opt/us-flight-delay-mlops/api.env
deploy/deploy_traveler.sh --env-file /opt/us-flight-delay-mlops/traveler.env
deploy/deploy_monitor.sh --env-file /opt/us-flight-delay-mlops/monitor.env
```

The live topology remains staging-only. AWS provisioning and these live commands must not be
run until every pre-activation gate in the four-hour runbook is green.
