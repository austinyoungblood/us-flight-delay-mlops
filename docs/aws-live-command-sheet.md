# AWS live-session command sheet

Execute this sheet only during the one Brief 09 Learner Lab session, after every gate in
`brief09-step-by-step.md` is green. Replace every `<...>` value only after the approved session is
active. Do not paste credentials into a command, screenshot, shell history, repository file, or
evidence record.

## Session facts and DynamoDB

```bash
export AWS_REGION=<ACADEMY_REGION>
aws sts get-caller-identity
aws configure get region
aws iam get-instance-profile --instance-profile-name <LEARNER_LAB_PROFILE_NAME>
AWS_REGION="$AWS_REGION" DYNAMODB_TABLE=flight-delay-events python infra/provision_dynamodb.py
aws dynamodb describe-table --region "$AWS_REGION" --table-name flight-delay-events
```

Expected table: partition key `pk` (String), GSI `event-date-created-at-index` with
`event_date` partition key and `created_at` sort key, billing `PAY_PER_REQUEST`.

## Console launch values

- Names: `flight-api`, `flight-user-ui`, `flight-monitor`; tag each `Project=us-flight-delay-mlops`.
- Type: `t3.small` default, or the closest Academy-permitted equivalent with at least 2 GiB RAM.
- AMI: current course-supported Amazon Linux or Ubuntu LTS, x86_64 or arm64 matching the images.
- Place all three in the same Learner Lab VPC and public subnet; enable public IPv4 for the two
  Streamlit hosts and for any host that needs the course SSH path.
- Attach the verified Learner Lab instance profile to API and monitor. Traveler needs no role.
- Apply the exact groups in [aws-security-groups.md](aws-security-groups.md).
- Access via the course-provided EC2 Instance Connect/session route, or SSH with the session's
  authorized key. Record the selected path; never commit the key.

Discover IPs after launch:

```bash
aws ec2 describe-instances --region "$AWS_REGION" \
  --filters 'Name=tag:Project,Values=us-flight-delay-mlops' 'Name=instance-state-name,Values=running' \
  --query 'Reservations[].Instances[].{Name:Tags[?Key==`Name`]|[0].Value,Private:PrivateIpAddress,Public:PublicIpAddress}' \
  --output table
```

## Per-host bootstrap and deploy

Clone reviewed `main` after the Brief 09 walkthrough/deployment-safety correction is merged. Brief 09
merge commit `1cafcab2b1ccec4dd2662a9ad9166fac9aa37ad4` and manifest source SHA
`355d99226883ebae1705d9f5a12eaffbe7bc6c8a` must both be ancestors of `HEAD`. The source SHA
identifies the bytes used to build the immutable images; it is not expected to equal repository
`HEAD`. Run the strict manifest validator before activation and the standard-library manifest reader
on each host; do not deploy a feature branch.

```bash
sudo DEPLOY_DRY_RUN=0 deploy/bootstrap_host.sh
sudo install -d -m 0750 -o root -g docker /opt/us-flight-delay-mlops
sudo install -m 0600 deploy/env/<COMPONENT>.env.template /opt/us-flight-delay-mlops/<COMPONENT>.env
sudoedit /opt/us-flight-delay-mlops/<COMPONENT>.env
```

Replace the API private-IP placeholder only in traveler env. Put the operator-supplied W&B
token only in API env. Then execute API first, traveler second, monitor third:

```bash
sudo deploy/deploy_api.sh --env-file /opt/us-flight-delay-mlops/api.env
sudo deploy/deploy_traveler.sh --env-file /opt/us-flight-delay-mlops/traveler.env
sudo deploy/deploy_monitor.sh --env-file /opt/us-flight-delay-mlops/monitor.env
```

Inspect without printing environment values:

```bash
docker ps --filter 'name=flight-delay-'
docker logs --tail 100 flight-delay-api
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/model-info
curl --fail http://127.0.0.1:8501/_stcore/health
```

Run smoke from the reviewed workstation or API host after replacing public URLs:

```bash
python scripts/aws_end_to_end_smoke.py --mode live \
  --api-url http://<API_REACHABLE_IP>:8000 \
  --traveler-url http://<TRAVELER_PUBLIC_IP>:8501 \
  --monitor-url http://<MONITOR_PUBLIC_IP>:8501 \
  --verify-dynamodb --region "$AWS_REGION" \
  --seed-demo-batch brief08-<YYYYMMDDTHHMMSSZ>-evidence \
  --output evidence/21_live_smoke_summary.json
```
