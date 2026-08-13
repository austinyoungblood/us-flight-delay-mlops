# AWS Academy deployment walkthrough

This is the operator checklist for the single authorized AWS Academy Learner Lab deployment session.
It translates the reviewed deployment specification into an exact execution order. Use it together
with `deploy/deployment_manifest.json`, `docs/aws-live-command-sheet.md`,
`docs/aws-security-groups.md`, and `docs/final-evidence-checklist.md`.

Do not start the lab until Section 1 is entirely green. Once the lab starts, do not edit application
source, retrain, change the threshold, move a W&B alias, change the DynamoDB schema, rebuild an image,
or substitute an image tag for a digest.

## 0. Security cleanup before doing anything else

The W&B key exposed during the promotion dry-run discussion must be treated as compromised.

1. In W&B, revoke that API key and generate a replacement.
2. Replace the GitHub Actions `WANDB_API_KEY` secret with the new key.
3. Store the replacement locally only in the ignored `.env` until the API host environment file is
   prepared. Never paste it into chat, a commit, a screenshot, or a shell command.
4. Delete `GHCR_PAT` from `.env` and revoke that temporary GitHub token; all deployment images are
   public and require no GHCR login.
5. Correct `WANDB_MODE=online,` to `WANDB_MODE=online` if the trailing comma is still present.
6. Confirm `.env` is ignored:

```bash
git check-ignore -v .env
git status --short
```

Expected: `.env` is reported by `git check-ignore`; `git status --short` does not list it.

## 1. Zero-AWS pre-activation gate

Complete these checks before selecting **Start Lab**.

### 1.1 Confirm the immutable identities

From the repository root:

```bash
git switch main
git pull --ff-only
git status --short
git log -1 --oneline
git merge-base --is-ancestor 1cafcab2b1ccec4dd2662a9ad9166fac9aa37ad4 HEAD
git merge-base --is-ancestor 355d99226883ebae1705d9f5a12eaffbe7bc6c8a HEAD
test -f docs/aws-deployment-walkthrough.md
python3 -S deploy/read_manifest.py image api
grep -Fx 'MODEL_DOWNLOAD_DIR=/tmp/flight-delay-model' deploy/env/api.env.template
PYTHONPATH=src .venv/bin/python scripts/validate_deployment_manifest.py
PYTHONPATH=src .venv/bin/python scripts/validate_evidence_manifest.py
bash -n deploy/*.sh deploy/lib/*.sh
DEPLOY_DRY_RUN=1 deploy/bootstrap_host.sh
```

Expected:

- the worktree is clean;
- both `git merge-base` commands exit `0`;
- the reviewed walkthrough and minimal-host deployment corrections are present on `main`;
- the manifest validator reports source freeze `355d99226883ebae1705d9f5a12eaffbe7bc6c8a`;
- Registry identity is `production:v0` at digest `865ddd18f6debd44f24a79fc71739f2a`;
- both validators and both shell checks pass.

The merged repository commit and image source revision are intentionally different. PR #2 merged at
`1cafcab2b1ccec4dd2662a9ad9166fac9aa37ad4`; all three images were built once from source revision
`355d99226883ebae1705d9f5a12eaffbe7bc6c8a`. Do not try to make those values equal.

### 1.2 Prove anonymous image access

```bash
docker logout ghcr.io || true
docker pull ghcr.io/austinyoungblood/us-flight-delay-mlops-api@sha256:7175844d53a46ed96c5cd3198e8fb6defbdf67bd0c640999914272b26e9433d4
docker pull ghcr.io/austinyoungblood/us-flight-delay-mlops-traveler@sha256:9afd05f6697609fbda7b130ff6e61afa29cab936981ae6f990fe5914fb71fb47
docker pull ghcr.io/austinyoungblood/us-flight-delay-mlops-monitor@sha256:7b038768c7474d7702909a747014e2725b77654d83aeb0fac1f1dac4db41ef62
```

Each output must show the exact requested digest. Do not log in to GHCR on the EC2 hosts.

### 1.3 Confirm public evidence and capture files 01–06

Use a private/incognito browser for GitHub, W&B project, dataset, and final-test pages. W&B Registry
is organization-restricted, so capture `production:v0` while authenticated; do not represent it as
anonymous evidence.

Capture these files under `evidence/` without showing tokens, `.env`, browser password overlays, or
terminal history:

1. `01_github_repository.png` — public repository and rendered README.
2. `02_github_pr_ci.png` — merged PR #2 and green required `validate` check.
3. `03_wandb_project.png` — project overview.
4. `04_wandb_experiments.png` — immutable historical final-test run evidence.
5. `05_wandb_dataset.png` — dataset `v0`, digest, and lineage.
6. `06_wandb_registry.png` — authenticated Registry `production:v0`, immutable digest, and retained
   `staging` alias.

Also retain the successful manual promotion dry-run URL in the live notes. The expected outcome is
`retain_current`, not an alias mutation.

### 1.4 Prepare the operator workspace

Open these files in separate tabs:

- this walkthrough;
- `docs/aws-live-command-sheet.md`;
- `docs/aws-security-groups.md`;
- `docs/final-evidence-checklist.md`;
- `evidence/evidence_manifest.json`;
- a private live-session notes page.

Prepare a sanitized notes template with fields for start/end UTC, region, VPC/subnet, instance
identities, security groups, instance profile, metadata options, DynamoDB identity, public/private
addresses, smoke IDs, demo batch ID, evidence filenames, deviations, and abort reason. Do not place
credentials or full environment-file contents in those notes.

**Gate 1:** Do not start the lab unless Sections 1.1–1.4 are all green and the replacement W&B key
is available.

## 2. Start the Learner Lab — T+00 to T+20

### 2.1 Start and record the session

1. Open AWS Academy Learner Lab.
2. Select **Start Lab** once.
3. Immediately record the UTC start time in the private session notes:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ
```

4. Open the AWS Console only after the indicator shows the lab is ready.
5. Establish AWS CLI access using the Academy-provided method. Keep temporary AWS credentials out of
   Git, container environment files, screenshots, notes, and shell history.

### 2.2 Discover live constraints

In the authenticated CLI environment:

```bash
export AWS_REGION=<ACADEMY_REGION>
aws sts get-caller-identity
aws configure get region
aws ec2 describe-vpcs --region "$AWS_REGION" --output table
aws ec2 describe-subnets --region "$AWS_REGION" \
  --query 'Subnets[].{SubnetId:SubnetId,VpcId:VpcId,AZ:AvailabilityZone,PublicIP:MapPublicIpOnLaunch}' \
  --output table
```

In the Console, identify the Academy-permitted instance profile, current supported Linux AMI, and an
allowed instance type with at least 2 GiB RAM. Record only non-secret identifiers. Do not create a
new IAM role unless the lab explicitly permits it and the frozen plan authorizes it.

### 2.3 Create or validate DynamoDB

```bash
AWS_REGION="$AWS_REGION" DYNAMODB_TABLE=flight-delay-events \
  PYTHONPATH=src .venv/bin/python infra/provision_dynamodb.py

aws dynamodb wait table-exists \
  --region "$AWS_REGION" \
  --table-name flight-delay-events

aws dynamodb describe-table \
  --region "$AWS_REGION" \
  --table-name flight-delay-events \
  --query 'Table.{Status:TableStatus,Key:KeySchema,Billing:BillingModeSummary.BillingMode,GSIs:GlobalSecondaryIndexes[].{Name:IndexName,Status:IndexStatus,Key:KeySchema,Projection:Projection.ProjectionType}}'
```

Expected:

- table `flight-delay-events` is `ACTIVE`;
- partition key is String `pk`;
- GSI `event-date-created-at-index` is `ACTIVE`;
- GSI keys are String `event_date` and `created_at`;
- projection is `ALL`;
- billing is `PAY_PER_REQUEST`.

**Gate 2:** Stop and record the sanitized error if credentials, region, DynamoDB, EC2 access, or an
instance-profile path is unavailable by T+20. Do not redesign the schema or application.

## 3. Create networking and three EC2 hosts — T+20 to T+65

### 3.1 Create the security groups

In **EC2 → Security Groups**, create all three in the same selected VPC:

1. `flight-user-ui-sg`
   - TCP 8501 from the exact reviewer/demo CIDR.
   - TCP 22 from **My IP** `/32` only if SSH is required.
2. `flight-api-sg`
   - TCP 8000 from `flight-user-ui-sg`.
   - Optional temporary TCP 8000 from operator `/32` for Swagger/smoke capture.
   - TCP 22 from operator `/32` only if required.
3. `flight-monitor-sg`
   - TCP 8501 from the exact reviewer/demo CIDR.
   - TCP 22 from operator `/32` only if required.

Keep normal outbound access needed for GHCR, W&B, DynamoDB, DNS, and bootstrap endpoints. Never open
SSH to `0.0.0.0/0`. Never open API 8000 to the world merely for convenience. If Academy permissions
reject the Traveler-SG source, use the Traveler private IP `/32` after launch and record the
deviation.

### 3.2 Launch the instances

Launch exactly three instances in the same VPC/subnet:

| Name | Security group | Instance profile | Public IPv4 | Application |
|---|---|---|---|---|
| `flight-api` | `flight-api-sg` | Academy-permitted profile | as needed for access/bootstrap | FastAPI :8000 |
| `flight-user-ui` | `flight-user-ui-sg` | none | yes | Streamlit :8501 |
| `flight-monitor` | `flight-monitor-sg` | Academy-permitted profile | yes | Streamlit :8501 |

For each instance:

1. Use the course-supported Amazon Linux or Ubuntu LTS AMI.
2. Use `t3.small` or the nearest permitted type with at least 2 GiB RAM.
3. Add tag `Project=us-flight-delay-mlops`.
4. Use the same VPC and subnet.
5. Attach only its listed security group.
6. Attach the instance profile only to API and Monitor.
7. Under advanced metadata options, use IMDS endpoint **enabled**, tokens **required**, and response
   hop limit **2** for API and Monitor when permitted.

Wait until every instance is `running` and all status checks pass. Record IDs, AMI/type, subnet/VPC,
public/private IPs, security groups, profile, and metadata options.

```bash
aws ec2 describe-instances --region "$AWS_REGION" \
  --filters 'Name=tag:Project,Values=us-flight-delay-mlops' 'Name=instance-state-name,Values=running' \
  --query 'Reservations[].Instances[].{Name:Tags[?Key==`Name`]|[0].Value,Id:InstanceId,Type:InstanceType,Private:PrivateIpAddress,Public:PublicIpAddress,Profile:IamInstanceProfile.Arn,Metadata:MetadataOptions}' \
  --output table
```

**Gate 3:** All three hosts must pass status checks and have the intended SG/profile relationships by
T+65. Otherwise preserve the strongest evidence and stop broad troubleshooting.

## 4. Bootstrap and deploy pinned images — T+65 to T+110

Perform these steps on API first, then Traveler, then Monitor. Use the course-approved connection
method. Never display the API environment file after adding the W&B key.

### 4.1 Get the reviewed deployment package on each host

If Git is not already installed, install it using the matching host command:

```bash
# Amazon Linux
sudo dnf install -y git

# Ubuntu (use this pair instead of the Amazon Linux command)
sudo apt-get update
sudo apt-get install -y git
```

Then clone reviewed `main`. The application images remain immutable at source revision `355d992…`;
the newer repository HEAD contains only merged deployment-safety/documentation corrections used by
the host scripts.

```bash
git clone https://github.com/austinyoungblood/us-flight-delay-mlops.git
cd us-flight-delay-mlops
git switch main
git pull --ff-only
git status --short
git merge-base --is-ancestor 1cafcab2b1ccec4dd2662a9ad9166fac9aa37ad4 HEAD
git merge-base --is-ancestor 355d99226883ebae1705d9f5a12eaffbe7bc6c8a HEAD
test -f docs/aws-deployment-walkthrough.md
python3 -S deploy/read_manifest.py image api
sudo DEPLOY_DRY_RUN=0 deploy/bootstrap_host.sh
```

Expected: reviewed `main`, clean status, both immutable ancestors present, digest-pinned API reference,
and Docker active. Record `git rev-parse HEAD` as the operator package SHA. Reconnect once if group
membership was added during bootstrap.

### 4.2 Prove instance-profile metadata on API and Monitor

On API and Monitor only, verify a role name exists at host level without printing credentials:

```bash
IMDS_TOKEN="$(curl --fail --silent --show-error -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)"
ROLE_NAME="$(curl --fail --silent --show-error \
  -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" \
  http://169.254.169.254/latest/meta-data/iam/security-credentials/)"
test -n "$ROLE_NAME"
printf 'host instance-profile role is available\n'
unset IMDS_TOKEN ROLE_NAME
```

Then prove the same role works inside the pinned component container:

```bash
IMAGE="$(python3 deploy/read_manifest.py image <api-or-monitor> \
  --manifest deploy/deployment_manifest.json)"
sudo docker pull "$IMAGE"
sudo docker run --rm \
  -e AWS_REGION="$AWS_REGION" \
  -e AWS_DEFAULT_REGION="$AWS_REGION" \
  "$IMAGE" python -c \
  'import boto3; value=boto3.client("sts").get_caller_identity(); print({"Arn": value["Arn"]})'
unset IMAGE
```

Expected: an assumed-role ARN for the Academy profile. If host metadata works but container STS does
not, inspect the instance metadata response hop limit before changing anything else. Never copy
temporary AWS credentials into a container environment file.

### 4.3 Deploy API

On `flight-api`:

```bash
sudo install -d -m 0750 -o root -g docker /opt/us-flight-delay-mlops
sudo install -m 0600 deploy/env/api.env.template /opt/us-flight-delay-mlops/api.env
sudoedit /opt/us-flight-delay-mlops/api.env
```

In the editor only:

- replace `__SET_AT_RUNTIME__` with the replacement W&B key;
- replace `AWS_REGION` with the actual Academy region;
- retain `DYNAMODB_TABLE=flight-delay-events`;
- retain `MODEL_DOWNLOAD_DIR=/tmp/flight-delay-model`.

Save, close, and do not print the file. Then:

```bash
sudo deploy/deploy_api.sh --env-file /opt/us-flight-delay-mlops/api.env
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/model-info
```

Confirm:

- Registry alias `production`, version `v0`;
- Registry/source digest `865ddd18f6debd44f24a79fc71739f2a`;
- bundle digest `2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572`;
- `deployment_purpose=academic_demo`;
- `internal_production_gate_passed=false`;
- Registry and DynamoDB dependencies are ready.

### 4.4 Deploy Traveler

On `flight-user-ui`:

```bash
sudo install -d -m 0750 -o root -g docker /opt/us-flight-delay-mlops
sudo install -m 0600 deploy/env/traveler.env.template /opt/us-flight-delay-mlops/traveler.env
sudoedit /opt/us-flight-delay-mlops/traveler.env
```

Replace `__API_PRIVATE_IP__` with the API instance's private IPv4 address. Then:

```bash
sudo deploy/deploy_traveler.sh --env-file /opt/us-flight-delay-mlops/traveler.env
curl --fail http://127.0.0.1:8501/_stcore/health
```

Open `http://<TRAVELER_PUBLIC_IP>:8501`. Confirm the form loads and the academic-demo/internal-gate
warning is visible. Traveler must have no W&B key and no AWS role.

### 4.5 Deploy Monitor

On `flight-monitor`:

```bash
sudo install -d -m 0750 -o root -g docker /opt/us-flight-delay-mlops
sudo install -m 0600 deploy/env/monitor.env.template /opt/us-flight-delay-mlops/monitor.env
sudoedit /opt/us-flight-delay-mlops/monitor.env
```

Replace `AWS_REGION` with the actual Academy region. Then:

```bash
sudo deploy/deploy_monitor.sh --env-file /opt/us-flight-delay-mlops/monitor.env
curl --fail http://127.0.0.1:8501/_stcore/health
```

Open `http://<MONITOR_PUBLIC_IP>:8501`. Confirm it loads, identifies `production:v0`, and shows the
academic-demo/internal-gate disclosure.

**Gate 4:** Do not proceed unless all three health checks pass, API identity is exact, DynamoDB is
ready, and both UIs display truthful governance. Inspect at most the named container's last 100 log
lines; never print its environment.

## 5. Run the real end-to-end smoke — T+110 to T+150

Use an Academy-authenticated operator shell that can reach all three application endpoints and
DynamoDB. Do not add `DYNAMODB_ENDPOINT_URL`; live mode rejects it.

```bash
export API_URL=http://<API_REACHABLE_IP>:8000
export TRAVELER_URL=http://<TRAVELER_PUBLIC_IP>:8501
export MONITOR_URL=http://<MONITOR_PUBLIC_IP>:8501
export DEMO_BATCH="deployment-$(date -u +%Y%m%dT%H%M%SZ)-evidence"

PYTHONPATH=src .venv/bin/python scripts/aws_end_to_end_smoke.py \
  --mode live \
  --api-url "$API_URL" \
  --traveler-url "$TRAVELER_URL" \
  --monitor-url "$MONITOR_URL" \
  --verify-dynamodb \
  --region "$AWS_REGION" \
  --seed-demo-batch "$DEMO_BATCH" \
  --output evidence/21_live_smoke_summary.json
```

The `deployment-` batch prefix is the validation format enforced by the reviewed smoke script. Do
not change the script or prefix during the session.

The JSON must prove:

- status `passed`;
- two different prediction IDs for identical request data;
- second prediction is a cache hit;
- both prediction records exist in real DynamoDB;
- retrieval and route context work;
- feedback revision `1` persists and round-trips;
- API, Traveler, and Monitor are ready;
- model identity is exact `production:v0` with academic-demo/internal-gate disclosure;
- 30 clearly labeled demo events are written.

Do not automatically rerun a mutating smoke after partial failure. First determine whether writes
occurred. Use a new batch ID only when the previous demo write did not occur.

**Gate 5:** API → DynamoDB, Traveler → API, and Monitor → DynamoDB must all be proven before evidence
capture.

## 6. Capture evidence — T+150 to T+210

Capture files 07–20 in exact order. Hide credentials, account details not required by the rubric,
`.env`, private keys, and browser password overlays.

1. `07_aws_ec2_instances.png` — three names, running state, and passed status checks.
2. `08a_aws_security_group_api.png`, `08b_aws_security_group_traveler.png`, and
   `08c_aws_security_group_monitor.png` — bounded inbound rules and Traveler → API relationship.
3. `09a_aws_iam_api.png`, `09b_aws_iam_traveler.png`, and `09c_aws_iam_monitor.png` — profile on
   API/Monitor and absent from Traveler.
4. `10a_aws_dynamodb_table.png` and `10b_aws_dynamodb_gsi.png` — table/GSI ACTIVE, keys,
   projection, and on-demand billing.
5. `11a_aws_dynamodb_prediction.png` and `11b_aws_dynamodb_feedback.png` — representative smoke item
   plus expanded feedback fields.
6. `12_aws_ec2_status_checks.png` — EC2 system, instance, and EBS status checks.
7. `13_app_api_docs.png` — `/docs` endpoints.
8. `14_app_health.png` — Registry and DynamoDB ready.
9. `15_app_model_info.png` — exact production identity and governance fields.
10. `16_app_traveler_prediction.png` — result and academic-demo warning.
11. `17_app_traveler_feedback.png` — persisted feedback confirmation.
12. `18_app_monitor_operations.png` — volume, latency, cache, status, distribution.
13. `19_app_monitor_drift.png` — target/input drift plus labeled-demo warning.
14. `20_app_monitor_feedback.png` — coverage, metrics, inspector, model identity/disclosure.

Update only the corresponding `status` and `source_url` fields in
`evidence/evidence_manifest.json`; do not mark an uncaptured criterion complete. Then run:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_evidence_manifest.py --require-files
```

Basic EC2 monitoring is sufficient: status metrics are emitted every minute and other basic metrics
normally use five-minute periods. Do not install the CloudWatch agent solely for screenshots.

## 7. Contingency and closeout — T+210 to T+240

Use remaining time only to correct runtime values, rerun an idempotent deployment script, or capture
missing evidence. Do not edit source, schema, model, release metadata, Registry aliases, or images.

Before ending the session:

1. Re-run the evidence validator with `--require-files`.
2. Copy `evidence/21_live_smoke_summary.json` and every screenshot off any EC2 host.
3. Record public URLs, exact resource IDs, deviations, final reachability, and UTC end time.
4. Confirm no secret appears in evidence, Git status, logs selected for submission, or notes.
5. Submit or preserve any required live-URL evidence while the lab is still active.
6. Use the Academy **End Lab** control when capture/reporting is complete or the four-hour limit is
   reached. Record whether ending the lab makes the applications unreachable.

If a source-code change would be required, stop. Preserve the sanitized error and open a separately
reviewed corrective change after the live session.

## 8. Final report checklist

Report:

- lab start/end UTC and total elapsed time;
- caller/region plus safe account context;
- VPC, subnet, instance IDs/types/AMI, public/private IPs, SGs, profile, and metadata options;
- DynamoDB table/GSI/billing/status;
- exact image and W&B identities;
- application URLs and final reachability;
- prediction IDs, cache result, feedback revision, and demo batch ID;
- evidence filenames and validator result;
- deviations, recoveries, blockers, and remaining rubric gaps;
- explicit attestation that no feature development, retraining, threshold change, schema redesign,
  alias mutation, image rebuild, or credential exposure occurred during the live session.

## Abort rules at a glance

Stop and preserve evidence when any of these occurs:

- Academy credentials are invalid or expire before critical provisioning;
- required EC2/DynamoDB permissions are unavailable;
- no allowed instance profile can authorize containerized API/Monitor after verifying IMDS hop limit;
- a pinned digest cannot run despite the same digest passing local rehearsal;
- identity differs from `production:v0` or the immutable digests;
- a fix would change application source, DynamoDB schema, model, release policy, or Registry alias;
- a secret appears in a capture—delete the unsafe local image and recapture safely.

Do not spend the full four-hour session improvising around a failed mandatory gate.

## AWS operator references

- [Configure EC2 instance metadata options](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-options.html)
- [IAM roles for Amazon EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html)
- [EC2 security groups](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-security-groups.html)
- [EC2 status checks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/monitoring-system-instance-status-check.html)
- [EC2 basic and detailed monitoring](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/manage-detailed-monitoring.html)
- [DynamoDB on-demand capacity](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/on-demand-capacity-mode.html)
