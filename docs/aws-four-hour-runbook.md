# Frozen four-hour AWS session runbook

This is the operational timebox for the one authorized Brief 09 AWS Academy session. Follow the
detailed `brief09-step-by-step.md` operator checklist. Before activation, every preflight gate below
must be green; otherwise do not start the lab.

## Pre-activation go/no-go

- Brief 09A/09B PR #2 is normally merged, its manual promotion dry run is green, `main` is clean,
  the Brief 09 walkthrough/deployment-safety correction is merged, and the deployment SHA is published.
- Public GitHub and README work in a private browser; PR Ruff, format, tests/coverage, and all
  three builds are green.
- Required W&B URLs work unauthenticated, or the precise visibility limitation is recorded.
- Three externally downloadable image references exactly match the manifest digests.
- Deployment/evidence validators, `bash -n`, dry runs, full test suite, image builds, and local
  DynamoDB rehearsal pass using the exact pinned images.
- Evidence filenames are allocated; the W&B key is available to the operator but is not in Git.
- Course lab access path is known without starting or validating the lab.

Any red item is a no-go. Fix it outside AWS and review this runbook again.

## T+00–20 — session identity and data plane

Execute the first section of the command sheet, record region and role/profile name, provision
the table, and verify key/GSI/billing identity.

- Go: caller/region are expected and table becomes ACTIVE with exact schema.
- No-go: identity is surprising, role is unavailable, table differs, or T+20 is reached.
- Recovery: retry only a known transient table status. Otherwise capture sanitized error/status,
  stop provisioning, and preserve existing evidence. Do not change application schema.

## T+20–65 — network and three hosts

Create the three reviewed security groups, launch exactly three named instances in one VPC,
attach the role to API/monitor, and record private/public IPs.

- Go: all instances have 2/2 status, intended groups, expected IPs, and roles.
- No-go: group permissions force broad exposure, role attachment fails, or hosts are not healthy.
- Recovery: apply the documented private-CIDR SG fallback once. Recreate only the affected host
  from the same launch fields. At T+65 capture the strongest topology evidence and stop broad
  investigation.

## T+65–110 — immutable application deployment

Bootstrap and deploy API first. Prove ready health and exact `production:v0` identity plus the
academic-demo/internal-gate disclosure, then deploy
traveler with API private IP and monitor with the shared table.

- Go: manifest SHA matches checkout; digest pulls succeed; three named containers are running;
  API and both Streamlit health checks pass.
- No-go: any digest differs, image is unavailable, secret is exposed, API identity differs, or a
  deployment-only retry does not clear the issue.
- Recovery: inspect only the named container's last 100 log lines, correct host env values, and
  rerun its idempotent deploy script. Never rebuild, retag, retrain, or mutate an alias. If an
  external registry is unavailable, use the already rehearsed digest-verified save/load fallback
  only if it exists in the manifest notes.

## T+110–150 — smoke and labeled monitoring batch

Run the exact smoke command with direct table verification and one unique demo batch. Retain its
JSON summary and verify representative prediction/feedback items.

- Go: health, identity, two unique records, cache proof (or explicitly documented timing miss),
  retrieval, feedback revision, direct table checks, UI health, and demo write all pass.
- No-go: persistence or identity fails, duplicate batch is reported, or dashboard cannot query.
- Recovery: run read-only health/retrieval checks, refresh the monitor cache, and use a new batch
  ID only if no writes occurred. Never retry a mutating smoke request automatically.

## T+150–210 — evidence capture

Follow `final-evidence-checklist.md` in numeric filename order. Capture AWS, application, GitHub,
and W&B identities with academic-demo/internal-gate warnings visible and without env values, tokens, SSH material,
account identifiers beyond what the rubric requires, or browser password-manager overlays.

- Go: every manifest criterion is captured and its source URL/instance field is updated.
- No-go: any rubric group lacks proof or an image exposes sensitive material.
- Recovery: delete the unsafe local screenshot, close sensitive panels, and recapture. Prefer one
  clear screenshot per criterion over exploratory browsing.

## T+210–240 — contingency and export

Run the evidence validator with `--require-files`, export the smoke JSON and final URLs, capture
only missing/unsafe evidence, and record live-only deviations.

- Go: evidence manifest validates, public URLs resolve, and the final report has exact identities.
- No-go: a missing item cannot be captured by T+235.
- Recovery: preserve strongest evidence, document the exact gap and reason, and stop. Do not begin
  feature development or extend the session into architecture changes.

At T+240, stop all work. The session objective is deployment and evidence, not feature repair.
