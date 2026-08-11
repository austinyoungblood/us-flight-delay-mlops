# Frozen security-group matrix

These are Console field values for the eventual single Learner Lab session. They are not
commands to run during Brief 08.

| Group | Attached host | Inbound | Source | Outbound |
|---|---|---|---|---|
| `flight-user-ui-sg` | `flight-user-ui` | TCP 8501 | `<GRADER_OR_DEMO_CIDR>` | TCP 8000 to `flight-api-sg`; HTTPS/DNS as VPC requires |
| `flight-api-sg` | `flight-api` | TCP 8000 | `flight-user-ui-sg` | HTTPS 443 for W&B and DynamoDB endpoints; DNS as VPC requires |
| `flight-monitor-sg` | `flight-monitor` | TCP 8501 | `<GRADER_OR_DEMO_CIDR>` | HTTPS 443 for DynamoDB endpoints; DNS as VPC requires |
| each host group | matching host | TCP 22 only if course access requires SSH | `<OPERATOR_CIDR>/32` | — |

For direct Swagger/smoke access, temporarily add TCP 8000 from `<OPERATOR_CIDR>/32`; remove it
after capture. Never open API port 8000 to `0.0.0.0/0` merely for convenience.

Preferred API ingress is the user-UI security-group reference. If Academy permissions do not
allow SG references, use only the user UI instance's private `/32` address, record the
limitation, and update it if that address changes. Public Streamlit ingress may use the exact
grader range; use `0.0.0.0/0` only when the grader range is unavailable, with that exception
called out in the evidence notes.

API and monitor use the course-provided EC2 instance profile for DynamoDB. Traveler receives
no AWS or W&B credentials. No security group should expose DynamoDB because it is an AWS API,
not a port on these instances.
