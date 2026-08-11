# Public deliverable visibility audit

Audit date: 2026-08-10. These checks used unauthenticated HTTP requests and did not load the local
W&B API key. Each URL returned HTTP 200 without a redirect:

- [W&B project](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops)
- [dataset artifact v0](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/dataset/flight-delay-bts-sampled/v0)
- [one-time final-test run](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/w4te9tla)
- [Registry collection](https://wandb.ai/austin-youngblood-university-of-denver/registry/model?selectionPath=austin-youngblood-university-of-denver/wandb-registry-Model/us-flight-arrival-delay-15m)

The frozen Registry identity is `wandb-registry-Model/us-flight-arrival-delay-15m`, alias `staging`,
version `v0`, digest `865ddd18f6debd44f24a79fc71739f2a`. HTTP 200 proves that the
unauthenticated page shell is reachable; it does not prove every nested artifact panel renders to a
logged-out viewer. A fresh private-browser visual check of project, artifact, run, and Registry
content remains a pre-activation go/no-go capture. Any login wall or restricted nested panel must be
recorded as the precise visibility limitation rather than represented as public proof.

The public GitHub URL is reserved as
`https://github.com/austinyoungblood/us-flight-delay-mlops`, but no repository existed at audit time.
Publication is an open gate pending explicit authorization to publish the audited reachable history.
