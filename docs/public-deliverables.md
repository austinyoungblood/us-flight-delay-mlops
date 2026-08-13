# Public deliverable visibility audit

Audit date: 2026-08-12. The W&B checks used unauthenticated HTTP requests and did not load the local
W&B API key. Each URL returned HTTP 200 without a redirect:

- [W&B project](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops)
- [dataset artifact v0](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/dataset/flight-delay-bts-sampled/v0)
- [one-time final-test run](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/w4te9tla)
- [Registry collection](https://wandb.ai/austin-youngblood-university-of-denver/registry/model?selectionPath=austin-youngblood-university-of-denver/wandb-registry-Model/us-flight-arrival-delay-15m)

The final pre-AWS Registry identity is `wandb-registry-Model/us-flight-arrival-delay-15m`, aliases
`production` and `staging`, version `v0`, digest `865ddd18f6debd44f24a79fc71739f2a`.
`production` is the course-required academic deployment alias; it does not supersede the recorded
failed internal production-quality gate. HTTP 200 proves that the
unauthenticated page shell is reachable; it does not prove every nested artifact panel renders to a
logged-out viewer. A fresh private-browser visual check of project, artifact, run, and Registry
content remains a pre-activation go/no-go capture. Any login wall or restricted nested panel must be
recorded as the precise visibility limitation rather than represented as public proof.

The full audited reachable Git history is published at the public
[GitHub repository](https://github.com/austinyoungblood/us-flight-delay-mlops). `main` contains the
accepted deployment-preflight merge (`521bb39bad46fbde328e9b386b39aebb3eb7a622`), and the
serving-alias/promotion work is documented in
[PR #2](https://github.com/austinyoungblood/us-flight-delay-mlops/pull/2). Its
GitHub Actions `validate` job proves Ruff, format, 179 tests with 80.21% branch coverage, both strict
manifest validators, shell syntax, and all three container builds. Branch protection requires that
context, one approval, and resolved conversations before merge.

The three frozen GHCR artifacts are:

- `ghcr.io/austinyoungblood/us-flight-delay-mlops-api@sha256:7175844d53a46ed96c5cd3198e8fb6defbdf67bd0c640999914272b26e9433d4`
- `ghcr.io/austinyoungblood/us-flight-delay-mlops-traveler@sha256:9afd05f6697609fbda7b130ff6e61afa29cab936981ae6f990fe5914fb71fb47`
- `ghcr.io/austinyoungblood/us-flight-delay-mlops-monitor@sha256:7b038768c7474d7702909a747014e2725b77654d83aeb0fac1f1dac4db41ef62`

All three carry source revision `355d99226883ebae1705d9f5a12eaffbe7bc6c8a`. On 2026-08-12 the GitHub
Packages API reported each package `public`. After `docker logout ghcr.io`, anonymous pulls of all
three exact references succeeded and returned the expected digests; no publisher credential was
available to those pulls.
