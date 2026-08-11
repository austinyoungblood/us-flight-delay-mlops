# Public deliverable visibility audit

Audit date: 2026-08-11. The W&B checks used unauthenticated HTTP requests and did not load the local
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

The full audited reachable Git history is published at the public
[GitHub repository](https://github.com/austinyoungblood/us-flight-delay-mlops). `main` is frozen at
the accepted Brief 07 merge (`02dccc4e3bd862b65df2e15b0de01215c24ca528`), and the Brief 08 work is
reviewable in [draft PR #1](https://github.com/austinyoungblood/us-flight-delay-mlops/pull/1). Its
GitHub Actions `validate` job proves Ruff, format, 151 tests with branch coverage, both strict
manifest validators, shell syntax, and all three container builds. Branch protection requires that
context, one approval, and resolved conversations before merge.

The three frozen GHCR artifacts are:

- `ghcr.io/austinyoungblood/us-flight-delay-mlops-api@sha256:2ef0ddc8cba713706834f62de617ae8fade3caec6dcf1def34bbef0e227c0c5e`
- `ghcr.io/austinyoungblood/us-flight-delay-mlops-traveler@sha256:3e9c7962b3867001a3c7636eb714d9dec5d97a0d24e5c1097ceb0cac5f33e987`
- `ghcr.io/austinyoungblood/us-flight-delay-mlops-monitor@sha256:2360169bd7398bc21dd8b9b5864567f64795caad46768aace32a8253dc0efb64`

All three carry source revision `8cfaf275e579bd9d6420450dec8e537014df5f2a`. Public visibility and
anonymous exact-digest pulls are a final publication gate and must be proven independently of an
authenticated publisher pull.
