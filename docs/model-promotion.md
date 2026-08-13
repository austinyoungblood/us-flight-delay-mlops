# Model selection and Registry promotion

This is the controlled model-selection and promotion lifecycle for the single Registry collection
`wandb-registry-Model/us-flight-arrival-delay-15m`. It selects only from compatible development and
validation evidence. January–May 2026 final-test evidence is historical release context and is
deliberately unavailable to the selector.

## Governance boundary

`configs/promotion_policy.yaml` pins the source project, collection, artifact type, dataset artifact
and digest, feature-schema hash, compatible evaluation protocol, mandatory operational gates,
maximum bundle size, ranking order, deterministic tie-breaking, target alias, and dry-run default.
Candidate metadata is strict: unknown or forbidden final/sealed-test fields are rejected, metrics
must be present and finite, and immutable lineage/inference identities must agree with policy.

The ranking order is average precision, F1, recall and ROC-AUC descending; Brier score, log loss and
bundle size ascending. An incumbent wins an exact metric tie, then `candidate_id` ascending is the
final stable tie-breaker. The selector returns `promote`, `retain_current`,
`no_eligible_candidate`, or `blocked_invalid_metadata` and has no W&B dependency.

`WandbRegistryAdapter` owns remote behavior. It enumerates explicit Registry versions (never
ambiguous `latest`), clean-downloads and hash-verifies candidate bundles, checks an alias precondition,
uses supported Registry linking for an exact candidate, and re-queries the postcondition. Selecting
the current exact production identity is an idempotent `retain_current` with zero mutation.

## Operator commands

Validate without network access:

```bash
PYTHONPATH=src python scripts/promote_model.py validate-policy
pytest tests/unit/test_promotion.py tests/integration/test_promotion_workflow.py
```

Run a real read-only decision (the default mode is dry-run):

```bash
PYTHONPATH=src python scripts/promote_model.py run \
  --mode dry-run \
  --target-alias production \
  --output promotion_decision.json \
  --log-wandb-run
```

Apply mode is permitted only by a deliberate manual GitHub `workflow_dispatch` with `dry_run=false`.
Ordinary pull-request CI never has Registry mutation behavior. Each attempt writes sanitized
`promotion_decision.json` containing policy/Git identity, candidates, metrics used, rejection reasons,
ranking, requested and actual actions, before/after alias identity, verification, and workflow
identity. Credentials are neither read into the record nor uploaded as evidence.

The current `production` alias is an academic/course deployment label. It does not certify the
model against the project's stricter internal production-quality gate, which remains failed and is
disclosed by release metadata, API responses, both user interfaces, deployment evidence, and docs.
