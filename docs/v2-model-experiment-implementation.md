# Governed v2 implementation

The v2 implementation completed its one-time governed development execution at implementation SHA
`6966562dcc2a7959f27e662e97cfeec8a4aa43a6`. It returned `decision=governed_stop` after all 12
November finalists had `status=no_eligible_threshold`; no winner lock exists, and December remained
unopened. The commands still default to offline preflight, and the durable marker prevents a second
applied development run. See the [governed v2 result](v2-model-experiment-result.md).

## Components

- `flight_delay.modeling.v2.protocol` verifies the byte-locked YAML and JSON protocol, every
  dependency hash, the immutable v1 evidence, exact 16/12 candidate matrices, 37-feature schema,
  temporal cutoffs, backend policy, gates, and unchanged production v0 identity.
- `flight_delay.modeling.v2.features` builds the canonical historical-state artifact, empirical-Bayes
  full-history and trailing-three-month lookups, SHA256 identity, and the shared batch/single-row
  transformer.
- `flight_delay.modeling.v2.data` admits only canonical train/validation Parquet sources, refuses
  `test.parquet`, samples model rows deterministically, builds historical state from full prior-month
  rows, and keeps December behind the separate qualification path.
- `flight_delay.modeling.v2.models` lazily imports the two optional model packages, preserves native
  categoricals, separates GPU/CPU execution metadata from candidate identity, disables outer-fold
  early stopping, and freezes base estimators before score calibration.
- `flight_delay.modeling.v2.selection` implements the high-recall primary metric, all eight CPU
  ranking tie breaks, unchanged November threshold selection, mandatory gates, and deterministic
  winner/governed-stop behavior.
- `flight_delay.modeling.v2.workflow` runs candidates sequentially, screens CatBoost on GPU and
  LightGBM on CPU, confirms the top four per family on CPU, advances the CPU-ranked top two per
  family, refits four CPU bases, and evaluates exactly 12 November variants.
- `flight_delay.modeling.v2.execution` provides one-time markers, clean-main enforcement, W&B group
  identity, immutable local decision/winner/state artifacts, production-v0 checks, and the separate
  no-refit December qualification handoff.

## Commands

These commands are safe offline preflights and do not read Parquet or import model runtimes:

```bash
python scripts/validate_v2_protocol.py
python scripts/run_v2_development.py
python scripts/run_v2_december_qualification.py
```

The two runners require both `--apply` and `--tracking online` for a real governed execution. Apply
also requires a clean `main`, exact optional dependency versions, complete W&B entity/project
configuration, and absent v2 markers. Development stops before December. Qualification refuses to
open December unless a hash-verified November winner, model, and October-31 state all exist.

## Dependency isolation

`lightgbm==4.7.0` and `catboost==1.2.10` live only in the optional `v2` environment and its constraints
file. Production Dockerfiles still install the base package without `v1` or `v2` extras. CI builds
all three images and asserts that neither model package is importable inside them.

## Validation and result boundary

All implementation tests use generated rows, fake constructors, in-memory tracking, and temporary
markers/bundles. They cover temporal leakage, state parity, family/backend sequencing, ranking,
gates, winner/stop behavior, December handoff, and dependency isolation without fitting BTS data or
contacting W&B, AWS, or the Registry.

The applied result retained only sanitized public evidence in Git. Raw decisions, thresholds,
historical-state payloads, candidate bundles, and model files remain ignored. Threshold eligibility
short-circuited before downstream November gates, so no downstream gate pass/fail claim is made.
Registry `production:v0` remains unchanged.
