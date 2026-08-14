# Governed v1 model experiment implementation

## Current state

- **PROTOCOL MERGED** — the immutable protocol is
  `us-flight-delay-v1-catboost-rolling-origin-v1`, SHA-256
  `a6b1de9de550d1bd94eae0e56f8d88d65801ec488b6c539fc64afbafa4ccfffb`.
- **IMPLEMENTATION UNDER REVIEW** — the guarded development and qualification runners are
  implemented but have not been applied.
- **REAL v1 TRAINING NOT YET RUN** — there are no v1 BTS results, winner, qualification result, or
  v1 W&B run.
- **PRODUCTION STILL v0** — this implementation has no Registry, alias, image-publication, AWS, or
  deployment mutation path.

The governed lifecycle is intentionally split:

`Protocol -> Implementation -> Development execution -> December qualification -> Fresh final`

Each transition requires separately persisted evidence. Development cannot continue into December,
and December cannot run without the immutable winner model and lock produced by a successful
November decision.

## Dependency isolation

CatBoost is an optional modeling dependency, pinned as `catboost==1.2.10` in the `v1` project extra.
Its dependencies that are not already constrained by the base lock are isolated in
`requirements-v1.lock`. CI installs both constraint files for modeling tests.

The API, Traveler, and Monitor Dockerfiles still install the base project only. CI starts Python in
each built image and proves `importlib.util.find_spec("catboost") is None`. The deployed v0 runtime
therefore does not inherit the v1 experimentation stack.

## Module boundaries

- `v1_protocol.py` validates the immutable protocol, protocol hash, lock, artifact dependencies, and
  fixed design.
- `v1_data.py` owns canonical Parquet access, period predicates, pre-read hashes, post-read temporal
  checks, and the exact 20-feature/native-category adapter.
- `v1_catboost.py` lazily imports CatBoost only for applied fitting, constructs exactly CB1-CB4, and
  wraps frozen-base sigmoid/isotonic calibration.
- `v1_selection.py` is pure selection logic for probability evidence, rolling ranking, the new v1
  threshold selector, complete gates, and deterministic winner selection.
- `v1_execution.py` owns preflight, durable markers, incumbent reconstruction, rolling execution,
  candidate bundles, winner freeze, and December handoff.
- `v1_tracking.py` defines the tracker seam. Tests use an in-memory tracker; the online adapter
  imports W&B only when a real tracked run starts.

Core experiment logic has no W&B dependency and no model-artifact or Registry operation.

## Data and leakage controls

Only these paths can enter v1 development or qualification:

- `data/processed/train.parquet`
- `data/processed/validation.parquet`

The manifest identity and the selected Parquet SHA-256 are checked before every read. Development
reads training rows for January through October 2025 and requests validation rows with an explicit November predicate.
Qualification requests validation rows with an explicit December predicate.
Returned dates are independently checked against the requested half-open interval.

`validation.parquet` is never accepted as an unrestricted development read. `test.parquet` and
arbitrary paths are rejected centrally. The historical January-May 2026 test has no executable v1
access path.

The adapter accepts only `flight_date`, the exact 20 protocol features in order, and `target`.
`flight_date` is ordering-only and `target` is label-only. It rejects post-outcome fields, missing or
empty categories, non-finite numerics, and schema drift. The four categories remain stable strings
for CatBoost's native categorical interface; there is no one-hot encoding, target encoding, or
feature expansion.

## Development execution

`python scripts/run_v1_development.py` is preflight-only by default. It validates the protocol,
protocol lock, dependency isolation, dataset-manifest identity, four candidates, four folds, three
calibration variants, and prohibited-state boundaries. It reads no Parquet, imports no CatBoost
runtime or W&B module, contacts no network, and creates no marker unless a local output file is
explicitly requested.

A future applied run requires both `--apply` and `--tracking online`, a clean Git worktree, the
protocol merge as an ancestor, exact CatBoost 1.2.10 metadata, canonical dataset identity, no prior
marker, and a complete W&B environment. The durable marker is written before the first real fit.
Any exception records only the error type and failed stage and permanently stops automatic reruns.

### Incumbent reconstruction gate

The first fitted path is the governed R3-sigmoid incumbent reconstruction, using the existing R3
configuration, time partitions, frozen-estimator calibration, fixed threshold, and reference-metric
comparison. CatBoost challengers are blocked unless every reconstruction metric reproduces within
the existing tolerance.

R3-base rolling evidence is separate, descriptive context: it is the uncalibrated R3 estimator on
the four rolling folds and never consumes a CatBoost finalist slot. The governed R3-sigmoid reconstruction
is the incumbent trust gate and historical November comparison.

### CatBoost search and November decision

Each of CB1-CB4 is fitted independently on each precommitted rolling-origin fold with CPU execution,
seed 42, `has_time=True`, stable chronological input, native categorical names, fixed iterations,
and no weights, `eval_set`, early stopping, or best-iteration reuse. Aggregate ranking follows all
six protocol tie breaks and advances exactly two bases.

Each selected base is refitted once through October. Its raw, frozen-base sigmoid, and frozen-base
isotonic variants are then constructed. Tests verify that calibration never changes base
predictions; the raw variant never consumes calibration labels.

Exactly six variants are evaluated once on November 16-30. The new v1 selector enforces recall,
precision, and predicted-positive-rate eligibility, then all six deterministic threshold tie
breaks. Probability metrics, contemporaneous-prior comparisons, independent ECE15, latency,
serialization, complete bundle size, and all governance evidence are retained. Only all-gates-pass
variants can win, using the eight protocol winner tie breaks.

Candidate bundles are local and temporary. Every file is hashed, an aggregate digest is computed,
and a clean-loaded model must reproduce probabilities. No model artifact is published. A zero-pass
outcome writes only sanitized stop evidence and keeps v0. A winner creates a new immutable local
model and lock and still stops before December.

## December qualification

`python scripts/run_v1_december_qualification.py` is also preflight-only by default. Apply is refused
after a governed stop, without a completed November marker, without exact lock/model/schema/hash
agreement, or after any prior December start.

An applied qualification reads only the December validation predicate, clean-loads the exact frozen
winner, predicts once, and applies the locked threshold and protocol gates. It contains no fit,
calibration, candidate switch, or threshold update. A pass can create only a local release-candidate
lock pointing to the same November model SHA. It cannot create a W&B model artifact, link a Registry
version, mutate an alias, publish an image, or deploy.

## Fresh-final boundary

This implementation includes no downloader and names no future month in executable code. Future
fresh-final selection remains governed solely by the protocol rule. Any reusable comparison or
bootstrap logic must be proven only with synthetic fixtures until that separately reviewed phase.
