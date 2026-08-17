# Governed v1 model experiment protocol

Protocol-lock status: **precommitted before training**.
At protocol lock time, no v1 model had been trained and no v1 performance result existed. The
contract below remains frozen and unchanged.

Execution status: development later completed with a governed stop; see the
[v1 result report](v1-model-experiment-result.md). Production remains the immutable Registry
`production:v0` release.

The machine-readable source of truth is
[`configs/v1_experiment_protocol.yaml`](../configs/v1_experiment_protocol.yaml). Its byte-level
SHA256 and prerequisite artifact hashes are sealed in
[`experiments/v1/protocol_lock.json`](../experiments/v1/protocol_lock.json). The strict validator
does not import CatBoost, W&B, or the AWS SDK and does not access the network.

## Motivation and primary family

Instructor feedback motivated one governed nonlinear iteration. The primary challenger is
`CatBoostClassifier` with follow-on implementation version `catboost==1.2.10`. CatBoost is selected
for nonlinear tabular learning and native high-cardinality categorical handling on the existing
individual-flight binary classification task. The dependency is predeclared as modeling-only and
optional; it is not installed by this PR and must not enter API, Traveler, or Monitor images while
they serve v0.

Prophet was considered but intentionally not selected. Prophet naturally fits aggregate time-series
forecasting rather than individual-flight binary classification. A later non-gating research
notebook may explore daily network delay prevalence, but Prophet has no influence on v1 selection or
acceptance criteria.

## Immutable boundaries

- Production remains Registry `production:v0`, digest
  `865ddd18f6debd44f24a79fc71739f2a`, bundle SHA256
  `2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572`, and threshold
  `0.1840285229739868`.
- The consumed January 1-May 31, 2026 historical final test is permanently prohibited from v1
  training, tuning, calibration, threshold selection, qualification, feature decisions, and gate
  changes. `data/processed/test.parquet` must not be opened by v1 development commands.
- Development may use only the existing 2025 train/validation data. Every fit frame must be sorted
  by `flight_date`; that field is ordering-only, while `target` is label-only.
- No production alias, v0 byte, deployment artifact, W&B object, or AWS resource is changed by this
  protocol.

## Feature contract

The 20 model features are exactly `flight_delay.data.prepare.PROCESSED_FEATURES`:

`Month`, `DayofMonth`, `DayOfWeek`, `Reporting_Airline`, `Origin`, `Dest`, `CRSDepTime`,
`CRSArrTime`, `CRSElapsedTime`, `Distance`, `route`, `scheduled_departure_hour`,
`scheduled_arrival_hour`, `scheduled_departure_minute_bucket`,
`scheduled_arrival_minute_bucket`, `is_weekend`, `scheduled_departure_sin`,
`scheduled_departure_cos`, `scheduled_arrival_sin`, and `scheduled_arrival_cos`.

The categorical features are exactly `Reporting_Airline`, `Origin`, `Dest`, and `route`; all others
are numeric. The validator also invokes the central leakage guard, so postdeparture and unapproved
fields fail closed.

## Rolling-origin design and fixed grid

Four rolling-origin folds are frozen:

| Fold | Training interval | Validation interval |
| --- | --- | --- |
| FOLD_1 | 2025-01-01 to 2025-07-01 exclusive | July 2025 |
| FOLD_2 | 2025-01-01 to 2025-08-01 exclusive | August 2025 |
| FOLD_3 | 2025-01-01 to 2025-09-01 exclusive | September 2025 |
| FOLD_4 | 2025-01-01 to 2025-10-01 exclusive | October 2025 |

Validation folds never control early stopping. Iterations are fixed in exactly four candidates:

| Candidate | Depth | Iterations | Learning rate | L2 leaf regularization |
| --- | ---: | ---: | ---: | ---: |
| CB1 | 6 | 300 | 0.05 | 3 |
| CB2 | 8 | 300 | 0.05 | 5 |
| CB3 | 6 | 500 | 0.03 | 5 |
| CB4 | 8 | 500 | 0.03 | 7 |

All use CPU Logloss, seed 42, `has_time=true`, no class weights, no file writing, and no early
stopping. Bayesian, Optuna, random, and manual out-of-grid tuning are forbidden. Mean AP, mean
ROC-AUC, mean log loss, mean Brier score, AP standard deviation, then lexical ID rank the bases;
exactly two CatBoost bases advance. The immutable R3 SGD/sigmoid incumbent is reconstructed and
reported on the same folds as a control but cannot consume a CatBoost finalist slot.

## Refit, calibration, and November selection

The top two bases refit on January 1-October 31, 2025. November 1-15 is calibration-only, with
exactly `none`, `sigmoid`, and `isotonic` variants per base. The base is frozen before calibrated
variants see labels. This yields exactly six possible CatBoost finalists.

November 16-30 is used once per finalist for probability evaluation, threshold choice, and gates.
Eligible thresholds require recall at least 0.60, precision at least 0.30, and predicted-positive
rate at most 0.50. Eligible thresholds maximize F1, with deterministic ties by precision, recall,
lower positive rate, proximity to 0.50, then higher threshold. A finalist with no eligible threshold
fails.

Every November gate is mandatory:

- AP is at least incumbent AP + 0.01, ROC-AUC at least incumbent ROC-AUC + 0.005, and AP lift over
  prevalence at least 1.35.
- Brier skill is positive; Brier and log loss beat contemporaneous priors and do not exceed the
  incumbent November values.
- Absolute probability/prevalence gap and ECE15 are each at most 0.03.
- Recall is at least 0.60, precision at least 0.30, F1 at least 0.38, and positive rate at most 0.50.
- Single-row p95 latency is below 25 ms and the serialized bundle is below 10 MiB.
- Lineage, schema, leakage, deterministic reconstruction, serialization/load/inference, prohibited
  test access, and runtime/convergence checks all pass.

Zero passing finalists ends v1 development honestly with production:v0 retained. Multiple passing
finalists use the frozen ordering in the machine protocol; no post-result tuning is allowed.

## December retrospective qualification

December 2025 is explicitly a **retrospective temporal qualification holdback**, not a pristine
final test. The November-frozen winner is evaluated once without retraining, recalibration,
threshold adjustment, candidate switching, or gate changes. Mandatory gates require positive Brier
skill, better-than-prior log loss, probability gap and ECE15 at most 0.05, AP lift at least 1.25,
ROC-AUC at least 0.60, recall at least 0.55, precision at least 0.25, F1 at least 0.36, positive rate
at most 0.60, latency below 25 ms, bundle below 10 MiB, and all integrity checks.

A December failure is recorded and produces no v1 release candidate. A pass may freeze a candidate
but does not authorize production promotion.

## Future fresh final holdout

No fresh month is named or accessed here. The holdout will be the first complete DOT/BTS Reporting
Carrier On-Time Performance month strictly after May 31, 2026 that was unused by prior project
decisions, unopened for v1 development, available only after this lock, and identified with archive
SHA256 before label evaluation. The full eligible month is used without a sample cap and becomes
immutable once selected.

The frozen v1 challenger and production v0 incumbent are evaluated once on identical rows. V1 must
pass all absolute December-style quality gates and must match or beat v0 AP and ROC-AUC while
strictly improving Brier score and log loss. Any failure retains production:v0, with no holdout-based
retuning. A complete pass makes v1 only eligible for a separately reviewed promotion decision.

The eventual report includes 500 paired day-block bootstrap replicates, seed 42, resampled by
`flight_date`, with descriptive percentile 95% intervals for v1-minus-v0 AP, ROC-AUC, Brier score,
and log loss. These intervals cannot override pass/fail gates.

## Registry rule

This protocol performs no Registry action. A future passing candidate must use a new immutable
artifact/version, record its artifact digest and protocol SHA, and may use `candidate-v1`. It must
never overwrite or mutate v0 or silently move `production`; promotion requires the untouched fresh
holdout to pass and a separate reviewed decision.
