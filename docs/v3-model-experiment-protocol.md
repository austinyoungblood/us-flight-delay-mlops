# Governed v3 seasonality and temporal-generalization protocol

Status: **precommitted before training**. No v3 model has been fit, no v3 result exists, December
2025 remains unopened, January–May 2026 remains permanently prohibited, and production remains
Registry `production:v0`.

The machine-readable source of truth is
[`configs/v3_experiment_protocol.yaml`](../configs/v3_experiment_protocol.yaml), SHA256
`061be599fd84a4ddbf06229c300fe4670272d176b22899f1515332923376ecff`. Its digests, feature schema,
weight policies, carried-forward configurations, candidate identities, and ensemble definitions are
sealed in [`experiments/v3/protocol_lock.json`](../experiments/v3/protocol_lock.json).

## Motivation

V2 completed with `governed_stop`. Its CPU-confirmed rolling-origin candidates reached roughly
0.336–0.338 mean precision at recall ≥ 0.60, but November degraded sharply: the best November
finalist, `CB04-none/sigmoid`, achieved precision 0.278481, recall 0.600840, and predicted-positive
rate 0.426617 against a locked requirement of precision ≥ 0.30, recall ≥ 0.60, PPR ≤ 0.50.

The failure is seasonal and temporal, not capacity-related. V3 therefore targets temporal
generalization: it adds a prior calendar year of history, prior-year seasonal state, deterministic
holiday structure, and recency weighting — while carrying forward v2's hyperparameters unchanged.
No gate is weakened, no model family is added, and no broad hyperparameter campaign is run.

## Immutable boundaries

- v0/v1/v2 protocol, lock, result, release, deployment, and manifest bytes are immutable and their
  SHA256 values are verified by the v3 dependency tree on every load.
- `data/manifests/source_manifest.json` and `data/manifests/processed_manifest.json` are never
  mutated. V3 writes `v3_source_manifest.json` and `v3_processed_manifest.json` beside them.
- Development may read 2024-01-01 through 2025-11-30 only. January 2024 is burn-in and contributes
  historical state but no model rows; model fitting begins 2024-02-01.
- December 2025 must not be read or decoded during development. It becomes readable only through a
  separate one-time qualification command after a November winner is frozen.
- January–May 2026 is permanently consumed and prohibited. It is excluded from the v3 source
  manifest entirely, so v3 data preparation cannot reach it.
- The genuine final test remains the first complete untouched BTS month strictly after 2026-05-31.
- V3 performs no AWS, Registry, deployment, or production action. Production stays `v0`, and the
  runtime API/Traveler/Monitor images gain no v3 modeling dependencies.

## Data expansion

Calendar year 2024 is added to the BTS Reporting Carrier On-Time Performance source. The v3 source
manifest covers 24 archives, 2024-01 through 2025-12, aggregating 709,735,704 bytes with digest
`673cac214739e8c0d2991a1bdbd1591a90e8907d7cf5bdbc34caddd72015b6af`. The twelve 2025 archives are
reused byte-identically from the v0 manifest — the downloader verified each against the existing
record and skipped it — so v0/v1/v2 lineage remains unchanged. Raw and generated data stay
Git-ignored; only the small canonical JSON manifests are versioned.

The v3 processed dataset is **uncapped**. V3 requires full eligible prior history for historical
state and all eligible model rows for the authoritative refit, so preparation applies no monthly
sample cap. Runtime is instead controlled at candidate-search time by the 50,000-row-per-month
deterministic cap. This is the one place v3 deliberately departs from the v0/v1/v2 preparation,
which sampled 75,000 rows per month; that older dataset is left untouched.

## Feature contract (48 features)

All 37 v2 features are retained unmodified. The schema is ordered as the 20 base schedule features,
then 6 deterministic seasonal features, then the 17 v2 historical features, then 5 seasonal
historical features.

### Seasonal historical features (prior occurrences only)

1. `prior_same_calendar_month_global_delay_rate`
2. `prior_same_calendar_month_carrier_delay_rate`
3. `prior_same_calendar_month_origin_delay_rate`
4. `prior_same_calendar_month_destination_delay_rate`
5. `prior_same_calendar_month_route_delay_rate`

These read same-calendar-month lookup tables keyed by calendar month plus entity, smoothed with the
same empirical-Bayes rule as v2 (prior strength 50, fallback to the historical global rate). Every
state cutoff is strictly before the first day of the model-row month, so any same-calendar-month
entry present in the state necessarily originates in a strictly earlier year; the implementation
asserts this invariant when the state is built. November 2025 features may therefore draw on
November 2024, and no November 2025 target can contribute to a November 2025 feature.

### Deterministic schedule-only seasonal features

6. `day_of_year_sin`, 7. `day_of_year_cos` — with `doy` the 1-based ordinal day and `days_in_year`
   366 in Gregorian leap years else 365, the angle is `2π(doy − 1)/days_in_year`.
8. `days_to_thanksgiving`, 9. `is_thanksgiving_window`
10. `days_to_christmas`, 11. `is_christmas_window`

All six are computed solely from the scheduled flight date, take no label input, and are identical
in batch training and single-row serving.

#### Bounded day-distance semantics

Thanksgiving is defined algorithmically as the **fourth Thursday of November**:
`November 1 + ((3 − November_1.weekday()) mod 7) + 21 days`. Christmas is December 25.

For each holiday the anchor candidates are that holiday in the flight date's year minus one, its
year, and its year plus one. The selected anchor minimises the absolute signed delta; ties resolve
to the **earlier** anchor date (reachable when consecutive anchors are an even number of days
apart). The signed delta is

```
raw = (anchor_date − scheduled_flight_date).days
```

so a positive value means the holiday is still in the future. The published distance feature is
`clip(raw, −30, +30)` as an integer. Bounding keeps the feature finite and monotone near the
holiday and constant far from it, so one anchor rule yields identical values in batch and single-row
paths without any dependence on surrounding rows.

The window flags use the **raw** signed delta; because both windows lie inside ±30, clipping cannot
affect them:

| Flag | Inclusive raw range | Meaning |
| --- | --- | --- |
| `is_thanksgiving_window` | `[−4, +2]` | Tuesday before Thanksgiving through the Monday after (7 days) |
| `is_christmas_window` | `[−10, +4]` | December 21 through January 4 (15 days) |

### Native categorical treatment (8 columns)

`Reporting_Airline`, `Origin`, `Dest`, `route`, `Month`, `DayOfWeek`, `scheduled_departure_hour`,
and `scheduled_arrival_hour` are handled by each library's native categorical interface. The four
integer columns are already model inputs; treating them natively changes their encoding only and
duplicates no target-derived information.

## Recency weighting

Exactly two training-weight policies are precommitted:

| Policy | Definition |
| --- | --- |
| `UNIFORM` | every fit row has weight 1 |
| `EXPONENTIAL_120D` | `weight = 0.5 ** (age_days / 120)`, then normalised to mean 1 within the fit partition |

`age_days = (fit_cutoff_date − scheduled_flight_date).days`, measured backward from the fit cutoff
(2025-07-31, 2025-08-31, 2025-09-30, and 2025-10-31 for FOLD_1…FOLD_4, and 2025-10-31 for the full
refit). Evaluation, calibration, and selection rows are never weighted. Weight policy is part of
candidate identity; execution backend is not.

## Candidate identities (8)

Four v2 configurations are carried forward with their exact committed hyperparameters — `LGBM12`,
`LGBM10`, `CB07`, `CB04` — crossed with the two weight policies:

`LGBM12-UNIFORM`, `LGBM12-EXP120`, `LGBM10-UNIFORM`, `LGBM10-EXP120`,
`CB07-UNIFORM`, `CB07-EXP120`, `CB04-UNIFORM`, `CB04-EXP120`.

No hyperparameter is re-searched, and no identity may be added after results are observed.

## Rolling folds

| Fold | Fit | Evaluate | State as of |
| --- | --- | --- | --- |
| FOLD_1 | 2024-02-01 → 2025-07-31 | August 2025 | 2025-07-31 |
| FOLD_2 | 2024-02-01 → 2025-08-31 | September 2025 | 2025-08-31 |
| FOLD_3 | 2024-02-01 → 2025-09-30 | October 2025 | 2025-09-30 |
| FOLD_4 | 2024-02-01 → 2025-10-31 | November 2025 | 2025-10-31 |

November is explicit development evidence in v3 because v1 and v2 already consumed it. Outer
evaluation is never used for early stopping.

## Ranking

Every fold scores the maximum precision achievable at recall ≥ 0.60 and PPR ≤ 0.50, or 0 if no
threshold qualifies. Candidates rank within family by:

1. highest worst-fold operating precision
2. highest FOLD_4 (November) operating precision
3. highest mean FOLD_2–FOLD_4 operating precision
4. highest mean all-fold operating precision
5. highest mean average precision
6. highest mean ROC-AUC
7. lowest mean log loss
8. lowest mean Brier score
9. lexical candidate ID

This deliberately prioritises temporal robustness over strong summer performance.

## Execution and advancement

LightGBM runs on CPU with bounded parallelism. CatBoost screening may use GPU device 0 with
sequential fits; CPU confirmation is authoritative for all advancement. Candidate fits are
sequential and stage runtimes are logged.

- 8 screening identities → top 2 per family
- 4 CPU-confirmed identities → top 1 per family
- 2 authoritative full-data CPU base refits

Historical state always uses full eligible prior history. Candidate search may use the deterministic
class-stratified 50,000-rows-per-month cap with seed 42; the authoritative refit uses all eligible
model rows.

## Final refit, calibration, and ensembles

The full base refit covers 2024-02-01 through 2025-10-31 with the November feature state frozen at
2025-10-31. Calibration fits on November 1–15, 2025; threshold selection uses November 16–30, 2025.

Each of the two bases yields `none`, `sigmoid`, and `isotonic` variants. Probability ensembles of
the two **uncalibrated** base scores are precommitted at 25/75, 50/50, and 75/25 LightGBM/CatBoost,
each also with `none`, `sigmoid`, and `isotonic`. Ensembles require no additional base-model fit.
That is 6 base variants plus 9 ensemble variants — **15 finalists**.

## November acceptance

Unchanged from v2 and mandatory in full: precision ≥ 0.30, recall ≥ 0.60, PPR ≤ 0.50, F1 ≥ 0.38,
plus the proper-scoring, calibration, discrimination, latency, serialization, lineage, leakage,
seasonal prior-year, weight-policy, and governance gates. If no finalist passes every gate, the
outcome is `governed_stop` and production remains `v0`. November is development evidence, not a
pristine qualification set.

## December qualification

December 2025 stays unopened during development. If a November winner exists, the base model(s),
ensemble weights, calibration, threshold, historical state, feature schema, and hashes are frozen,
and a separate one-time command may open December. No retraining, recalibration, threshold change,
historical-state update, or candidate switch is permitted. December is retrospective qualification,
not the genuine final test.
