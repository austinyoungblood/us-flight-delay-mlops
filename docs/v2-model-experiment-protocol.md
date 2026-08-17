# Governed v2 model-improvement protocol

Protocol-lock status: **precommitted before training**.
At protocol lock time, no v2 model had been fit and no v2 result existed. The contract below remains
frozen and unchanged.

Execution status: development later completed with a governed stop; see the
[v2 result report](v2-model-experiment-result.md). December 2025 remained unopened, and production
remains Registry `production:v0`.

The machine-readable source of truth is
[`configs/v2_experiment_protocol.yaml`](../configs/v2_experiment_protocol.yaml). Its byte-level
SHA256, exact candidate matrices, immutable v1 evidence hashes, and prerequisite lineage are sealed
in [`experiments/v2/protocol_lock.json`](../experiments/v2/protocol_lock.json).

## Motivation and immutable boundaries

V1 stopped because no CatBoost finalist simultaneously achieved recall at least 0.60, precision at
least 0.30, and predicted-positive rate at most 0.50. Its strongest high-recall precision was about
0.2761. V2 must improve ranking and separation sufficiently to cross the unchanged precision gate;
threshold relaxation is prohibited.

- V1 protocol, result, and report bytes are immutable and sealed in the v2 lock.
- Development may use January-November 2025 only. January supplies burn-in history; model rows begin
  February 1.
- December labels must not be read or preprocessed unless a November winner is later frozen and a
  separate qualification command is explicitly authorized.
- `data/processed/test.parquet` and every January-May 2026 label remain permanently prohibited.
- This protocol performs no AWS, Registry, deployment, W&B, or production action. Production stays
  `v0`.

## Leakage-safe historical feature contract

The existing 20 pre-departure schedule features are retained. Seventeen numeric features are added:

1. `prior_global_delay_rate`
2. `prior_carrier_delay_rate`
3. `prior_origin_delay_rate`
4. `prior_destination_delay_rate`
5. `prior_route_delay_rate`
6. `prior_carrier_route_delay_rate`
7. `prior_carrier_origin_delay_rate`
8. `prior_carrier_destination_delay_rate`
9. `prior_origin_departure_hour_delay_rate`
10. `prior_destination_arrival_hour_delay_rate`
11. `log_route_support`
12. `log_carrier_route_support`
13. `recent_global_delay_rate_3m`
14. `recent_carrier_delay_rate_3m`
15. `recent_origin_delay_rate_3m`
16. `recent_destination_delay_rate_3m`
17. `recent_route_delay_rate_3m`

Every model row in month M uses labels strictly before M. Empirical-Bayes rates use prior strength
50 and the historical global rate. Unseen keys fall back to that global rate, and supports use
`log1p(count)`. The trailing rates use only the three complete months immediately before M.

January is burn-in rather than a model month. July, August, September, and October evaluation state
ends June 30, July 31, August 31, and September 30 respectively. Both November halves use the same
October-31 state. A later December qualification must reuse that October-31 state without November
updates.

## Historical-state artifact and serving parity

The implementation produces a deterministic `flight-delay-historical-state-v1` artifact containing
the as-of date, global counts, full-history and trailing-window lookup tables, smoothing metadata,
and feature schema. Lookup entries are sorted lexically and serialized as canonical UTF-8 JSON with
sorted keys, compact separators, no NaN, and SHA256 identity.

Each candidate bundle must reference the exact state digest and schema digest. The same pure
single-row transformer used by serving is applied by the batch training transformer. Synthetic tests
must prove identical values and demonstrate that only fields available in the scheduled-flight API
request plus frozen state are required.

## Data use and temporal search

Search model rows are sampled deterministically without replacement, seed 42, up to 75,000 rows per
month from February-October. Historical lookup tables are always calculated from all eligible prior
month rows, never the model-row sample. Authoritative refits use every eligible February-October row.
Source, eligible, and model-row counts and hashes must be recorded.

| Fold | Fit model rows | Evaluation month | State cutoff |
| --- | --- | --- | --- |
| FOLD_1 | February-June | July | June 30 |
| FOLD_2 | February-July | August | July 31 |
| FOLD_3 | February-August | September | August 31 |
| FOLD_4 | February-September | October | September 30 |

Outer evaluation months cannot control early stopping.

## Exact LightGBM matrix

LightGBM is pinned to `4.7.0`, uses native categorical features, and screens on CPU with `n_jobs=20`.
All candidates also use binary objective, seed 42, `verbosity=-1`, deterministic column-wise mode,
and `subsample_freq=1`.

### Pre-training subsampling correction

This correction was made before any real v2 fit, W&B run, or result artifact existed. The original
protocol intentionally varied `subsample` between `0.8` and `1.0`, but LightGBM defaults
`subsample_freq` to `0`, which left that declared search dimension inactive. Setting the common
parameter `subsample_freq=1` activates the existing dimension: candidates with `subsample=0.8`
perform row bagging, while candidates with `subsample=1.0` remain effectively full-row fits. No
candidate identity row, CatBoost value, feature, period, gate, ranking rule, backend policy, or
advancement rule changed. No observed model result influenced this correction.

| ID | leaves | depth | rate | estimators | child | subsample | columns | L2 | L1 | cat smooth | cat L2 | class weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LGBM01 | 127 | 8 | .02 | 800 | 100 | 1.0 | 1.0 | 1 | 1 | 50 | 10 | 1.0 |
| LGBM02 | 31 | 12 | .05 | 500 | 200 | .8 | .8 | 10 | 0 | 10 | 20 | 1.5 |
| LGBM03 | 31 | -1 | .02 | 1200 | 50 | 1.0 | .8 | 10 | 0 | 10 | 20 | 1.0 |
| LGBM04 | 31 | -1 | .05 | 1200 | 50 | .8 | 1.0 | 1 | 0 | 10 | 10 | 1.5 |
| LGBM05 | 127 | -1 | .05 | 1200 | 50 | .8 | .8 | 10 | 1 | 50 | 20 | 1.5 |
| LGBM06 | 127 | 12 | .05 | 500 | 50 | .8 | 1.0 | 1 | 0 | 20 | 20 | 1.0 |
| LGBM07 | 127 | -1 | .02 | 500 | 200 | .8 | .8 | 10 | 1 | 10 | 10 | 1.5 |
| LGBM08 | 31 | 12 | .02 | 1200 | 200 | .8 | 1.0 | 1 | 1 | 20 | 20 | 1.5 |
| LGBM09 | 31 | 12 | .05 | 1200 | 200 | 1.0 | 1.0 | 10 | 0 | 50 | 10 | 1.25 |
| LGBM10 | 31 | 12 | .05 | 500 | 50 | .8 | .8 | 10 | 1 | 50 | 10 | 1.0 |
| LGBM11 | 31 | -1 | .05 | 500 | 50 | 1.0 | 1.0 | 10 | 1 | 10 | 20 | 1.5 |
| LGBM12 | 31 | -1 | .02 | 500 | 200 | .8 | 1.0 | 10 | 0 | 50 | 20 | 1.0 |
| LGBM13 | 63 | 12 | .02 | 500 | 50 | 1.0 | .8 | 1 | 0 | 10 | 10 | 1.5 |
| LGBM14 | 127 | -1 | .05 | 1200 | 200 | 1.0 | .8 | 1 | 0 | 10 | 20 | 1.5 |
| LGBM15 | 127 | 12 | .05 | 1200 | 200 | .8 | .8 | 1 | 1 | 10 | 10 | 1.0 |
| LGBM16 | 31 | -1 | .05 | 500 | 200 | 1.0 | .8 | 1 | 1 | 50 | 20 | 1.0 |

## Exact CatBoost matrix

CatBoost is pinned to `1.2.10` and uses native categorical features. All candidates use Logloss,
seed 42, `has_time=true`, no file writing, and no outer-fold early stopping.

| ID | depth | iterations | rate | L2 | random | bagging | borders | CTR | class weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CB01 | 6 | 500 | .05 | 12 | .5 | 3 | 128 | 1 | 1.0 |
| CB02 | 10 | 1200 | .02 | 3 | 2 | 0 | 254 | 2 | 1.5 |
| CB03 | 10 | 500 | .05 | 3 | .5 | 1 | 254 | 1 | 1.5 |
| CB04 | 6 | 800 | .02 | 12 | .5 | 0 | 254 | 2 | 1.0 |
| CB05 | 8 | 1200 | .05 | 12 | 2 | 0 | 128 | 1 | 1.5 |
| CB06 | 6 | 1200 | .03 | 3 | .5 | 3 | 128 | 2 | 1.5 |
| CB07 | 10 | 500 | .02 | 7 | 2 | 3 | 128 | 2 | 1.0 |
| CB08 | 6 | 1200 | .05 | 3 | 2 | 3 | 254 | 1 | 1.0 |
| CB09 | 10 | 1200 | .02 | 3 | .5 | 0 | 128 | 1 | 1.0 |
| CB10 | 6 | 500 | .05 | 12 | 2 | 3 | 254 | 2 | 1.5 |
| CB11 | 10 | 500 | .03 | 12 | 2 | 0 | 254 | 1 | 1.0 |
| CB12 | 8 | 1200 | .02 | 12 | .5 | 3 | 254 | 1 | 1.5 |

The matrices were selected before results using seed-42 greedy maximin distance over parameter
domain indices normalized to `[0, 1]`; seeded shuffle order resolves ties. Backend is execution
metadata and is excluded from candidate identity.

## GPU screening and authoritative CPU confirmation

CatBoost screening is sequential on GPU device `0`; concurrent GPU fits are prohibited. LightGBM
screening remains bounded CPU work and does not require a CUDA build. The top four candidates from
each family advance to the same four-fold confirmation on CPU. CatBoost changes only `task_type`
from GPU to CPU; hyperparameters stay identical. CPU confirmation metrics alone rank advancement.
The top two CPU-confirmed candidates per family refit on all February-October rows, on CPU only.

The primary fold metric is maximum precision among thresholds satisfying recall at least 0.60 and
positive rate at most 0.50; no qualifying threshold scores zero. Candidate ranking uses mean primary
metric, worst-fold primary metric, mean AP, mean ROC-AUC, mean log loss, mean Brier, primary-metric
standard deviation, then lexical ID.

## November finalists and gates

The four CPU-refit bases are frozen before `none`, `sigmoid`, and `isotonic` variants use November
1-15, producing exactly 12 finalists. November 16-30 selects an eligible threshold requiring recall
at least 0.60, precision at least 0.30, and positive rate at most 0.50, then maximizes F1 with the v1
tie-break semantics.

Every finalist must also pass F1 at least 0.38; positive Brier skill; Brier and log loss below their
contemporaneous priors; probability/prevalence gap and equal-frequency ECE15 at most 0.03; AP at
least `0.2923880567429311`; ROC-AUC at least `0.6331178113133866`; latency, bundle size, lineage,
schema, leakage, state integrity, training/serving parity, R3 reconstruction, serialization, runtime,
and prohibited-access gates. A v2 improvement claim is permitted only after all gates pass.

Zero passing finalists produces a governed stop and retains production `v0`. A winner freezes model,
calibration, threshold, historical state, schemas, hashes, and protocol/code lineage.

## December, tracking, and release

Development and qualification use separate, default-dry-run CLIs and v2-only markers. Applied
development is one-time and fail-closed. December can be opened once only after a frozen November
winner exists; it cannot refit, recalibrate, change threshold, update state, switch candidates, or
change gates.

Applied development uses W&B group `v2-<protocol-sha>-<implementation-sha>` and records hardware,
backends, CUDA visibility, package versions, feature-state digest, dataset lineage, and protocol/code
identity. Development cannot mutate the Registry. Neither modeling dependency may enter the API,
Traveler, or Monitor images serving v0.

The genuine final holdout remains the first complete untouched BTS month strictly after May 31,
2026 under the existing eligibility rule. It is not accessed by this protocol or its implementation.
