# Brief 03 model-selection stop report

## Outcome

No candidate passed every predeclared validation gate. The workflow stopped before building the
display-only route asset, freezing a release bundle, creating W&B Registry aliases, or opening the
sealed January–May 2026 test split. No release candidate was selected.

All successful runs used clean Git commit `e5f14459e5242e434beed9a7e9771e264cb60129` and exact dataset
artifact `austin-youngblood-university-of-denver/us-flight-delay-mlops/flight-delay-bts-sampled:v0`,
digest `2ecdb5a6a60b23ed1ee1d603fb976516`.

## Partitions

| Purpose | Dates | Rows | Prevalence |
| --- | --- | ---: | ---: |
| Base fit | Jan–Aug 2025 | 600,000 | 0.227640 |
| Candidate B tuning | Sep 2025 | 75,000 | 0.166320 |
| Final-candidate refit | Jan–Sep 2025 | 675,000 | 0.220827 |
| Sigmoid calibration | Oct 2025 | 75,000 | 0.203213 |
| Validation and threshold | Nov–Dec 2025 | 150,000 | 0.236660 |
| Sealed final test | Jan–May 2026 | not accessed | not accessed |

## Candidate B bounded tuning

September selection ranked average precision, then ROC-AUC, latency, and serialized size. Thresholds
were not tuned on September. Variant 1 (`alpha=1e-5`, `class_weight=None`) won.

| Variant | Alpha | Class weight | AP | ROC-AUC | p95 ms | Bytes | W&B run |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 1e-5 | None | 0.261734 | 0.657839 | 7.513 | 134,064 | [5lyv9r6l](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/5lyv9r6l) |
| 2 | 1e-5 | balanced | 0.256890 | 0.652492 | 7.578 | 134,080 | [guoxf6ua](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/guoxf6ua) |
| 3 | 1e-4 | None | 0.258837 | 0.655285 | 7.875 | 134,064 | [pifmb78e](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/pifmb78e) |
| 4 | 1e-4 | balanced | 0.260941 | 0.657684 | 7.854 | 134,080 | [a8ayk3et](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/a8ayk3et) |
| 5 | 1e-3 | None | 0.259028 | 0.652306 | 7.698 | 134,064 | [ah4heyn5](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/ah4heyn5) |
| 6 | 1e-3 | balanced | 0.259679 | 0.654583 | 7.750 | 134,080 | [39135lx5](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/39135lx5) |

## Validation comparison

Thresholds maximize F1 subject to recall at least 0.60, with the declared deterministic tie-breaks.

| Candidate | AP | ROC-AUC | Brier | Log loss | Mean-probability gap | ECE | Threshold | Recall | F1 | p95 ms | Bundle bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated A | 0.317313 | 0.618852 | 0.176132 | 0.534790 | 0.030811 | 0.030811 | 0.163192 | 0.771853 | 0.411807 | 4.196 | 25,163 |
| Candidate B | 0.304081 | 0.602779 | 0.188077 | 0.581644 | 0.106883 | 0.106883 | 0.096774 | 0.761937 | 0.404338 | 8.741 | 143,886 |

Calibrated A passed lineage, leakage, ROC-AUC, Brier, log loss, recall, F1, latency, and size. It
failed AP (`0.317313 < 0.320719`), ECE (`0.030811 > 0.03`), and mean-probability gap
(`0.030811 > 0.03`). Candidate B passed lineage, leakage, recall, latency, and size; it failed AP,
ROC-AUC, Brier, log loss, ECE, mean-probability gap, and F1.

The calibrated A run is [ohxuvitz](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/ohxuvitz),
with source artifact
[`flight-delay-candidate-a-calibrated:v0`](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/model/flight-delay-candidate-a-calibrated/v0),
digest `416eaeb620f655ebc088de2ba84d943a`. Candidate B is
[utekaf6m](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/utekaf6m),
with source artifact
[`flight-delay-candidate-b:v0`](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/model/flight-delay-candidate-b/v0),
digest `07aa68face758d70d448fe3e563ae962`.

## Release state

- Selected winner: none
- Frozen bundle and selection lock: not created
- W&B Registry collection/version: not created
- `staging`: absent
- Final-test run/report/marker: absent; test remains sealed
- `production`: absent
- Retuning after validation: none

An initial diagnostic run, `4jta5up5`, stopped before fitting when it found that sampled artifact rows
were not pre-sorted within a month. The partition implementation was corrected to perform a stable
date sort, covered by a regression test, and committed before the successful bounded runs.

## Final validation

- Python `3.11.15`
- Ruff: passed
- Formatting: 42 files already formatted
- Tests: 69 passed
- Branch coverage: 82.96% (80% minimum)
- W&B audit: all nine pre-test runs finished, all used exact dataset `v0` and digest, and none logged
  a `test/` or `final_test/` metric
- `flight-delay-api:brief03`: built, image `ef6327cbd2cb...`
- `flight-delay-user-ui:brief03`: built, image `9434034b9a20...`
- `flight-delay-monitor-ui:brief03`: built, image `b95bf5db1864...`
