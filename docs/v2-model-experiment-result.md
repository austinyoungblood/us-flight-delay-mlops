# Governed v2 dual-boosting development result

## Outcome

The governed v2 development execution completed with a **governed stop**. LightGBM and CatBoost
both improved rolling-origin ranking relative to v1, but none of the 12 late-November finalists had
a threshold that simultaneously satisfied the three precommitted eligibility requirements:

- recall >= 0.60
- precision >= 0.30
- predicted-positive rate <= 0.50

Every finalist returned `status=no_eligible_threshold`. Threshold eligibility therefore
short-circuited evaluation before the downstream November gates; this report makes **no pass/fail
claim** about those downstream gates. No winner or winner lock was created, December was not opened,
and Registry `production:v0` remains unchanged.

The compact machine-readable companion is
[`experiments/v2/development_result.json`](../experiments/v2/development_result.json).

## Governed identity and execution

| Field | Value |
| --- | --- |
| Protocol | `us-flight-delay-v2-historical-propensity-dual-boost-v1` |
| Protocol SHA-256 | `8e57b0f63656003c9981b3b5e44623e0b7c556f6e0c7222352ac38dd5119420a` |
| Implementation Git SHA | `6966562dcc2a7959f27e662e97cfeec8a4aa43a6` |
| Started | `2026-08-14T22:44:06.128373+00:00` |
| Completed | `2026-08-17T19:34:10.480915+00:00` |
| Duration | 68 hours 50 minutes 4.353 seconds |
| Marker status | `complete` |
| Decision | `governed_stop` |
| December opened | `false` |
| Historical test accessed | `false` |

The applied run used LightGBM `4.7.0` and CatBoost `1.2.10`. LightGBM screening was bounded CPU
work; CatBoost screening was sequential on GPU device `0`; and the authoritative confirmation and
all full refits were CPU-only, exactly as precommitted. The shared November historical-feature state
was frozen as of October 31, 2025.

## Authoritative CPU confirmation

The table reports the four-fold CPU-confirmation metrics used for advancement. “Mean precision” is
the mean of each fold's maximum precision at recall >= 0.60 and predicted-positive rate <= 0.50.
Values are rounded to six decimal places for display; full precision is retained in the JSON result.

| Candidate | Family | Mean precision | Worst fold | Mean AP | Mean ROC-AUC | W&B run |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LGBM12 | LightGBM | 0.336777 | 0.253025 | 0.362252 | 0.682114 | [v6bm3cy3](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/v6bm3cy3) |
| LGBM10 | LightGBM | 0.336261 | 0.250603 | 0.363461 | 0.682463 | [kwtdnq2q](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/kwtdnq2q) |
| LGBM01 | LightGBM | 0.335777 | 0.251410 | 0.361033 | 0.681393 | [efc7y4vz](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/efc7y4vz) |
| LGBM16 | LightGBM | 0.335748 | 0.251661 | 0.360358 | 0.679620 | [s2s69efw](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/s2s69efw) |
| CB07 | CatBoost | 0.338105 | 0.256052 | 0.363089 | 0.684059 | [5te9q8a8](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/5te9q8a8) |
| CB04 | CatBoost | 0.337715 | 0.255446 | 0.361206 | 0.683163 | [4450asos](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/4450asos) |
| CB10 | CatBoost | 0.337239 | 0.253785 | 0.361498 | 0.683265 | [1sk0ga0o](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/1sk0ga0o) |
| CB06 | CatBoost | 0.337183 | 0.253539 | 0.361870 | 0.683268 | [7f44prnu](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/7f44prnu) |

CPU ranking advanced `LGBM12`, `LGBM10`, `CB07`, and `CB04` to full February–October refit. Their
uncalibrated, sigmoid, and isotonic variants produced the exact 12 November finalists.

## November threshold evidence

The strongest November high-recall frontier came from CB04-none and CB04-sigmoid. Both variants
reached precision `0.278481` at recall `0.600840` and predicted-positive rate `0.426617`. Their
probability rankings were identical; sigmoid changed only the numeric threshold.

| CB04 variant / comparison | Precision | Recall | F1 | PPR |
| --- | ---: | ---: | ---: | ---: |
| none/sigmoid: maximum precision at recall >= .60 | 0.278481 | 0.600840 | 0.380572 | 0.426617 |
| none/sigmoid: best unrestricted F1 | 0.268702 | 0.670168 | 0.383601 | 0.493159 |
| none/sigmoid: maximum recall at precision >= .30 | 0.300073 | 0.486870 | 0.371301 | 0.320819 |
| isotonic: maximum precision at recall >= .60 | 0.273203 | 0.638130 | 0.382602 | 0.461848 |
| isotonic: maximum recall at precision >= .30 | 0.303568 | 0.475840 | 0.370666 | 0.309941 |

All 12 finalist runs are retained for traceability:

| Finalist | Status | W&B run |
| --- | --- | --- |
| LGBM12-none | `no_eligible_threshold` | [bt5q98e4](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/bt5q98e4) |
| LGBM12-sigmoid | `no_eligible_threshold` | [z2by3map](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/z2by3map) |
| LGBM12-isotonic | `no_eligible_threshold` | [zk84csat](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/zk84csat) |
| LGBM10-none | `no_eligible_threshold` | [savbk5yy](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/savbk5yy) |
| LGBM10-sigmoid | `no_eligible_threshold` | [hpmhxwq8](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/hpmhxwq8) |
| LGBM10-isotonic | `no_eligible_threshold` | [s4womnbv](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/s4womnbv) |
| CB07-none | `no_eligible_threshold` | [uz2ld1qp](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/uz2ld1qp) |
| CB07-sigmoid | `no_eligible_threshold` | [k2c63egv](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/k2c63egv) |
| CB07-isotonic | `no_eligible_threshold` | [b3insd4a](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/b3insd4a) |
| CB04-none | `no_eligible_threshold` | [192thf0w](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/192thf0w) |
| CB04-sigmoid | `no_eligible_threshold` | [hcj30i3s](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/hcj30i3s) |
| CB04-isotonic | `no_eligible_threshold` | [8i2dz386](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/8i2dz386) |

## Interpretation

V2 materially improved rolling-origin ranking compared with v1: CPU-confirmed candidates from both
families averaged roughly `0.336`–`0.338` precision at the high-recall operating frontier. That
improvement did not generalize to late-November selection. The best November high-recall precision,
`0.278481`, was only modestly above v1's approximately `0.276` ceiling and remained below the locked
`0.30` requirement.

LightGBM and CatBoost converged to similar November behavior: their best high-recall precisions were
approximately `0.276` and `0.278`, respectively. Sigmoid calibration preserved each base model's
ranking frontier, while isotonic calibration did not create an eligible operating point. The
evidence therefore points to temporal robustness and seasonality—not threshold relaxation or a
different probability calibration—as the next modeling problem.

Governance operated as designed: improvement on development folds was insufficient for promotion
without late-period eligibility, so the experiment stopped before December and the deployed
incumbent remained unchanged.

## Governance and evidence boundaries

- December qualification was not authorized, opened, or evaluated.
- The consumed January–May 2026 historical test was not read by v2.
- No November winner, winner lock, promotable model artifact, or Registry version was created.
- No Registry alias, deployment, AWS resource, v1 artifact, v2 protocol, or production threshold was
  changed by this evidence submission.
- Registry `production:v0` remains the deployed governed incumbent.

The ignored local `decision.json`, `execution_marker.json`, and historical-state artifact are
represented in the compact result by path, byte size, and SHA-256. The 55 MB decision file, the
historical-state payload, candidate bundles, model files, raw threshold tables, environment values,
and credentials are intentionally excluded from Git.
