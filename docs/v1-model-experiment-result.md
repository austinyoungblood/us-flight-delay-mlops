# Governed v1 CatBoost development result

## Outcome

The governed v1 development execution completed with a **governed stop**. Exactly six CatBoost
finalists reached November operating-point evaluation, but none had a threshold that simultaneously
satisfied all three precommitted requirements:

- recall >= 0.60
- precision >= 0.30
- predicted-positive rate <= 0.50

No November winner was selected or frozen. December qualification was not authorized or opened, and
Registry `production:v0` remains unchanged.

The machine-readable companion to this report is
[`experiments/v1/development_result.json`](../experiments/v1/development_result.json).

## Governed identity and execution

| Field | Value |
| --- | --- |
| Protocol | `us-flight-delay-v1-catboost-rolling-origin-v1` |
| Protocol SHA-256 | `a6b1de9de550d1bd94eae0e56f8d88d65801ec488b6c539fc64afbafa4ccfffb` |
| Implementation Git SHA | `1923658881a0c5cbc9c7c03595671a6aced71dc6` |
| Started | `2026-08-14T18:51:36.359832+00:00` |
| Completed | `2026-08-14T19:38:11.625011+00:00` |
| Duration | 46 minutes 35.265 seconds |
| Marker status | `complete` |
| Decision | `governed_stop` |
| Historical test accessed | `false` |

The durable stop report states: “No finalist passed every mandatory November gate. Production
remains v0.” No November winner lock exists.

## Finalist threshold evidence

Every finalist has status `no_eligible_threshold`. The table reports each finalist's best
unrestricted-F1 operating point and two constrained near-miss measurements. Values are rounded to six
decimal places for display; full precision is retained in the machine-readable result.

| Finalist | Best F1 | Precision | Recall | PPR | Threshold | Max recall at precision >= .30 and PPR <= .50 | Max precision at recall >= .60 and PPR <= .50 | W&B run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CB2-none | 0.377396 | 0.269942 | 0.626970 | 0.459252 | 0.207177 | 0.444590 | 0.273058 | [a62xjekg](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/a62xjekg) |
| CB2-sigmoid | 0.377396 | 0.269942 | 0.626970 | 0.459252 | 0.201492 | 0.444590 | 0.273058 | [i9hlnog9](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/i9hlnog9) |
| CB2-isotonic | 0.376919 | 0.269140 | 0.628676 | 0.461874 | 0.219837 | 0.430541 | 0.269140 | [my75aegk](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/my75aegk) |
| CB4-none | 0.378981 | 0.273585 | 0.616465 | 0.445544 | 0.213304 | 0.455882 | 0.276101 | [l1ai1u2k](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/l1ai1u2k) |
| CB4-sigmoid | 0.378981 | 0.273585 | 0.616465 | 0.445544 | 0.209085 | 0.455882 | 0.276101 | [mskckd0f](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/mskckd0f) |
| CB4-isotonic | 0.378406 | 0.270079 | 0.631828 | 0.462575 | 0.231076 | 0.451812 | 0.274958 | [4twrtfg5](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/4twrtfg5) |

## Interpretation

The predicted-positive-rate ceiling was **not** the principal blocker. At the strongest
unrestricted CB4 operating point, recall (0.616465) and PPR (0.445544) satisfied their constraints,
but precision was only 0.273585. No threshold simultaneously achieved precision >= 0.30 and recall
>= 0.60.

For CB4, requiring precision >= 0.30 reduced maximum recall to approximately 0.456. Requiring recall
>= 0.60 limited maximum precision to approximately 0.276. This pattern indicates insufficient class
ranking and separation in the high-recall operating region rather than merely a poor threshold
choice.

Sigmoid calibration preserved the same precision-recall frontier as the uncalibrated model; only
the numeric threshold changed. Isotonic calibration changed the available cutoffs slightly but did
not create an eligible operating point.

Threshold eligibility short-circuited the finalist workflow before downstream November gates. This
report therefore makes **no pass/fail claim** about November Brier score, log loss, ECE, probability
calibration, inference latency, or serialized bundle size.

## Period-matched incumbent comparison

The following descriptive comparison uses the historical frozen R3 November operating point and the
CB4-none best unrestricted point for the same November selection period.

| Metric | R3 | CB4-none | CB4 - R3 |
| --- | ---: | ---: | ---: |
| Precision | 0.250173 | 0.273585 | +2.34 percentage points |
| Recall | 0.711266 | 0.616465 | -9.48 percentage points |
| F1 | 0.370153 | 0.378981 | +0.88 percentage points |
| Predicted-positive rate | 0.562167 | 0.445544 | -11.66 percentage points |

CatBoost produced a more selective operating point with higher precision and slightly higher F1,
but lower recall. It did not satisfy the precommitted precision/recall requirements and was therefore
ineligible for promotion. This descriptive comparison is not evidence of better probability
calibration or better final-test performance.

## Governance boundaries

- The development execution stopped before December by design; December qualification was not
  authorized, opened, or evaluated.
- The consumed January-May 2026 historical test remained untouched by the governed v1 development
  execution.
- No November winner model or winner lock was created.
- No v1 model artifact or Registry version was created, and no Registry alias was changed.
- No deployment or AWS resource was changed.
- Registry `production:v0` remains the deployed model identity.

The ignored local execution marker, decision, and stop report are represented by exact path, byte
size, and SHA-256 records in the sanitized machine-readable result. Raw threshold tables, temporary
candidate bundles, model files, environment values, and credentials are intentionally excluded.
