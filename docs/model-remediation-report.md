# Brief 04 model-remediation stop report

## Outcome

No calibrated finalist passed every mandatory November gate. Brief 04 stopped before qualification
lock creation or December access. Route statistics, the release bundle, W&B model artifact, Registry
aliases, and final-test evidence were not created.

All twelve runs used clean Git commit `62883dfe91f7dc0e1803397a684ef06077787344` and exact dataset
artifact `austin-youngblood-university-of-denver/us-flight-delay-mlops/flight-delay-bts-sampled:v0`,
digest `2ecdb5a6a60b23ed1ee1d603fb976516`.

## Calibration-metric audit

Independent fixtures cover perfect calibration, deliberate miscalibration, ECE differing from the
global mean gap, empty bins, repeated probabilities, and deterministic quantile boundaries.
Recalculation of Brief 03 showed that its equality was legitimate because every bin error had the
same sign:

| Historical model | Mean gap | ECE10 equal-width | ECE15 equal-frequency | MCE15 |
| --- | ---: | ---: | ---: | ---: |
| Calibrated A | 0.030811 | 0.030933 | 0.030811 | 0.045262 |
| Candidate B | 0.106883 | 0.106883 | 0.106883 | 0.131110 |

## Six-base rolling-origin search

Each configuration completed all July–October folds without a convergence warning. Ranking used
mean AP, mean ROC-AUC, AP standard deviation, mean log loss, then lexical ID.

| Rank | ID | Mean AP | Mean ROC-AUC | AP std | Mean log loss | Mean p95 ms | W&B run |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | R3 | 0.342282 | 0.667879 | 0.075121 | 0.516165 | 4.249 | [yow2rxcy](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/yow2rxcy) |
| 2 | R4 | 0.336695 | 0.662797 | 0.075674 | 0.509225 | 3.611 | [t92pufga](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/t92pufga) |
| 3 | R1 | 0.336126 | 0.661933 | 0.075463 | 0.504550 | 3.657 | [9qg3fhei](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/9qg3fhei) |
| 4 | R5 | 0.336058 | 0.662261 | 0.074992 | 0.509270 | 3.542 | [xfybjrdw](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/xfybjrdw) |
| 5 | R2 | 0.335468 | 0.661699 | 0.076055 | 0.507973 | 3.580 | [d8d11wxy](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/d8d11wxy) |
| 6 | R0 | 0.332807 | 0.661331 | 0.073597 | 0.736784 | 3.645 | [cl450m8s](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/cl450m8s) |

Authorized bases were R3, R4, and the required R0 control.

## November finalist comparison

Calibration used November 1–15; selection and thresholds used November 16–30. The period-specific
prior had prevalence `0.197731`, Brier `0.158633`, and log loss `0.497241`.

| Finalist | AP | ROC-AUC | Brier | Log loss | Gap | ECE15 | Recall | F1 | Failed gates | W&B run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| R3 sigmoid | 0.282388 | 0.628118 | 0.153686 | 0.481410 | 0.008805 | 0.011191 | 0.711266 | 0.370153 | AP, F1 | [4ejk3ir2](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/4ejk3ir2) |
| R3 isotonic | 0.277117 | 0.627431 | 0.153814 | 0.481827 | 0.008639 | 0.013341 | 0.716649 | 0.369483 | AP, F1 | [bcqar6gu](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/bcqar6gu) |
| R4 sigmoid | 0.279443 | 0.626100 | 0.153939 | 0.482028 | 0.010366 | 0.011664 | 0.741597 | 0.368488 | AP, F1 | [a9uy3mzw](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/a9uy3mzw) |
| R4 isotonic | 0.273528 | 0.625174 | 0.154003 | 0.482320 | 0.010252 | 0.013015 | 0.744748 | 0.368037 | AP, F1 | [1gpyb2bg](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/1gpyb2bg) |
| R0 sigmoid | 0.279417 | 0.626754 | 0.153917 | 0.481918 | 0.011803 | 0.012647 | 0.706801 | 0.369369 | AP, F1 | [o4s9acza](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/o4s9acza) |
| R0 isotonic | 0.274682 | 0.626307 | 0.153957 | 0.482106 | 0.011652 | 0.012321 | 0.701812 | 0.368926 | AP, F1 | [vwqg69dl](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/vwqg69dl) |

Every finalist passed proper-scoring, calibration, recall, latency, estimated size, lineage, schema,
leakage, convergence, and serialization gates. Every finalist failed AP `>= 0.320719` and F1
`>= 0.41`; therefore no winner or threshold was frozen for December.

## Release state

- Qualification lock and December marker/report: absent
- Route asset and immutable release bundle: absent
- W&B release model artifact: absent
- Registry `staging`: absent
- Final-test marker/report/run: absent; test remains sealed
- Registry `production`: absent
- Post-selection retuning: none

The W&B audit found exactly six base and six finalist runs, all finished with the exact dataset and
Git SHA, no December/qualification/test metrics, and no prematurely logged model artifact.
