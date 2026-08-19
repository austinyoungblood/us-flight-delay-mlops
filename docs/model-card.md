# R3 sigmoid model card — governed `production:v0` release

## Intended use

Estimate, before departure, whether a scheduled U.S. domestic flight will arrive at least 15
minutes late. The frozen release is served as W&B Registry `production:v0` for a controlled
academic demonstration. Its metadata reports `deployment_purpose=academic_demo` and
`internal_production_gate_passed=false`; the alias is not approval for real-world operational use.

## Model and data

R3 is an averaged SGD logistic classifier using the approved schedule-only feature schema plus
cyclical schedule/month transforms. It was fitted on January-October 2025 and sigmoid-calibrated on
November 1-15, 2025. Its threshold `0.1840285229739868` was locked from November 16-30 development
evidence. Dataset and model lineage are frozen in `release/selection_lock.json`.

The 2025 route reliability asset is display-only and never enters the model feature matrix.

## Evaluation and status

The one-time January-May 2026 final test preserved useful discrimination (ROC-AUC `0.623520`, AP
lift `1.384332`) and passed latency, size, lineage, leakage, schema, serialization, load, and
inference-contract gates. It failed calibration and proper-scoring gates: Brier Skill Score
`-0.013549`, log loss `0.520272` versus prior `0.517856`, probability gap `0.079739`, and ECE15
`0.079739`.

At the one-time final-test decision checkpoint, the model retained only `staging` because it failed
the internal production-quality gate. A later governed deployment assigned the same immutable `v0`
release the `production` alias for the academic demonstration without changing the model, threshold,
bundle, test decision, or failed gate. See [`final-test-report.md`](final-test-report.md) for the
historical decision and [`release/release_decision.json`](../release/release_decision.json) for the
current frozen serving identity.

## Limitations

- The model is sensitive to temporal probability drift.
- The low threshold produces high recall (`0.893765`) and a high predicted-positive rate
  (`0.815821`), with precision `0.233309`.
- It provides risk estimates, not live flight status or guarantees.
- The final-test result cannot be used for post-test model or threshold changes.
