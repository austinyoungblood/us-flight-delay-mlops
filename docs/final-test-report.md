# One-time final-test report

> **Historical decision record:** This report preserves the state at the one-time final-test
> checkpoint. The unchanged Registry `v0` was later assigned the `production` alias for an
> `academic_demo` deployment while `internal_production_gate_passed=false` remained unchanged. The
> later alias did not reopen the test, retrain or recalibrate the model, change its threshold, or
> reverse any gate below.

## Decision

The immutable R3 sigmoid candidate did not pass every precommitted final-test production gate.
At this historical checkpoint, Registry version
`wandb-registry-Model/us-flight-arrival-delay-15m:v0` therefore retained `staging` and did not receive
`production`. No post-test retraining or threshold adjustment was authorized.

## Locked candidate and lineage

- Candidate: R3 SGD, sigmoid calibration
- Configuration: `loss=log_loss`, `penalty=l2`, `alpha=1e-5`, `average=true`, no class weighting,
  `max_iter=1000`, `tol=0.001`, seed 42
- Base fit: January-October 2025
- Calibration: November 1-15, 2025
- Threshold: `0.1840285229739868`, selected once during remediation and not recomputed
- Dataset: `flight-delay-bts-sampled:v0`
- Dataset digest: `2ecdb5a6a60b23ed1ee1d603fb976516`
- Selection-lock SHA-256: `a730a25c34a9f259b3ca02eb92c4ad44c1e75f50fd52ce270a940e4a60142340`
- Aggregate bundle digest: `2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572`
- Registry/source digest: `865ddd18f6debd44f24a79fc71739f2a`

The Registry `staging` alias was clean-downloaded and every locked file hash was verified before
the evaluator opened the test split.

## One-time evaluation evidence

- Period: January 1-May 31, 2026
- Rows: 375,000
- W&B run: [w4te9tla](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/w4te9tla)
- Marker status: `complete`
- Marker evaluation count: `1`
- Marker SHA-256: `1251e769b5a6d749d96d6c2c5754983a01d622ed647d8289faf5c4816014f6ee`
- Machine-result SHA-256: `dc1c97db7babc74ed0a50e0b9fd808314d7330eadc836508db9a8bcb997d8886`
- Release-decision SHA-256: `edb1e439eec366645c199a1eb81e1f401f3ed06eb08863485b4bb373e245a07d`

## Metrics and gates

| Gate | Requirement | Observed | Result |
| --- | --- | ---: | --- |
| ROC-AUC | `>= 0.58` | 0.623520 | Pass |
| AP lift over prevalence | `>= 1.20` | 1.384332 | Pass |
| Brier Skill Score | `> 0` | -0.013549 | **Fail** |
| Log loss versus prior | `< 0.517856` | 0.520272 | **Fail** |
| Probability/prevalence gap | `<= 0.05` | 0.079739 | **Fail** |
| Equal-frequency ECE15 | `<= 0.05` | 0.079739 | **Fail** |
| p95 inference latency | `< 25 ms` | 5.269653 ms | Pass |
| Bundle size | `< 10 MiB` | 554,964 bytes | Pass |
| Lineage/schema/leakage | all pass | all pass | Pass |
| Serialization/load/inference | all pass | all pass | Pass |

Contemporaneous positive prevalence was `0.212963`, so the no-skill AP baseline was `0.212963`.
The constant-prior Brier score was `0.167610`; model Brier was `0.169880`.

Reported non-gating threshold metrics:

- Accuracy: `0.351893`
- Precision: `0.233309`
- Recall: `0.893765`
- F1: `0.370026`
- Predicted-positive rate: `0.815821`
- Confusion matrix: TN `60,583`, FP `234,556`, FN `8,484`, TP `71,377`

## Governance outcome

The model retained useful ranking lift and high recall, but its probabilities were systematically
high relative to 2026 prevalence and did not beat the contemporaneous constant-prior predictor on
proper scores. At that checkpoint, the result was a governed `staging` release, not production
qualification. The final test is consumed and must never be used for retraining, recalibration, or
threshold selection.

## Subsequent serving state

The same immutable release is now served as `production:v0` solely for the controlled academic
demonstration. The current frozen identity remains registry digest
`865ddd18f6debd44f24a79fc71739f2a`, bundle SHA-256
`2677b7093d66637852705d33bca006c3b78d8119f4d7268801453aa18c22f572`, and threshold
`0.1840285229739868`, with the internal production gate still false. See the
[`release decision`](../release/release_decision.json) and
[`deployment manifest`](../deploy/deployment_manifest.json).

## Historical checkpoint validation

- Python 3.11.14
- Ruff check and formatting: pass; 48 files formatted
- Pytest: 91 passed
- Branch coverage: 80.29% with an enforced 80% minimum
- Marker rerun audit: refused before test access with `final-test marker already exists`
- W&B audit: exactly one finished final-test run with locked data and bundle lineage
- Registry audit at that time: `v0` had `latest` and `staging`; `production` was absent
- Historical API image: `sha256:532f260988b78ffbcaa039dabf8cf6734fd9caa99877bdb4c8ecfa5d7c210de8`
- Historical User UI image: `sha256:0fc962b14ee704da034498e0137a7b19475917808ad1804fac20c37ab203fb6f`
- Historical Monitor UI image: `sha256:3e50b103f21b27e32dfbc10864397eca0b566c60cc632b74caf170494e2c8dbf`
