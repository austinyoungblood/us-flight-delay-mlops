# Governed release policy

This policy was committed before any January-May 2026 final-test access. It changes only the
self-imposed release decision inherited from model remediation; it does not revise or erase the
earlier model-selection or remediation results.

## Governance decision

The earlier absolute average-precision gate of `0.320719` and F1 gate of `0.41` remain historical
evidence but are retired from the release decision. A no-skill scorer's average precision equals the
positive prevalence, so an absolute value measured in one time period is not portable to another
period with a different prevalence. F1 also depends on the selected decision threshold, and neither
absolute cutoff is a course-rubric requirement.

No additional model, feature, hyperparameter, calibration, training-window, or threshold search is
authorized. R3 with sigmoid calibration is selected from already-observed remediation evidence because
it had the highest November 16-30 average precision (`0.2823880567429311`) and ROC-AUC
(`0.6281178113133866`) among the six calibrated finalists. Its exact previously selected threshold
must be reconstructed from the existing remediation evidence and must not be recomputed.

Reconstruction metrics must match the remediation evidence within an absolute floating-point tolerance
of `1e-9`. This tolerance accommodates sub-nanounit numerical drift between the original Python
3.11.15 run and the Python 3.11.14 release environment; it is orders of magnitude below the
precision of every release gate and does not permit a model or threshold change.

## Final-test production gate

The immutable staged candidate receives `production` only if every gate below passes on the single
January-May 2026 final-test evaluation:

- ROC-AUC is at least `0.58`.
- Average precision divided by contemporaneous positive prevalence is at least `1.20`.
- Brier Skill Score against the contemporaneous constant-prior predictor is greater than `0`.
- Model log loss is lower than constant-prior log loss.
- Absolute mean predicted-probability minus prevalence gap is at most `0.05`.
- Fifteen-bin equal-frequency ECE is at most `0.05`.
- p95 single-request inference latency is below `25 ms`.
- Bundle size is below `10 MiB` (`10,485,760` bytes).
- Exact lineage, feature schema, leakage guard, serialization, model load, and inference contract all
  pass.

Accuracy, precision, recall, F1, confusion counts, and predicted-positive rate are reported but do
not control the release decision. If any production gate fails, the exact immutable candidate
remains available under `staging`; modeling does not restart and no post-test tuning is permitted.

## Test-access controls

The final-test evaluator must refuse dirty or mismatched Git state, altered policy or bundle hashes,
incorrect dataset lineage, an unverified `staging` alias, or a pre-existing one-time marker. It must
create a durable marker before reading final-test labels and may execute only once.
