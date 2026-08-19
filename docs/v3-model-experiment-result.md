# Governed v3 seasonal/temporal development result

## Outcome

The governed v3 lifecycle completed through an explicitly adopted recovery with a
`governed_stop`. All 15 November finalists returned `no_eligible_threshold`: none had a threshold
that simultaneously satisfied the three precommitted eligibility requirements:

- precision >= 0.30
- recall >= 0.60
- predicted-positive rate (PPR) <= 0.50

Threshold eligibility short-circuited evaluation, so the downstream November gates were **not
evaluated** and this report makes no pass/fail claim about them. No winner, winner lock, or winner
model was created. December qualification did not occur, and Registry `production:v0` remains the
deployed incumbent without modification.

The compact machine-readable companion is
[`experiments/v3/development_result.json`](../experiments/v3/development_result.json).

## Governed identity and recovery execution

| Field | Value |
| --- | --- |
| Protocol | `us-flight-delay-v3-seasonal-temporal-generalization-v1` |
| Protocol SHA-256 | `061be599fd84a4ddbf06229c300fe4670272d176b22899f1515332923376ecff` |
| Original implementation Git SHA | `3dea562f06365df166c89af6e851a817a2db00fc` |
| Recovery implementation Git SHA | `3b8ccbbe03f6973f10df15c43f6a6d4367dcb483` |
| Exact-selector corrective commit | `d5cf5da6e01787aca7838265e5bfd28818f37d5d` |
| Recovery ID | `v3-threshold-recovery-20260818-01` |
| Original marker SHA-256 | `90ae06f5bc81a3b86393d077866a2a2d65e3478d5920802dc4bee285a7fe9c1d` |
| Source tracking evidence SHA-256 | `84811e6a32227da3f99214142c584e2dba38aa8b13754c36cae14ae5dad6b9fa` |
| Termination record SHA-256 | `c84c0d4752e30284197bf0674607c629ec34690460c8e594901d64091f76b588` |
| Authorization whole-file SHA-256 | `19d1e161f8f83144e5bd4e8c1d39a35cfd58c8b3176273adbe332cd18bab4949` |
| Completed recovery marker SHA-256 | `0a35d12ba2928ec592ed29736def29e03900726072f4d312b9241ac794941180` |
| Recovery exit status | `0` |
| Adopted decision | `governed_stop` |

The original execution completed screening, authoritative CPU confirmation, and both full refits,
then encountered a threshold-sweep performance defect during its first November finalist. The
operator froze the available evidence and recorded termination before authorizing a bounded
recovery. That partial finalist was not used as scientific selection evidence. The original marker
remains byte-for-byte preserved with `status=started`, accurately recording that the original
process itself never completed.

The correction replaced the slow exact selector with a mathematically equivalent `O(N log N)`
implementation. Candidate definitions, data periods, ranking, eligibility, downstream gates, and
advancement rules did not change; this was an execution-performance correction, not a scientific
protocol change.

## Reconstructed advancement

Read-only source tracking evidence reconstructed the completed original screening and CPU
confirmation outcomes. The recovery recomputed advancement with the frozen ranking rules rather
than hardcoding candidate choices. It did not repeat screening or CPU confirmation and recovered
the following two authoritative advanced bases:

| Family | Advanced candidate | Weight policy |
| --- | --- | --- |
| LightGBM | `LGBM12-UNIFORM` | `UNIFORM` |
| CatBoost | `CB04-UNIFORM` | `UNIFORM` |

The recovery rebuilt and verified the governed development lineage, reproducing the November
historical-state SHA-256
`08d98a0bad462091897e91e8dad16d8882ea49a5a685abc5b08e2f1bed3f31b4`.

## Authoritative two-base refit

Only the two full refits that could not be recovered as reusable model objects were repeated. Both
used all 12,192,141 eligible February 2024–October 2025 full-refit rows and the frozen `UNIFORM`
weight policy.

| Family | Candidate | Weight policy | Full-refit rows | Runtime (seconds) | Historical-state SHA-256 |
| --- | --- | --- | ---: | ---: | --- |
| LightGBM | `LGBM12-UNIFORM` | `UNIFORM` | 12,192,141 | 255.347 | `08d98a0bad462091897e91e8dad16d8882ea49a5a685abc5b08e2f1bed3f31b4` |
| CatBoost | `CB04-UNIFORM` | `UNIFORM` | 12,192,141 | 3,617.745 | `08d98a0bad462091897e91e8dad16d8882ea49a5a685abc5b08e2f1bed3f31b4` |

## November finalist result

The recovery reevaluated all 15 finalists from scratch using the exact optimized threshold sweep;
the incomplete original finalist was ignored. Every finalist stopped at threshold eligibility, and
zero finalists reached downstream gate evaluation.

| Finalist | Status | W&B run |
| --- | --- | --- |
| `LGBM12-UNIFORM-none` | `no_eligible_threshold` | [22tmyhb2](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/22tmyhb2) |
| `LGBM12-UNIFORM-sigmoid` | `no_eligible_threshold` | [4s38i7qs](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/4s38i7qs) |
| `LGBM12-UNIFORM-isotonic` | `no_eligible_threshold` | [7t0qfyzx](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/7t0qfyzx) |
| `CB04-UNIFORM-none` | `no_eligible_threshold` | [hp8ujo13](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/hp8ujo13) |
| `CB04-UNIFORM-sigmoid` | `no_eligible_threshold` | [nil3cr08](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/nil3cr08) |
| `CB04-UNIFORM-isotonic` | `no_eligible_threshold` | [gehktinj](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/gehktinj) |
| `ENS25-none` | `no_eligible_threshold` | [k9eue97s](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/k9eue97s) |
| `ENS25-sigmoid` | `no_eligible_threshold` | [amdrvqf4](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/amdrvqf4) |
| `ENS25-isotonic` | `no_eligible_threshold` | [7irechyf](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/7irechyf) |
| `ENS50-none` | `no_eligible_threshold` | [9su8cyup](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/9su8cyup) |
| `ENS50-sigmoid` | `no_eligible_threshold` | [0dwcljdd](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/0dwcljdd) |
| `ENS50-isotonic` | `no_eligible_threshold` | [pidoavcc](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/pidoavcc) |
| `ENS75-none` | `no_eligible_threshold` | [pm8er060](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/pm8er060) |
| `ENS75-sigmoid` | `no_eligible_threshold` | [81rvr08s](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/81rvr08s) |
| `ENS75-isotonic` | `no_eligible_threshold` | [tjlrtc1l](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/runs/tjlrtc1l) |

## Operating-frontier near misses

The strongest high-recall frontier came from `ENS25-none` and `ENS25-sigmoid`. Values below are
rounded to six decimal places for display.

| Operating-region example | Precision | Recall | F1 | PPR |
| --- | ---: | ---: | ---: | ---: |
| ENS25 none/sigmoid: maximum precision at recall >= .60 and PPR <= .50 | 0.281099 | 0.600028 | 0.382844 | 0.425727 |
| ENS25 none/sigmoid: precision >= .30 and PPR <= .50 frontier | 0.300020 | 0.479384 | 0.369063 | 0.318677 |
| CB04-UNIFORM-none: best F1 among the listed operating-region examples | 0.273576 | 0.646500 | 0.384461 | 0.471313 |

At the high-recall point, precision missed the locked `0.30` requirement by `0.018901`, or 1.8901
percentage points. Moving to precision `0.300020` reduced recall to `0.479384`, well below its locked
`0.60` requirement. Raw, sigmoid, and isotonic candidates plus three ensemble mixtures all failed
to produce a simultaneous `P >= .30 / R >= .60 / PPR <= .50` point. The evidence therefore
identifies a precision/recall frontier limitation, not a justification to relax the threshold,
eligibility rules, or calibration policy.

## Interpretation

V3 broadened the modeling challenge substantially: it used 2024–2025 history, seasonal and holiday
features, leakage-safe same-calendar-month historical propensity, `UNIFORM` versus
`EXPONENTIAL_120D` weighting, temporal-robustness ranking across LightGBM and CatBoost, three
calibration treatments, and three ensemble mixtures. That additional sophistication did not earn
promotion. Late-November evidence still could not satisfy the precommitted operating constraints.

This is an unsuccessful challenger, not a failed serving system. Release governance worked as
designed: it required simultaneous operational eligibility, stopped before qualification when that
evidence was absent, and retained the already deployed v0 identity rather than promoting a more
complex but ineligible model.

## Recovery engineering and incident handling

The recovery lifecycle preserved scientific and operational accountability:

- The exact-selector performance defect was identified during the first original November
  finalist, and the operator captured a durable termination record before recovery.
- Existing source evidence was frozen. The original marker, source log identity, tracking evidence,
  termination record, and authorization are independently hash-addressed.
- The selector was corrected with a mathematically equivalent `O(N log N)` implementation and
  equivalence/benchmark evidence; no protocol, candidate, ranking, or gate changed.
- Completed screening and CPU confirmation were reconstructed from immutable W&B evidence rather
  than rerun. Frozen ranking rules were reapplied to reconstruct advancement.
- Only the two unrecoverable full refits were repeated. All 15 November finalists were then
  evaluated from scratch, excluding the original partial finalist.
- The recovery decision and adopted canonical decision are byte-identical, with SHA-256
  `3ee90070d06026a8502a417eca6e87d734d21db71dec438d6bfdec55dca61cc8`.

The canonical adoption record has payload SHA-256
`6b74788f6d6be97b900afbba7f765280d5a203f850bb3586a9ce750b8dc5b937` and explicitly records
`original_marker_rewritten=false`, `production_remains=v0`, `december_opened=false`, and
`historical_test_accessed=false`.

## Governance and evidence boundaries

- For the actual governed v3 execution and recovery, December was not opened or evaluated for model
  selection. No December qualification occurred because v3 produced no winner.
- “December 2025 was transiently materialized once during pre-run implementation testing, but no
  model was scored against it, no threshold/calibration/model-selection decision used it, and the
  test was replaced with synthetic data before governed v3 execution.”
- The consumed January–May 2026 historical final test remained untouched.
- Registry `production:v0`, its threshold, deployment, aliases, and artifacts remained untouched.
- The original `status=started` marker remains byte-for-byte unchanged; recovery adoption records
  the governed stop without rewriting history.
- No canonical `winner_lock.json` or `winner.joblib` exists.
- The authoritative canonical decision is 510,702,885 bytes because it retains every finalist's
  full unique-score threshold table. It and the other raw recovery, model, historical-state,
  threshold, and W&B-local artifacts intentionally remain ignored.

This documentation work uses only the compact hash-addressed representation committed to Git; it
does not rerun models, access governed data, contact cloud services, mutate W&B, or alter release
state.
