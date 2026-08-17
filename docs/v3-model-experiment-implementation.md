# Governed v3 implementation

Status: **implementation and synthetic tests only**. No v3 model has been fit, no v3 result exists,
December 2025 is undecoded, January–May 2026 is untouched, and production remains `production:v0`.

The frozen design is [`configs/v3_experiment_protocol.yaml`](../configs/v3_experiment_protocol.yaml),
SHA256 `061be599fd84a4ddbf06229c300fe4670272d176b22899f1515332923376ecff`, sealed by
[`experiments/v3/protocol_lock.json`](../experiments/v3/protocol_lock.json). See
[the protocol](v3-model-experiment-protocol.md) for the scientific design.

## Modules

| Module | Responsibility |
| --- | --- |
| `modeling/v3/protocol.py` | Byte-level protocol/lock validation, feature and identity constants |
| `modeling/v3/seasonal.py` | Schedule-only seasonal features, scalar and vectorized paths |
| `modeling/v3/features.py` | V3 historical state and the 48-feature transforms |
| `modeling/v3/weighting.py` | The two precommitted training-weight policies |
| `modeling/v3/models.py` | 8 identities, 8 native categoricals, weighted fits, calibration, ensembles |
| `modeling/v3/selection.py` | Fold metrics, the nine-level ranking, and the November gates |
| `modeling/v3/data.py` | Fail-closed split access and the search/refit/November matrices |
| `modeling/v3/workflow.py` | Screening → CPU confirmation → refit → 15 finalists |
| `modeling/v3/execution.py` | Preflight, runtime estimation, durable markers, December handoff |
| `data/prepare_v3.py` | Uncapped development preparation plus December-only qualification materialization |

## Design decisions worth knowing

**The v3 state composes the v2 state rather than reimplementing it.** `V3HistoricalState` holds a
real `flight_delay.modeling.v2.features.HistoricalState`, so all 37 retained v2 features are
produced by the exact v2 code path and cannot silently drift. V3 adds only the five
same-calendar-month tables and the six deterministic seasonal columns. V2 refusals are re-raised as
`V3FeatureError` so callers guard one error type.

**The prior-year invariant is enforced structurally, then asserted.** Every state cutoff is strictly
before the first day of the model-row month, so a same-calendar-month entry in the state can only
come from an earlier year. The state also carries a `same_calendar_month_max_year` ledger, and both
transform paths refuse a lookup whose calendar month already contains the model row's own year. The
November 2025 state built from history through 2025-10-31 reports max year 2024 for month 11.

**The batch transform is vectorized and proven equal to the serving path.** A per-row loop over the
~12.2 M-row refit matrix would have dominated the overnight budget, so lookups are resolved by
reindexing pandas frames built from the JSON-keyed tables. `test_v3_features` asserts the batch and
single-row paths agree to 1e-12, which is the training–serving parity requirement.

**The fold matrix carries November; the refit matrix does not.** FOLD_4 evaluates November 2025, so
the rolling-fold matrix must contain November rows — but every fold's fit window ends at or before
2025-11-01, so no fold ever fits on them. The authoritative refit matrix stays strictly
2024-02-01 → 2025-10-31.

**Weight policy is identity, not hyperparameter.** `LGBM12-UNIFORM` and `LGBM12-EXP120` build
byte-identical constructors; only `sample_weight` differs. `UNIFORM` passes `None`, which is exactly
equivalent to a vector of ones and keeps that path identical to an unweighted fit. Backend never
enters identity.

**The R3 control runs on the canonical v1/v2 dataset, never on the v3 population.** R3 is a
control check on the frozen incumbent, so it must reproduce on the exact data that historically
produced the frozen R3 metrics. `reconstruct_r3_control` calls the canonical v1 loader — which
validates the v1 protocol, verifies `data/manifests/processed_manifest.json`, reads only
`train.parquet` and `validation.parquet`, and refuses `test.parquet` — and the applied path passes
it the repository root, never `prepared.raw_history` or `prepared.raw_november`. The decision record
carries `r3_control_dataset_manifest_digest` and `v3_dataset_manifest_digest` side by side and
asserts they differ.

**December materializes only into the Git-ignored qualification workspace.** Requiring
`data/manifests/v3_processed_manifest.json` to flip to `december_2025_decoded: true` would have
dirtied the worktree that the clean-main guard depends on and invalidated the frozen winner's code
lineage. Instead `materialize_december_qualification_data` writes
`artifacts/v3/qualification/data/v3_december.parquet` and
`artifacts/v3/qualification/december_manifest.json`, and runs only after the frozen winner, its
lineage, and the October-31 state have all been validated. The tracked development manifest is
byte-identical before and after qualification.

**December is never decoded during development.** Development routing has no December branch at all
— `split_for_month` returns `None` for 2025-12 and `prepare_v3_dataset` has no December parameter to
pass — so the development manifest describes 23 months (2024-01 → 2025-11) and no December parquet
exists. Only `materialize_december_qualification_data` can decode it, and only against the exact
authorization string `december-2025-qualification-authorized`. The data guard independently refuses
any manifest that claims a December decode, and `require_allowed_v3_path` rejects both
`v3_december.parquet` and the sealed `test.parquet`. The v3 source manifest excludes every 2026
archive outright, so v3 preparation cannot reach January–May 2026 at all.

## Data status

| Artifact | Value |
| --- | --- |
| v3 source manifest digest | `673cac214739e8c0d2991a1bdbd1591a90e8907d7cf5bdbc34caddd72015b6af` |
| Archives | 24 (2024-01 → 2025-12), 709,735,704 bytes |
| v3 processed manifest digest | `4f8f6744b593ee89b32bc9cb9de4c0e848093df3657e2e9441cbc684d567d66f` |
| Months decoded | 23 (2024-01 → 2025-11) |
| `v3_history` rows | 12,717,511 (prevalence 0.21367) |
| `v3_november` rows | 555,295 (prevalence 0.20560) |
| December 2025 decoded | no |
| January–May 2026 decoded | no |

The twelve 2025 archives were reused byte-identically from the v0 manifest: the downloader verified
each against its existing record and reported them as skipped. Preparation took 63 s at eight-way
month parallelism. Raw and generated data stay Git-ignored; only the canonical JSON manifests are
versioned.

## Dry-run runtime estimate

`python scripts/run_v3_development.py` reports an advisory estimate built from real row counts and
the frozen stage structure. It reads the manifest only — never the parquet rows.

| Stage | Identities | Fit rows | Estimate |
| --- | ---: | ---: | ---: |
| Screening LightGBM CPU | 4 | 15,600,000 | 10.0 min |
| Screening CatBoost GPU | 4 | 15,600,000 | 6.3 min |
| CPU confirmation LightGBM | 2 | 7,800,000 | 5.0 min |
| CPU confirmation CatBoost | 2 | 7,800,000 | 25.0 min |
| Full refit LightGBM CPU | 1 | 12,192,141 | 7.8 min |
| Full refit CatBoost CPU | 1 | 12,192,141 | 39.1 min |

Total estimate ≈ **1.8 h**, inside the overnight budget. The throughput constants are conservative
engineering estimates for this host, **not** measurements of a real v3 fit — no v3 model has been
trained. The applied run logs true per-stage runtimes, which should replace them afterwards. Treat
the estimate as order-of-magnitude: a 2–3× miss still finishes overnight.

## Commands

```bash
make validate-v3        # byte-level protocol and lock validation
make v3-dry-run         # preflight plus the runtime estimate; opens no data
make prepare-v3-data    # uncapped preparation, December stays sealed
```

Applied execution is deliberately locked: `run_v3_development.py --apply` requires a clean worktree
on `main` with the frozen protocol commit as an ancestor, online tracking, and write-once markers
under `artifacts/v3/`. December needs the separate `run_v3_december_qualification.py --apply`.

## Verification

- 680 tests pass; branch coverage 86.22% against the 86% gate.
- `ruff check` and `ruff format --check` are clean across 157 files.
- v1 and v2 suites are unchanged and still pass; the only shared edit is an additive extension of
  `ALLOWED_MODEL_FEATURES` for the 11 new feature names.
- Runtime API/Traveler/Monitor images are untouched and gain no v3 modeling dependency; v3 reuses
  the existing `v2` extra rather than adding its own, and `validate_dependency_isolation` asserts it.
