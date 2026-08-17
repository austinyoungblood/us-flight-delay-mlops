# Data provenance and policy

The data pipeline uses the official BTS Reporting Carrier On-Time Performance archives at
`https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip`
for January 2025 through May 2026. The 17 verified ZIPs total 501,916,656 bytes. Their stable
`source_manifest.json` digest is
`0a2e8ef929dee6bc1d5a8cdcf8f0161ce0f11bb54c3809dbea1040f09624b561`.

Preparation removes canceled, diverted, missing-target, and invalid scheduled-field records, then
takes a deterministic class-stratified sample capped at 75,000 rows/month with seed 42. Across all
months, the source has 9,882,415 rows: 168,939 cancellations, 26,565 diversions, one missing target,
two invalid schedule rows, 9,686,908 model-eligible rows, and 1,275,000 sampled rows.

| Split | Half-open interval | Rows | Target prevalence | Bytes |
| --- | --- | ---: | ---: | ---: |
| Train | `[2025-01-01, 2025-11-01)` | 750,000 | 0.2190653 | 12,155,843 |
| Validation | `[2025-11-01, 2026-01-01)` | 150,000 | 0.2366600 | 2,483,451 |
| Sealed test | `[2026-01-01, 2026-06-01)` | 375,000 | 0.2129627 | 6,088,402 |

The stable `processed_manifest.json` digest is
`c8aa583cbfe7ad8ee4bdcedaa8d479e2056541c71296f222ae0e0a410a48cdaf`. Two complete preparation
runs reproduced the same manifest and Parquet SHA-256 values. The online W&B dataset is
[`austin-youngblood-university-of-denver/us-flight-delay-mlops/flight-delay-bts-sampled:v0`](https://wandb.ai/austin-youngblood-university-of-denver/us-flight-delay-mlops/artifacts/dataset/flight-delay-bts-sampled/v0),
digest `2ecdb5a6a60b23ed1ee1d603fb976516`.

Run `make download-data` and `make prepare-data` from the repository root. Raw ZIPs, processed
Parquet files, models, W&B cache, `.env`, and credentials are ignored and must never be committed.
Only the small canonical JSON manifests are versioned. Model selection subdivides the development data into
January–August base fit (600,000 rows), September tuning (75,000), January–September refit
(675,000), October calibration (75,000), and November–December validation (150,000). Both calibrated
candidates failed at least one mandatory validation gate, so the January–May 2026 test split remains
unread, unscored, and sealed.

The remediation evaluation reused the same immutable artifact and source hashes. It used
January–October only for four
rolling-origin base folds and final refit, November 1–15 for calibration, and November 16–30 for
selection. No finalist passed the November gate, so December and the final test were not read by the
remediation evaluators.

## V3 seasonality dataset (separate lineage)

The governed v3 experiment adds calendar year 2024 without touching any v0/v1/v2 artifact. Its
provenance lives in `v3_source_manifest.json` (24 archives, 2024-01 through 2025-12, 709,735,704
bytes, digest `673cac214739e8c0d2991a1bdbd1591a90e8907d7cf5bdbc34caddd72015b6af`). The twelve 2025
archives are reused byte-identically from `source_manifest.json`; the downloader verified each
against its existing record and skipped it, so v0/v1/v2 lineage hashes are unchanged. No 2026
archive appears in the v3 manifest at all.

V3 preparation is **uncapped**: it retains every model-eligible row rather than sampling 75,000 per
month, because v3 requires full eligible prior history for its seasonal state and all eligible model
rows for its authoritative refit. Runtime is controlled instead by a 50,000-row-per-month
deterministic search cap applied at candidate-selection time. The v3 splits are written to
`data/processed_v3/` and remain Git-ignored.

| Split | Half-open interval | Rows | Target prevalence |
| --- | --- | ---: | ---: |
| `v3_history` | `[2024-01-01, 2025-11-01)` | 12,717,511 | 0.2136677 |
| `v3_november` | `[2025-11-01, 2025-12-01)` | 555,295 | 0.2056006 |

The stable `v3_processed_manifest.json` digest is
`4f8f6744b593ee89b32bc9cb9de4c0e848093df3657e2e9441cbc684d567d66f`. It describes 23 decoded months
(2024-01 through 2025-11). **December 2025 is not decoded**: preparation refuses that month unless
given the exact qualification authorization, so no `v3_december.parquet` exists and the manifest
records `december_2025_decoded: false`. January–May 2026 remains sealed, unread, and excluded from
the v3 source manifest entirely.

Run `make prepare-v3-data` from the repository root.
