# Data provenance and policy

Brief 02 uses the official BTS Reporting Carrier On-Time Performance archives at
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
Only the small canonical JSON manifests are versioned. Brief 03 subdivides the development data into
January–August base fit (600,000 rows), September tuning (75,000), January–September refit
(675,000), October calibration (75,000), and November–December validation (150,000). Both calibrated
candidates failed at least one mandatory validation gate, so the January–May 2026 test split remains
unread, unscored, and sealed.
