# Governed v3 interrupted-execution recovery design

Status: **implemented and synthetically tested; not executed**.

This recovery path addresses an execution that completed screening, CPU confirmation, and both
authoritative full refits, then entered its first November finalist while the original exhaustive
threshold algorithm was still running. The original marker remains `status=started`, and no
`decision.json` exists. The implementation does not authorize signaling or inspecting the live
process, opening December, reading the historical test, fitting a model, or contacting
W&B/AWS/Registry. Those actions remain prohibited until a separate operator handoff and explicit
authorization are complete.

## Implemented command surface

Every command is dry-run by default. The `--apply` examples below document a future operator
sequence; they were not run while implementing this mechanism.

```bash
# Static recovery preflight: no data rows, model runtime, tracking, or network.
PYTHONPATH=src python scripts/run_v3_recovery.py --recovery-id <recovery-id>

# After the original process has been terminated externally, freeze the handoff record.
PYTHONPATH=src python scripts/create_v3_recovery_termination_record.py \
  --recovery-id <recovery-id> \
  --source-root <original-worktree> \
  --source-log <original-log> \
  --original-pid <pid> \
  --wrapper-exit-status <status> \
  --termination-mechanism '<operator-supplied mechanism>' \
  --termination-reason threshold_sweep_performance_defect \
  --confirm-original-terminated \
  --apply

# Export completed screening/CPU evidence through the read-only W&B API.
PYTHONPATH=src python scripts/export_v3_recovery_evidence.py \
  --recovery-id <recovery-id> \
  --apply

# Freeze the explicit authorization after both prior records exist.
PYTHONPATH=src python scripts/create_v3_recovery_authorization.py \
  --recovery-id <recovery-id> \
  --selector-test-command '<exact command>' \
  --selector-test-result '<exact result>' \
  --benchmark-command '<exact command>' \
  --benchmark-result '<exact result>' \
  --apply

# Run the authorized two-base refit and all 15 November finalists.
PYTHONPATH=src python scripts/run_v3_recovery.py \
  --recovery-id <recovery-id> \
  --tracking online \
  --apply

# Separately adopt a completed recovery without rewriting the source marker.
PYTHONPATH=src python scripts/adopt_v3_recovery.py \
  --recovery-id <recovery-id> \
  --apply
```

The source evidence exporter ignores the partial original November finalist and requires exactly
eight completed screening runs plus four completed CPU-confirmation runs. It validates run IDs,
URLs, timestamps, group, protocol and implementation lineage, family, backend, candidate identity,
configuration, weight policy, feature-state lineage, and four complete fold summaries. Existing v3
ranking functions then recompute top-two and top-one advancement. The immutable evidence JSON,
termination record, and authorization cross-reference each other by SHA-256.

Recovery outputs live only under `artifacts/v3/recovery/<recovery-id>/`. A separate adoption record
copies completed recovery artifacts into previously absent canonical paths while preserving the
original `status=started` marker byte-for-byte. Adoption refuses an existing canonical decision.

## Current durability boundary

Source inspection establishes the following without inspecting the live process or its memory:

| Stage evidence | Current durability | Recovery consequence |
| --- | --- | --- |
| Original execution identity | Protocol SHA, implementation SHA, and start time are written to the immutable initial marker | Preserve its original bytes and SHA-256; never delete or silently replace it |
| Historical feature state | Written atomically before screening | Potentially reusable only after its bytes, schema, as-of date, protocol, and tracked digest all verify |
| Screening and CPU confirmation | Fold metrics and candidate identity are sent to one tracked run per candidate/stage | Potentially reconstructable read-only from a complete, exact original run set; missing or ambiguous evidence fails closed |
| Advancement ranking | Computed in memory from CPU confirmation | Recompute with the unchanged pure ranking function from verified tracked evidence and require exactly one advanced identity per family |
| Two full-refit base models | Held only in process memory | Not durably recoverable; after the process ends they must be refit from the exact authorized full-refit data |
| First November finalist | Model/bundle checks occur before threshold selection, but no durable decision is written until all finalists finish | Treat the partial finalist as incomplete; do not use it as selection evidence |
| Final decision/winner | Written only after all 15 finalists complete | Cannot be inferred from partial work; recovery must evaluate all 15 finalists |

The current implementation has no supported checkpoint interface for extracting the two fitted
models from a running process. Ptrace, signal-driven injection, monkey-patching, or memory scraping
would interfere with the live execution and would not provide governed serialization evidence.
Those models are therefore **not recoverable** under the present design.

## Smallest defensible recovery mechanism

Recovery requires a separately reviewed implementation and explicit operator authorization. It
must not run while the original process is alive.

1. **Preserve the original execution.** After separately authorized process disposition, copy the
   original marker and any already-written state into a new immutable recovery directory. Record
   their source paths, sizes, SHA-256 values, the original protocol SHA, original implementation
   SHA, start time, and the fact that `decision.json` was absent. Do not edit or delete the original
   marker.
2. **Create an explicit recovery authorization.** Require a signed/committed authorization payload
   naming the original marker SHA, original implementation SHA, fixed reason
   `threshold_algorithm_performance_defect`, recovery implementation SHA, allowed source evidence,
   and the prohibition on December/test/Registry/AWS access.
3. **Use a separate recovery marker.** A dedicated default-dry-run command should atomically create
   `artifacts/v3/recovery/<recovery-id>/execution_marker.json`. It must link to the original marker,
   start as `status=started`, record each verification/refit/finalist stage, and end as either
   `complete` or `failed`; it must never rewrite the original marker.
4. **Reconstruct screening evidence read-only.** Query only the exact original W&B group and require
   the complete expected screening and CPU-confirmation run cardinality, exact candidate identities,
   protocol SHA, original implementation SHA, backend, four fold IDs, and all ranking metrics.
   Recompute summaries and advancement with the existing pure functions. Any missing, duplicate,
   unfinished, or inconsistent run aborts recovery. Original runs are never changed.
5. **Rebuild only the necessary data state.** Re-open only the already-authorized January 2024–
   November 2025 development sources. Verify committed manifests and Parquet hashes before reads,
   keep December undecoded, and reject `test.parquet`. Recomputed selection/refit frames and the
   October 31 state must match the original recorded lineage and state digest exactly.
6. **Refit exactly two bases.** Because the authoritative full-refit models were not checkpointed,
   refit only the one verified advanced LightGBM identity and one verified advanced CatBoost
   identity, on CPU, over the unchanged full February 2024–October 2025 rows with their frozen
   weight policies. Do not rerun screening or CPU confirmation.
7. **Evaluate all finalists from the beginning.** Rebuild the unchanged six base variants and nine
   ensemble variants, then evaluate all 15 November finalists with the exact optimized unique-score
   sweep. Do not reuse the incomplete first-finalist result. New recovery tracking runs must use a
   distinct group linked to the original execution; creating them requires explicit online-tracking
   authorization.
8. **Write separate recovery evidence.** Atomically write a recovery decision, sanitized finalist
   evidence, reconstructed-screening digest, refit model hashes, and the completed recovery marker.
   If a winner exists, freeze recovery-specific model/lock artifacts without overwriting any
   original path.
9. **Keep December fail-closed.** The existing December handoff must continue rejecting the original
   `status=started` marker. A later, separately reviewed handoff may accept a completed recovery
   marker only after verifying its full link to the preserved original evidence. Recovery itself
   must not open December.

## Recovery invariants

- The v1/v2/v3 protocol YAML and lock bytes remain unchanged.
- Candidate identities, CPU ranking, weight policies, features, periods, calibration variants,
  threshold eligibility, all six tie breaks, gates, and production `v0` remain unchanged.
- Original marker, state, and W&B evidence are immutable inputs, never silently corrected.
- Screening evidence is reused only if it is complete and independently reconstructs the exact
  authoritative advancement; otherwise recovery stops.
- Both full-refit models are refit because no durable model checkpoint exists.
- A partial finalist is never treated as a completed finalist.
- Recovery has a new marker, new identity, new tracking group, and explicit reason.
- December, the January–May 2026 historical test, AWS, Registry, and deployment remain out of scope.

## Estimated recovery runtime

The committed v3 processed manifest contains `12,192,141` full-refit rows per base. Using the existing
conservative planning rates, the two required fits are approximately:

- LightGBM CPU: `12,192,141 / 26,000` = 468.9 seconds (7.8 minutes)
- CatBoost CPU: `12,192,141 / 5,200` = 2,344.6 seconds (39.1 minutes)
- two-base fit subtotal: 2,813.6 seconds (46.9 minutes)

The exact grouped sweep measured 1.88 seconds for 555,295 unique synthetic scores, so 15 worst-case
November sweeps add roughly 28 seconds of threshold computation rather than hours. Allowing for
manifest verification, development-frame/state reconstruction, two model serializations, 15 model
prediction/calibration/bundle checks, read-only evidence reconstruction, and explicitly authorized
tracking, the defensible recovery estimate is **60–90 minutes** on the referenced host. This is an
engineering estimate, not a deadline guarantee; any later authorized recovery should replace it
with recorded stage timings.
