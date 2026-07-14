# SAFA R9 v7 generation benchmark failure

## Outcome

`r9-report-only-formal-v7` is permanently failed and must not be retried. The
fixed generation benchmark request was written once, then RSS sampling failed
while the first process image for a later run was in an exec transition:

```text
process 1814850 has no unique VmRSS while in live state R
```

This was not a CUDA OOM and not a generation-model failure. The v7 sampler read
`VmRSS` from `/proc/<pid>/status`; Linux may temporarily omit that field for a
live process image during exec. Treating that state as malformed was therefore
an implementation error in the resource observer.

## Frozen evidence

- Failed implementation commit: `113840831deab60fdb06f5cee82584187d02aa20`
- Benchmark request SHA256: `f766bdd479df5de1d42410b30a5003082979cf989dd4ac15957fd313b9bd4e5f`
- Failure report SHA256: `0a5bbeb2eb4646545819a8ef05b7aefd2d8ede570a3632dccd621bc927eb6eae`
- 62-file artifact inventory SHA256: `c5739c0586f8c2769d6deca812cb22cefef87ce87bebd357f1069e2c199ff644`
- Completed run before failure: `native__batch_2`
- Partially materialized run: `native__batch_4`
- Retry allowed: `false`

The immutable machine-readable report remains at:

```text
artifacts/r9_meanflow_flow_map_guidance/campaigns/
r9-report-only-formal-v7/benchmark_failure_report.json
```

## Corrective boundary

The replacement observer uses one identity-locked algorithm only:

1. Read `/proc/<pid>/stat` and lock PID plus start time.
2. Read resident pages from `/proc/<pid>/statm` and multiply by
   `SC_PAGE_SIZE`.
3. Read `stat` again and count the sample only if PID and start time are still
   identical.
4. Exclude an exited or reused PID, count a zombie as zero, and fail a malformed
   live identity.

The corrected run uses new campaign ID `r9-report-only-formal-v8` and new
artifact roots. No v7 request or result is reused.
