# R9 B v4 resource-distribution failure

Campaign `r9-report-only-formal-v4` was stopped before any run completed because the controller filled GPU slots in index order instead of spreading work across the four contracted GPUs.

- Observed live distribution: GPU 0/1/2/3 had 4/4/4/0 workers.
- The controller and all 12 workers were terminated in an orderly peer-failure cleanup. No v4 controller or worker remains live.
- The partial campaign root is read-only and must never be resumed.
- Preserved evidence: 637 files, 522 PNG files, 332 `per_sample` rows, and 0 completion contracts.
- Frozen inventory SHA256: `f75d6e341cb1f8aa2fc1fdf9b76fe02504a97df60864ddb9fb5cb83eb328e9f8`.

The failure was caused by `_admit_worker` restarting its GPU probe at index 0 for every worker. The superseding v5 controller uses a persistent capacity-aware round-robin cursor, validates the admitted GPU UUID, and starts from a new immutable campaign root.
