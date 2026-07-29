# R9 phase handoff: stale-lease boundary

## Scope and baseline

This is an audit handoff, not a final closeout or Pareto result.  The Git
baseline is `9d7d81df88911e789e269e0142e6b0be81e96ccf` (`master` and
`origin/master` at handoff).  The formal worktree was clean when this record
was prepared.

Two minimal R9 e2e repairs were pushed before the attempted e2e run:

- `287fcac fix(r9): scope Full admission temperatures to selected GPUs`;
- `9d7d81d fix(r9): bind E2E guard to materialized policy`.

They address direct admission/guard defects only.  They do not add Node3, a
v5 controller, a crash-recovery protocol, or a new general launch framework.

## Evidence retained from Confirm512

`paper_eta_0p125` is the selected **report-only Confirm512 input**.  The
selection reused 2,560 existing PNG files and five evaluator results; its
generation and evaluator execution counts are both zero.  The source
selection and gate evidence are bound by
`artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v8/confirm512_supersessions/report-only-v3/f4323db51df0c4980a3b8160bd741ec72aa45a4836bf3c1f4fde5ee0f86a83f0/`.

This is not a privacy or anonymity result.  The Confirm512 evidence used one
ArcFace recognizer and one seed.  It may choose the input to the next formal
gate; it does not establish a final winner.

## Current evidence boundary

- The historical ledger contains 89 entries.  The current-policy checkpoint
  inventory contains 193 SHA-deduplicated checkpoints.  Unified 512 screening
  has started for none of those checkpoints: completed screening count is 0.
- `full_e2e` preparation completed with six artifact writes and execution
  counts `generation=0`, `evaluator=0`.  Its prepared plan is
  `artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/full_e2e/plan.json`
  (`plan_sha256=369f4495a7c95ebe663b1a9c1ad45c7ae417f8e8abe24aadd4503d8cf24abbf2`).
  Preparation is not an R9 Full result.
- No R9 Full 2,048 generation, matched-native evaluation, unified checkpoint
  screening, Pareto candidate, Pareto formal evaluation, or final closeout
  result exists at this handoff.

## Exact e2e blocker

The e2e admission encountered the stale global GPU-slot lease at
`/tmp/safa-r9-gpu-slots-v1/gpu_2246576ff1516ec44b1c78e9.slot_0.lock`.
The record names worker `evaluator-smoke:arcface`, campaign
`r9-report-only-formal-v9`, GPU UUID
`GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6`, and resource-contract SHA256
`5594e0ab1dd5c0727ba4b5eb4356a83de97069a3b30931e195648b86b924751d`.
The lock-file SHA256 observed at handoff is
`f68981a937e02e05fbaab5248d92d274827ba509332f04bf6208222e53797d7b`.

The worker PID was absent and no kernel lock was held, but reclaim correctly
refused because the required peer-status evidence was missing from the
current policy root:
`artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/worker_status/9fceef6060adf6904a4808f96d9346debad4192d987a8eb66e36cca72a33b460.json`.
The reproducible failure is recorded in
`artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/full_e2e/operator_logs/e2e-run-policyfix-20260729T1851CST.log`.

Do not delete the lock, fabricate a worker status, or treat an old nested
staging state as current evidence.  Do not merge or use `/tmp/safa-node3-*`.

## Strict next-stage order and gates

1. Perform a separately reviewed, auditable lease migration or recovery that
   preserves the stale record and produces a policy-valid terminal ownership
   result.  **Gate:** the official admission path succeeds without manual
   lock deletion or invented status evidence.
2. Run the existing eight-sample e2e with GPU 0--3 and batch size 2 in an
   independent tmux session.  **Gate:** each candidate/native generation and
   both evaluator outputs are present, hash-bound, and the worker terminals
   are valid.
3. Run R9 Full 2,048 winner plus matched native by the verified path.
   **Gate:** exact sample coverage, locked configuration and checkpoints,
