# SAFA R9 formal v9 closeout report

Created: 2026-07-30T20:21:18.664430+08:00

## Bottom line

R9 Full finished its official execution and wrote the formal full gate, but the locked winner did not pass. The gate verdict is `failed_locked_winner`. Because this is not a pass, the 193-checkpoint 512 screening and formal Pareto evaluation were not started in this closeout.

## Deep experiment conclusions for next stage

These conclusions are intentionally about different failure modes and next actions; they are not just metric restatements.

1. **formal gate discipline**
   - Conclusion: R9 should be recorded as a failed formal winner, not a near-pass. The affect and distribution gains are real, but the formal contract is conjunctive: one hard identity/quality failure invalidates the arm.
   - Guidance: Next-stage selection should optimize the exact formal gate from the start instead of treating Confirm512 or affect metrics as enough evidence to lock a winner.

2. **identity robustness**
   - Conclusion: The privacy story is recognizer-dependent. AdaFace and FaceNet had complete coverage and acceptable privacy upper bounds, but ArcFace coverage dropped to 2044/2048 and made the identity report unusable.
   - Guidance: The next model should treat exact-one-face preservation under ArcFace as a primary objective. A candidate that improves two recognizers but breaks the third is not robust enough for formal anonymization.

3. **face geometry and detector compatibility**
   - Conclusion: The failed ArcFace coverage is likely a geometry/landmark compatibility problem, not simply a semantic anonymization problem. Four samples were enough to collapse the hard gate.
   - Guidance: Before another large sweep, inspect the four ArcFace-missing samples and add a detector-compatible face-preservation diagnostic to every screening stage.

4. **sharpness and local quality**
   - Conclusion: R9 improved FID, KID, NIQE, e0, and edev, but still failed sharpness relative to native. Aggregate distribution metrics hid a local face quality loss that the formal gate caught.
   - Guidance: The next stage can spend some of the large affect margin to recover high-frequency facial detail. Decoder schedule, guidance strength, or a direct sharpness/landmark regularizer should be tested before wider checkpoint ranking.

5. **visual review role**
   - Conclusion: The visual review found zero severe failures, yet the candidate still failed. Human severe-only inspection is useful for catching obvious collapses, but it is too coarse to certify identity coverage or sharpness preservation.
   - Guidance: Keep visual review as a safety gate, but do not use it as evidence that a candidate is formally safe or high quality.

6. **screening strategy**
   - Conclusion: The 193-checkpoint Pareto branch is not invalid as data, but it is not a valid R9 successor until the formal Full gate failure is acknowledged. Starting it under the old R9 success assumption would mix exploratory screening with formal closure.
   - Guidance: If the 193 checkpoint set is used next, label it as a new exploratory phase and include ArcFace coverage and sharpness gates in the 512 screening itself.

7. **metric tradeoff**
   - Conclusion: R9 has more affect displacement than it needs for the gate, while it lacks enough detector-compatible detail. The bottleneck is no longer whether the model can move expression-space metrics; it is whether it can do so without damaging formal face evidence.
   - Guidance: The next search should prioritize preserving face structure first and then tune anonymization strength, rather than increasing guidance until affect metrics look strong.

8. **execution engineering**
   - Conclusion: The run failures before the final gate were mostly operational: stale heldout markers, monitor-chain strictness, CPU oversubscription, and authorized external GPU PID drift. None of these changed the final scientific verdict.
   - Guidance: Do not build a larger controller for the next phase. Keep simple tmux sessions, strict artifacts, and small direct fixes; spend engineering time on model/evaluator alignment instead.

9. **next-stage starting point**
   - Conclusion: The useful R9 result is a negative boundary: `paper_eta_0p125` is good enough to show promising affect movement but not good enough to serve as the formal locked winner.
   - Guidance: Start the next conversation from this failed gate, not from Confirm512. The first task should be a small diagnostic of ArcFace-missing samples and sharpness loss, followed by a redesigned 512-screening gate.

## What R9 tested

R9 tested the locked `paper_eta_0p125` MeanFlow guidance setting against matched native generation on the formal 2048-sample set. In plain language, it asked whether this anonymization setting keeps affect/quality while reducing identity risk strongly enough to pass the formal gates.

## Real artifacts and counts

- Full gate: `artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/full/gate_contract.json` sha256 `d5d08ba06e22f8f9b3f638fbb8de42afd1629cd28ce6b515ae0d70109c275afd`.
- Report-only finalizer: `artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/full/report_only_finalize.json` sha256 `92ba23baf1f9cfd11ec3fc4b0e6d11507d2e3f617bbd64b33a2117b9d1b23667`.
- Finalizer execution counts: generation 0, evaluator 0, heldout 0.
- Full PNG count: 6152 total = 2048 native generated + 2048 winner generated + 2048 matched native copies + 8 contact sheets.
- Generation results: 32 JSON files; evaluator results: 4 JSON files; heldout raw evidence: `artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/full/heldout_raw_evidence.json`.
- Visual review: `artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/full/visual_reviews/winner.json`, 64 reviewed samples, severe count 0.

## Gate failures

- `arcface_coverage_not_2048`
- `quality:arcface_not_exactly_one_face_per_image`
- `quality:sharpness_below_gate`
- `arcface_identity_report_unavailable`

## Main metrics

- e0: native 0.04921828883198032, winner 0.7533507201085285, delta 0.7041324312765482, one-sided 95% [0.6884717989528554, 0.7197785037979428].
- edev: native 0.2628216306741251, winner 0.5432848038581142, delta 0.28046317318398906, one-sided 95% [0.26838984005967176, 0.29221970387199003].
- niqe: native 4.736061214287843, winner 4.724278868354088, delta -0.01178234593375492, one-sided 95% [-0.04006974066520166, 0.01715403860156869].
- sharpness: native 687.2150477969803, winner 591.7002634128564, delta -95.51478438412391, one-sided 95% [-110.28155540622059, -80.70555113297583].

Identity heldout recognizers:
- adaface: status available, coverage 2048, reason None.
  - native: {'auc': 0.4871354103088379, 'eer': 0.509765625, 'status': 'available', 'tar_at_far': {'0.0001': 0.00048828125, '0.001': 0.00048828125}}
  - winner: {'auc': 0.5128679275512695, 'eer': 0.494140625, 'status': 'available', 'tar_at_far': {'0.0001': 0.001953125, '0.001': 0.0048828125}}
- arcface: status unavailable, coverage 2044, reason incomplete_exact_one_face_coverage.
  - native: {'reason': 'incomplete_exact_one_face_coverage', 'status': 'unavailable'}
  - winner: {'reason': 'incomplete_exact_one_face_coverage', 'status': 'unavailable'}
- facenet: status available, coverage 2048, reason None.
  - native: {'auc': 0.4943356513977051, 'eer': 0.49560546875, 'status': 'available', 'tar_at_far': {'0.0001': 0.00048828125, '0.001': 0.001953125}}
  - winner: {'auc': 0.5128538608551025, 'eer': 0.49365234375, 'status': 'available', 'tar_at_far': {'0.0001': 0.00146484375, '0.001': 0.00390625}}

## Historical closeout state

- Historical ledger: `artifacts/closeout/historical-ledger-v1-precommit-5e5ec305-20260726/experiment_ledger.jsonl`, rows 89, sha256 `5d8f50fd9eead11f2094905718fbcb9f55fefebd1c979fe49242f772a159c1e3`.
- Latest checkpoint plan: `artifacts/closeout/historical-canonical-512-v1/checkpoint_plan_final__b10a45cb36872258.json`, eligible candidates 193, distinct checkpoint SHA 193.
- Latest candidate manifest: `artifacts/closeout/historical-canonical-512-v1/candidate_manifest__5dbb82fdb1c89d8f.json`, candidate_count 193.
- 512 checkpoint screening and Pareto evaluation remain unrun by design because the Full gate failed.

## Resource evidence

- Formal runtime guard samples: 803; max CPU 78.34943301133977; max RAM percent 23.37337557526516; max disk percent 59.17162409928397.
- Formal monitor samples: 292; max CPU 84.54258675078864; max RAM percent 23.44371864029653.
- Report-only finalize2 log: `artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/operator_logs/full-reportonly-finalize2-20260730T200805CST.log`. It ran CPU-only and wrote no new generation/evaluator/heldout execution.

## Closeout files

- `artifacts/closeout/r9-formal-v9-closeout-20260730T202118CST/machine_summary.json`
- `artifacts/closeout/r9-formal-v9-closeout-20260730T202118CST/pareto_table.csv`
- `artifacts/closeout/r9-formal-v9-closeout-20260730T202118CST/invalid_experiments_appendix.md`
- `artifacts/closeout/r9-formal-v9-closeout-20260730T202118CST/visual_checklist.csv`
- `artifacts/closeout/r9-formal-v9-closeout-20260730T202118CST/resource_cost_table.csv`
