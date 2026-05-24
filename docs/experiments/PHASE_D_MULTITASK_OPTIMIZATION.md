# Phase D Multitask Optimization Decision

Date: 2026-05-25

## Current Decision

Do not implement a new multitask optimization method for Phase D now.

Use `monitor10_rawbest_fixed16` raw as the current stable baseline. It gives the
best short-run balance among the completed Phase C cycle-step ablations, while
the privacy guard is still blocked. Adding PCGrad, GradNorm, CAGrad, uncertainty
weighting, FAMO, or another optimizer-side method now would mix variables before
the current baseline is reproduced and before privacy is unblocked.

No code was changed for this decision. No new training was started.

## Evidence Used

The Phase C ablation held monitor interval, best checkpoint selection, data,
model scale, seed, Stage 2 length, and `lambda_cycle=0.01` fixed. The only
changed variable was the cycle-step setting.

| Run | Raw cosine | Raw single_face_eq1 | Conflict | Weighted cycle/FM |
| --- | ---: | ---: | ---: | ---: |
| `monitor10_rawbest_fixed8` | 0.768856 | 0.988281 | 0.250000 | 0.185173 |
| `monitor10_rawbest_fixed16` | 0.885797 | 1.000000 | 0.153846 | 0.182473 |
| `monitor10_rawbest_schedule_4_8_16` | 0.885008 | 1.000000 | 0.307692 | 0.166599 |
| `monitor10_rawbest_schedule_4_8_16_32` | 0.879287 | 0.998047 | 0.384615 | 0.160650 |

The weighted cycle/FM ratio is generally about `0.15-0.35` across the recorded
Phase C histories. That does not show the lambda-weighted cycle objective
overpowering flow matching. Direction conflict is still present, but fixed16 has
the lowest final conflict among the four completed runs.

The current privacy-side result is also blocked. For the fixed16 raw single-face
artifact, the guard requires `latent_cosine_mean >= 0.95`, but the recorded value
is `0.8763568043956591`, so `privacy_skipped` is `true` and full privacy metrics
were not reported.

## Why Not Add PCGrad Or GradNorm Now

- The weighted cycle/FM ratio does not show cycle dominating flow matching after
  `lambda_cycle=0.01` is applied.
- Fixed16 already gives a stable short-run baseline: final raw cosine `0.885797`,
  raw single_face_eq1 `1.0`, and final conflict `0.153846`.
- The privacy guard has not passed, so adding a new multitask optimizer now would
  make it harder to tell whether later changes come from the optimizer, the cycle
  setting, lambda balance, or privacy-side fixes.

## Next Decision Rules

- Keep fixed16 raw as the current stable baseline until it is reproduced or
  replaced by a clearly better controlled run.
- Consider PCGrad only if later fixed16-style reproductions keep conflict above
  `0.3` while weighted cycle/FM remains in a reasonable range.
- Consider lambda changes, adaptive weighting, or GradNorm only if weighted
  cycle/FM is consistently too low or too high.
- Do not claim a privacy pass until the privacy guard passes and full privacy
  metrics are reported.

## Source Artifacts Checked

- `docs/experiments/PHASE_C_CYCLE_STEP_ABLATION.md`
- `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed8/last_metrics.json`
- `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed16/last_metrics.json`
- `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16/last_metrics.json`
- `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16_32/last_metrics.json`
- `docs/experiments/PHASE_E_PRIVACY_EVAL.md`
- `artifacts/eval/stability_balanced_debug_monitor10_rawbest_fixed16_single_face/result.json`

## What This Does Not Claim

- It does not claim PCGrad, GradNorm, CAGrad, uncertainty weighting, or FAMO is
  bad for this project.
- It does not claim direction conflict is solved.
- It does not claim fixed16 passes privacy.
- It does not claim final image quality or final privacy performance.
