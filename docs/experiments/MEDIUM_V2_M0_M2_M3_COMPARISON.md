# MEDIUM V2 M0/M2/M3 Comparison

Date: 2026-05-30

## Status

Pending for M2 and M3. M0 epoch100 and current read-only metrics are documented in MEDIUM_V1_STAGE2_M0_EPOCH100.md.

## Main Question

Recommend M2 or M3 only if it beats M0 on the intended Stage2 tradeoff: useful representation preservation, stable single-face generation, and acceptable image quality. Privacy must come from a formal privacy eval, not from the M0 ad-hoc probe.

## Theory Note

The comparison should not claim that relation loss adds a special first-order gradient source. The correct claim is that Gram relation uses O(B^2) pairwise geometry constraints in the batch. This can improve batch-local representation geometry discriminability and optimization conditions. M3 then tests whether a projected update can keep this relation signal while controlling gradient conflict with flow matching.

## Curves

scripts/plot_m2_m3_curves.py must produce these files when all inputs exist:

- m2_curves.png
- m3_curves.png
- m0_m2_m3_comparison.png
- m3_projection_diagnostics.png

The script is expected to fail fast when required JSON or required fields are missing. A missing curve is better than a fake curve. M2/M3 quality cadence is NIQE every epoch and FID/KID every 20 epochs.

## Comparison Table

| Run | Status | Best/selected epoch | raw utility | raw source preserved | single_face_eq1 | NIQE | FID | Privacy status | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| M0 | baseline only | 100 documented; current read stage_epoch_1based 113 | pending normalized comparison | 0.859375 at epoch100; 0.76171875 current | 1.0 | 7.167329834326688 at epoch100; 6.495435328441205 current | 126.25408172607422 at epoch100 | ad-hoc only, ad_hoc_ignore_guard=true | not main line |
| M2 | pending | pending | pending | pending | pending | pending | pending | pending formal eval | pending |
| M3 | pending | pending | pending | pending | pending | pending | pending | pending formal eval | pending |

## Completion Gate

This document remains pending until M2 and M3 have history JSON, last metrics JSON, quality JSON, generated curves, and a formal privacy evaluation if privacy is discussed as a pass/fail result.
