# R10 triangle exploration day-one closeout

Date: 2026-07-31 (Asia/Shanghai)

## Decision

Day-one exploratory work is closed. Fixed32 retained only `eta0p125_baseline`, but the selected24 checkpoint pilot produced 0/24 hard-gate survivors. The required `>=6` gate was not met. Full193, stage128, and stage512 were not prepared or started.

This is exploratory stage32 evidence. It is not a formal winner, privacy proof, U95 result, FID/KID conclusion, or full-set result.

## R9 Full diagnostic

The archived exact-one failure was reproduced on the same four union sample IDs, twice per source/native/candidate image, with locked buffalo_l at 224x224. The diagnostic did not modify the formal evaluator or rerun FID/KID. Across 2048 archived sharpness pairs, candidate mean was 591.700263 versus native mean 687.215048; the candidate was sharper in 852/2048 pairs.

Evidence: [diagnostic summary](../../r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/full/diagnostics/r9_full_failures_v1/diagnostic_summary.json).

## Fixed32 interval diagnostic

All arms completed the locked prefix32 with exact-one coverage 32/32. Values are raw cosine or metric units, not percentages.

| Arm | E0 | delta E0 | delta Edev | NIQE | Sharpness | ArcFace delta | Failed gates | R / Q / P |
|---|---:|---:|---:|---:|---:|---:|---|---|
| eta0p125_baseline | 0.818705 | 0.801631 | 0.382596 | 4.689844 | 668.338728 | 0.007479 | none | 0.091607 / 0.051862 / 0.626057 |
| eta0p125_disable_i1 | 0.730764 | 0.713689 | 0.293963 | 4.440067 | 640.223391 | 0.002629 | E0 | -0.025648 / 0.007613 / 0.868525 |
| eta0p125_disable_i2 | 0.744067 | 0.726992 | 0.330054 | 4.757183 | 632.757261 | 0.004604 | E0; sharpness | -0.007911 / -0.004138 / 0.769791 |
| eta0p125_disable_i3 | 0.698792 | 0.681717 | 0.318402 | 4.768756 | 661.617563 | 0.007944 | E0 | -0.068278 / 0.041284 / 0.602807 |

The baseline was the only provisional gate survivor and Pareto arm. Privacy is point ArcFace delta only; U95, FID, and KID were absent and not interpreted.

Evidence: [metric manifest](../fixed32_evaluation/fixed32_metric_manifest.json) and [screening summary](../fixed32_evaluation/screening/summary.json).

## Selected24 checkpoint pilot

The locked pilot contained 12 raw and 12 EMA checkpoints: 23 latent-output and one pixel-output. Generation and materialization completed 24x32 rows. Official typed source/native/candidate buffalo_l evaluation gave exact-one 32/32 for every candidate.

Shared-native per-sample distribution:

| Metric | Min | Median | Max | Mean |
|---|---:|---:|---:|---:|
| NIQE | 2.724730 | 4.667020 | 7.360938 | 4.739949 |
| Sharpness | 196.895670 | 574.984713 | 1590.661201 | 668.827823 |

Distribution across the 24 candidate arm means:

| Metric | Min | Median | Max | Mean |
|---|---:|---:|---:|---:|
| NIQE | 5.022925 | 6.116072 | 6.988252 | 6.094866 |
| Sharpness | 76.712415 | 201.456673 | 327.106656 | 219.433173 |

Representation semantics match Fixed32: E0 is mean candidate cosine; delta E0 and delta Edev are candidate mean minus shared-native mean; ArcFace delta is mean source-candidate cosine minus mean source-native cosine. These are raw cosine-point differences.

Outcome: 0/24 hard-gate survivors, 0 Pareto arms, and 0 selected arms. Failure counts were NIQE 24, sharpness 24, E0 17, delta E0 15, and ArcFace delta 9. The `>=6` continuation condition is false.

Evidence: [selected24](../checkpoint_fixed32_pilot/selected24/selected24.json), [preparation](../checkpoint_fixed32_pilot/preparation_v1/preparation_manifest.json), [materialization](../checkpoint_fixed32_pilot/evaluation_v1/rows_typed_v2/materialization_summary.json), and [screening](../checkpoint_fixed32_pilot/evaluation_v1/screening_typed_v2/summary.json).

## Milestones and transport

The code/evidence milestones are recorded in `machine_summary.json`; the screening evidence ends at `e176c6930295ca8fb0feecce078214617e91f928`. Before this closeout, master was 12 commits ahead of origin. One push attempt failed during SSH banner exchange (`UNKNOWN port 65535`) and was not retried.
