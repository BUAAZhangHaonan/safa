# SAFA R8 MeanFlow Guidance Matrix (2026-07-13)

## Conclusion

The R8 calibration matrix completed all 21 arms without OOM, but it did not produce a valid winner. Direct visual review covered every arm: 7 passed, 14 failed, and 733 severe failures were recorded across 1,344 arm-sample decisions. `paper_split_eta0.25` was the only numerical candidate, but it failed visual review on 9 of 64 samples. No arm passed both gates, no selection was locked, and the 2,048-sample full phase was not run.

## Experiment Contract

- Code revision: `b2a265fae0b451763049ca9785c3a325f2315e20` on `master`.
- Phase: calibration only, 21 arms, 64 fixed samples per arm, seed `1337`.
- Sample manifest SHA-256: `ffc1f04f671533ee1498f4b03565826920afcc4e5c6ab244fc6f9b7aa680f964`.
- MeanFlow checkpoint SHA-256: `4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d`.
- Locked schedule: `t_cut=0.25`, guided times `[1.0, 0.75, 0.5, 0.25]`, unguided times `[0.25, 0.125, 0.0]`.
- Campaign root: `artifacts/r8_meanflow_flow_map_guidance/campaigns/r8-calibration-v3`.

The matrix covered the registered official flow-map, paper split, initial-noise oracle, and native unguided families. The campaign contract is `campaign_contract.json`; execution state is `status/matrix_status.json` under the campaign root.

## Calibration Result

All 21 runs ended with `status=passed` and `exit_code=0`. The largest recorded peak allocation was about 4.53 GB, and no log reported CUDA OOM.

`paper_split_eta0.25` was the sole numerical candidate. Its v3 calibration metrics were:

| Metric | Value |
|---|---:|
| Mean E0 cosine | 0.836750 |
| Mean Edev cosine | 0.705796 |
| FID | 144.418243 |
| KID mean | 0.028404 |
| NIQE mean | 4.613746 |
| Sharpness mean | 537.974177 |
| Candidate NFE | 8 |

## Visual Review and Selection Gate

The committed review contract covered all 21 arms and the same 64 registered sample IDs per arm. The review file SHA-256 is `a6dc4bd1a9d6c5b99e7fc65419948be92109859ecb1733619f6fbb12d67ffc14`.

| Review outcome | Count |
|---|---:|
| Arms reviewed | 21 |
| Passing arms | 7 |
| Failing arms | 14 |
| Total severe failures | 733 |
| `paper_split_eta0.25` severe failures | 9/64 |

The seven visual passes did not include the sole numerical candidate. Therefore the numerical and visual pass sets had an empty intersection. Validation with `require_passed=True` rejected the review as designed, so selection could not proceed.

## Reproducibility Audit

The v2 and v3 campaigns used identical registered inputs:

| Check | Result |
|---|---:|
| Same 21-arm set and arm-config digest | 21/21 |
| Same ordered 64 sample IDs and digest | 21/21 |
| Same seed (`1337`) | 21/21 |
| Same checkpoint, evaluator, VAE, index, features, held-out evaluators, and manifest contracts | 21/21 |
| Bit-identical matched-native outputs | 1,344/1,344 |
| Bit-identical native-arm candidate outputs | 64/64 |
| Bit-identical guided/noise candidate outputs | 0/1,280 |

This is not a PNG-encoding-only difference: guided/noise candidate cosine values, route loss histories, FID, and KID also changed between v2 and v3, while native values remained identical. The input-noise builder derives a stable seed from each sample ID, samples with a per-sample CPU `torch.Generator`, then transfers the tensor to CUDA. The guidance execution path does not force deterministic CUDA algorithms or a deterministic cuBLAS workspace. The supported conclusion is therefore that the guided/noise candidate path is not bitwise reproducible on CUDA; this audit does not identify one specific kernel as the cause.

## Decision and Limits

- `visual_review.json` records `passed=false` for the complete multi-arm review.
- No `selection.json` was produced and no winner was locked; `require_passed=True` rejected promotion as designed.
- The full 2,048-sample phase was not launched.
- The result is limited to one fixed 64-sample manifest and seed.
- Future comparisons must use registered numerical tolerances unless the CUDA path is first made bitwise deterministic and revalidated.

Primary evidence is under `artifacts/r8_meanflow_flow_map_guidance/campaigns/r8-calibration-v2` and `artifacts/r8_meanflow_flow_map_guidance/campaigns/r8-calibration-v3`. The full visual decision record is `visual_review.json` under the v3 campaign root. The selected candidate metrics are in `calibration/paper_split_eta0.25/per_sample.jsonl` and `calibration/paper_split_eta0.25/quality.json` under that root.
