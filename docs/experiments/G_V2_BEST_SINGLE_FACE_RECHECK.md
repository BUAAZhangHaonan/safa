# G V2 Best Single-Face Recheck

Date: 2026-05-25

## Current Decision

This recheck is a privacy protocol blocker, not a privacy pass.

The fail-fast guard passed on generated images, but the full privacy recognizer
pass did not complete. The eval exited with code 1 after ArcFace found 2 faces
in a clean source image, which violates the privacy protocol requirement that
each source image has exactly one recognizer-detected face. Therefore this run
does not have TAR, EER, AUC, or privacy ROC results.

No training, eval rerun, code change, or artifact regeneration was done for this
document update. This document only records the already existing Phase0 outputs.

## What Changed

- The reported artifact is the `g_v2_best` single-face recheck:
  `artifacts/eval/g_v2_best_single_face_recheck`.
- The eval index is `data/index/val_single_face.jsonl`, with 3969 samples.
- Full generated PNG output exists at
  `artifacts/eval/g_v2_best_single_face_recheck/generated_images`.
- Generation quality was computed for the same generated image set:
  `artifacts/eval/g_v2_best_single_face_recheck/generation_quality.json`.

## What Stayed Fixed

- E0 checkpoint stayed fixed at `artifacts/checkpoints/e0/best.pt`.
- G checkpoint stayed fixed at `artifacts/checkpoints/g_v2_best/best.pt`.
- The eval used the raw checkpoint path, not EMA.
- Dataset feature cache stayed fixed at `artifacts/e0_features/val_single_face`.
- Sampling seed stayed fixed at `1337`, with stable `x_init` enabled.
- Privacy recognizers stayed fixed to ArcFace, FaceNet, and AdaFace.

## Checkpoints And Eval Mode

| Item | Value |
| --- | --- |
| Eval config | `configs/eval_g_v2_best_single_face_recheck.yaml` |
| E0 checkpoint | `artifacts/checkpoints/e0/best.pt` |
| E0 sha256 | `5f165c520fad315dd1550676c6515c3480585e8ea0dcf1841fd678c8f1963e0f` |
| G checkpoint | `artifacts/checkpoints/g_v2_best/best.pt` |
| G sha256 | `adcbfd09ddaa4aa7ffb572c4ed5216e4f62e3a438950d133e414eb4718824ef3` |
| Checkpoint model | `raw` |
| EMA in training config | `ema.enabled: false` |
| EMA eval in training config | `evaluate_ema: false` |

The raw/EMA point matters here. This result reports `checkpoint_model: raw`.
It should not be read as an EMA result.

## Artifacts Checked

| Artifact | Path |
| --- | --- |
| Result JSON | `artifacts/eval/g_v2_best_single_face_recheck/result.json` |
| Quality JSON | `artifacts/eval/g_v2_best_single_face_recheck/generation_quality.json` |
| Per-sample rows | `artifacts/eval/g_v2_best_single_face_recheck/per_sample.jsonl` |
| Generated images | `artifacts/eval/g_v2_best_single_face_recheck/generated_images` |
| Sample grid dir | `artifacts/eval/g_v2_best_single_face_recheck/samples` |

Generated image count is 3969. The quality JSON also reports 3969 generated
images and 3969 real images.

## Affective And Single-Face Metrics

These values were read from `result.json`.

| Metric | Value |
| --- | ---: |
| `latent_cosine.mean` | 0.9682674512753305 |
| `latent_cosine.p10` | 0.9517428636550903 |
| `source_prediction_preserved.mean` | 0.9135802469135802 |
| `label_accuracy_generated.mean` | 0.5399344923154447 |
| `face_detect_ge1_rate` | 1.0 |
| `single_face_eq1_rate` | 1.0 |
| `zero_face_rate` | 0.0 |
| `multi_face_rate` | 0.0 |

The single-face guard side is clean for generated images: at least one detected
face is 1.0, exactly one detected face is 1.0, and both zero-face and multi-face
generated rates are 0.0.

## Generation Quality

These values were read from `generation_quality.json`.

| Metric | Value |
| --- | ---: |
| FID | 144.543212890625 |
| KID mean | 0.13880713284015656 |
| KID std | 0.012893247418105602 |
| NIQE mean | 5.422948851819098 |
| NIQE std | 1.0067968732400285 |

These quality numbers compare the 3969 generated single-face PNGs against the
3969 real single-face validation images. They are not privacy metrics.

## Privacy Guard And Blocker

| Field | Value |
| --- | --- |
| `privacy_guard_pass` | `true` |
| `privacy_skipped` | `true` |
| `skip_reason` | `privacy_protocol_blocker` |
| `metrics.privacy` | `{}` |

`privacy_guard_pass=true` only means the pre-privacy guard passed. It does not
mean privacy evaluation passed.

The blocker happened after the guard passed. During the clean privacy recognizer stage,
ArcFace detected 2 faces in a source image. The privacy protocol requires
exactly one recognizer-detected face for each clean source image, so the run
raised a protocol blocker and stopped before privacy metrics were attached.

## What This Does Not Prove

- It does not prove anonymization privacy.
- It does not prove a privacy pass.
- It does not provide TAR, EER, AUC, or privacy ROC results.
- It does not compare ArcFace, FaceNet, and AdaFace privacy scores.
- It does not show that the generated-image face guard is enough for full
  privacy evaluation. The source-side recognizer protocol still has to pass.

## Next Step

Fix or quarantine the source-side protocol issue first: identify the clean
source sample where ArcFace detects 2 faces, decide whether it is an index/data
problem or a recognizer protocol problem, and only then rerun the full privacy
eval. Report privacy metrics only after the recognizer pass completes without a
protocol blocker.
