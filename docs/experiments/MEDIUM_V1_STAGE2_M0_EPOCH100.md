# MEDIUM V1 Stage2 M0 Epoch100

Date: 2026-05-30

## Decision

M0 is no longer the main line. Keep it as the Stage2 baseline and comparison anchor only. Do not continue M0 as the primary experiment after the current run.

M0 cannot be written as a formal privacy pass. The only privacy probe recorded here is ad-hoc, and the run metadata must be described exactly as ad_hoc_ignore_guard=true with not_formal_privacy_pass=true.

## Scope

- Run: g_medium_v1_stage2_m0
- Checkpoint dir: artifacts/checkpoints/g_medium_v1_stage2_m0
- Quality dir: artifacts/eval/g_medium_v1_stage2_m0/quality
- Training log: artifacts/logs/train_g_medium_v1_stage2_m0_gpu3_6.log
- Privacy probe: artifacts/privacy/medium_v1_m0_epoch100_probe_gpu1
- GPU3-6 training was still active when this document was written. This document only reads artifacts and does not change training state.

## Epoch100 Metrics

The epoch100 metrics below were checked against checkpoint history and quality JSON artifacts.

| Metric | Value |
| --- | ---: |
| stage_epoch_1based | 100 |
| loss | 0.058011283247172835 |
| flow_loss_raw | 0.05787898943275213 |
| cycle_loss_raw | 0.013229383102804422 |
| grad_norm | 0.08047994378805161 |
| raw latent cosine mean | 0.9238038249313831 |
| raw source prediction preserved | 0.859375 |
| raw single_face_eq1 rate | 1.0 |
| raw face_detect_ge1 rate | 1.0 |
| raw zero_face rate | 0.0 |
| raw multi_face rate | 0.0 |
| EMA latent cosine mean | 0.9226631131023169 |
| EMA source prediction preserved | 0.85546875 |
| EMA single_face_eq1 rate | 1.0 |
| NIQE mean | 7.167329834326688 |
| NIQE std | 1.703940598670236 |
| FID | 126.25408172607422 |
| KID mean | 0.11806682497262955 |
| KID std | 0.019659586250782013 |

Quality artifact checks:

- stage2_epoch_0100_raw_niqe.json: 512 generated images.
- stage2_epoch_0100_raw_distribution.json: 3969 generated images and 3969 real images.

## Latest Current Metrics

The latest last_metrics.json read for this note was stage_epoch_1based 113. It was read only; no training process was touched.

| Metric | Value |
| --- | ---: |
| stage_epoch_1based | 113 |
| loss | 0.05862721115648747 |
| flow_loss_raw | 0.058515693648159506 |
| cycle_loss_raw | 0.01115175370015204 |
| grad_norm | 0.07834115911722184 |
| raw latent cosine mean | 0.8635160811245441 |
| raw source prediction preserved | 0.76171875 |
| raw single_face_eq1 rate | 1.0 |
| raw face_detect_ge1 rate | 1.0 |
| raw zero_face rate | 0.0 |
| raw multi_face rate | 0.0 |
| EMA latent cosine mean | 0.9145197030156851 |
| EMA source prediction preserved | 0.83984375 |
| EMA single_face_eq1 rate | 1.0 |
| NIQE mean | 6.495435328441205 |
| NIQE std | 2.064470171901143 |

No epoch113 FID/KID artifact was found in the quality cadence. The latest checked FID/KID point in this note is epoch100.

## Ad-hoc Privacy Probe

Artifact: artifacts/privacy/medium_v1_m0_epoch100_probe_gpu1.

This probe is useful as a rough signal only. It is not a formal privacy pass. The recorded metadata says:

- ad_hoc_ignore_guard=true
- not_formal_privacy_pass=true
- stage_epoch_1based=100
- generated_image_count=512
- num_pairs=512

Recognizer summary:

| Recognizer | AUC | EER | TAR@FAR=1e-3 | TAR@FAR=1e-4 |
| --- | ---: | ---: | ---: | ---: |
| adaface | 0.5686988830566406 | 0.453125 | 0.0 | 0.0 |
| arcface | 0.5475692749023438 | 0.46875 | 0.001953125 | 0.001953125 |
| facenet | 0.6074790954589844 | 0.43359375 | 0.009765625 | 0.009765625 |

Face guard summary for the probe:

- face_detect_ge1_rate: 1.0
- single_face_eq1_rate: 1.0
- zero_face_rate: 0.0
- multi_face_rate: 0.0
- latent_cosine_mean: 0.9238019218901172

## Interpretation

M0 kept strong single-face stability through epoch100 and the current read point. The weak point is quality: epoch100 FID is 126.25408172607422, which is worse than the Stage1 long200_v4 FID of 49.21614074707031. That makes M0 a baseline, not the main line.

The privacy numbers do not rescue M0 because they come from an ad-hoc probe with ad_hoc_ignore_guard=true. They can guide later checks, but they cannot be used as a formal privacy result.
