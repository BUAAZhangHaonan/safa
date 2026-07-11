# SAFA R7 Independent-Prior Matrix (2026-07-12)

## Conclusion

The four R7 runs finished training and evaluation, but none met the minimum gate for phase 2. Phase 2 was therefore not started.

- Independent-prior with cap `0.25` learned strong representation alignment, but image quality and face detection collapsed together.
- Reducing the cap to `0.05` preserved image quality, but the cosine and rank signal fell back near the coupled control.
- Halving the generator learning rate did not resolve the high-alignment quality collapse.
- Privacy was not verified. Face detection and representation alignment are not privacy evidence.

The next experiment should freeze the pretrained prior and train a zero-at-null centered adapter. Further sweeps over only `lambda_repr`, the PU cap, or the generator learning rate are not supported by this matrix.

## Reproduction

The matrix ran from repository HEAD `e67841998fdad76e060fd53cbec093a76755d668` (`experiment: allow authorized GPU sharing for R7`). The main supporting commits were:

- `c52652e`: independent-prior many-to-many semantics.
- `6a5e8ba`: strict independent-prior pairing validation.
- `a1d4509`: the four R7 experiment configs.
- `3f9ba01`: the four-GPU train/eval runner.
- `e678419`: authorized sharing of a busy GPU while preserving explicit GPU binding.

All runs used seed and sampling seed `1337`, the same e15 MeanFlow-SiT checkpoint, `resume_mode: model_weights_only`, full generator training, one stage-2 epoch, batch size `4`, AdamW, no AMP, no EMA, and no LPIPS term. Many-to-many pairing used `balanced_epoch_cycle`, one target per source, and pairing seed `1337`. The representation objective used `lambda_repr=0.5` and `repr_learning_rate=3e-5`. Distribution evaluation was disabled during training and run afterward.

| GPU | Config | M2M semantics | Flow condition | Generator LR | Repr step cap |
|---:|---|---|---|---:|---:|
| 0 | `configs/medium_v2/experiments/r7_coupled_embedding_cap025_lr1e4_gpu0.yaml` | coupled control | embedding | `1e-4` | `0.25` |
| 1 | `configs/medium_v2/experiments/r7_independent_prior_cap025_lr1e4_gpu1.yaml` | independent prior | learned null | `1e-4` | `0.25` |
| 2 | `configs/medium_v2/experiments/r7_independent_prior_cap005_lr1e4_gpu2.yaml` | independent prior | learned null | `1e-4` | `0.05` |
| 3 | `configs/medium_v2/experiments/r7_independent_prior_cap025_lr5e5_gpu3.yaml` | independent prior | learned null | `5e-5` | `0.25` |

Evaluation used one common sampling seed and the same validation index:

- Cosine and face rate came from the full 3,969-image evaluation.
- FID used 2,048 real and 2,048 generated images through `scripts/r5_eval.py`.
- KID and NIQE used `scripts/eval_generation_quality.py`, the canonical R6 2,048-row real index (SHA256 `cbff8aaf5b151a01345437fb94443909fe5cd22d4d8d63db8c3f1a0f9706e905`), `subset_seed=1337`, and 2,048 generated images.
- Sharpness used the first 512 generated images in lexical order and `cv2.Laplacian(gray, CV_64F).var()`.
- Spearman came from the fixed 256-sample training validation, not the 3,969-image evaluation.
- Visual severity counts came from the fixed 64-image review with shared sample IDs and noise.

## Results

Lower is better for FID, KID, and NIQE. Higher is better for the other numeric metrics. KID and NIQE show mean plus population standard deviation.

| Run | FID | KID | NIQE | Sharp mean / median | Cosine | Spearman | Face rate | Severe visual issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Coupled, cap `0.25`, LR `1e-4` | 58.6891 | 0.032593 +/- 0.006910 | 5.7732 +/- 1.4275 | 280.12 / 184.38 | 0.196632 | 0.026242 | 0.998488 | 14/64 |
| Independent, cap `0.25`, LR `1e-4` | 134.0608 | 0.079272 +/- 0.011004 | 8.1832 +/- 2.7275 | 160.83 / 95.00 | 0.763082 | 0.713538 | 0.849836 | 31/64 |
| Independent, cap `0.05`, LR `1e-4` | 62.9036 | 0.035920 +/- 0.006997 | 5.6003 +/- 1.4590 | 318.97 / 209.14 | 0.204867 | 0.049618 | 0.997984 | 18/64 |
| Independent, cap `0.25`, LR `5e-5` | 133.5936 | 0.081939 +/- 0.011303 | 7.0660 +/- 2.1955 | 232.63 / 136.95 | 0.709063 | 0.697229 | 0.845049 | 27/64 |

## Phase-2 Gate

The minimum phase-2 gate was FID `<=45`, sharpness mean `>=280`, cosine `>=0.30`, face rate `>=0.99`, severe visual issues `<=4/64`, positive `pu_effective_repr_lr`, and observed repr/FM step ratio no greater than configured cap plus `0.01`. KID, NIQE, and Spearman were diagnostic metrics without separate gate values.

- **Coupled control:** FID failed; sharpness passed; cosine failed; face rate passed; visual review failed. PU activity passed (`2.21e-7 > 0`; ratio `0.1451 <= 0.26`). Overall: fail.
- **Independent, cap 0.25:** FID, sharpness, face rate, and visual review failed; cosine passed. PU activity passed (`5.55e-8 > 0`; ratio `0.1475 <= 0.26`). Overall: fail.
- **Independent, cap 0.05:** FID, cosine, and visual review failed; sharpness and face rate passed. PU activity passed (`1.25e-8 > 0`; ratio `0.0305 <= 0.06`). Overall: fail.
- **Independent, cap 0.25, low LR:** FID, sharpness, face rate, and visual review failed; cosine passed. PU activity passed (`1.81e-8 > 0`; ratio `0.1531 <= 0.26`). Overall: fail.

No run met all gate items, so no checkpoint was resumed into phase 2.

## Interpretation

The cap-`0.25` independent-prior runs reached cosine `0.71-0.76` and Spearman `0.70-0.71`, but both moved to FID `133-134`, KID about `0.08`, face rate about `0.85`, and 27-31 severe cases. This is not a useful alignment gain. It is a systematic image-distribution collapse.

The cap-`0.05` run restored the quality side: its FID, KID, NIQE, sharpness, and face rate were close to or better than the coupled control. Its cosine `0.204867` and Spearman `0.049618`, however, were also close to the control. The smaller cap removed most of the independent-prior representation effect.

The `5e-5` run retained high cosine and Spearman but remained almost identical to the failed cap-`0.25`, LR-`1e-4` quality regime. A lower generator learning rate therefore did not solve the mechanism. These two cap settings and two learning rates expose the same trade-off from opposite sides; another scalar sweep is unlikely to change it.

The next direction is a **frozen-prior, zero-at-null centered adapter**. The pretrained prior path should remain frozen, and a conditional residual should be parameterized as `Delta(z) = A(z) - A(z_null)`. This makes the residual exactly zero at the null condition while allowing a separate conditional path to learn representation control. The key test is whether this structural constraint preserves the pretrained image distribution while raising cosine, not whether another loss weight can balance two competing updates to the same prior.

## Artifacts And Limits

- Runner status and logs: `artifacts/logs/r7_matrix_all/`.
- Checkpoints and training metrics: `artifacts/checkpoints/r7_*/`.
- Full evaluation outputs: `artifacts/r5_eval_r7_*/result.json`, `per_sample.jsonl`, and `generated_images/`.
- KID, NIQE, and sharpness supplement: `artifacts/r7_quality_supplement/`.
- Review images: each `artifacts/r5_eval_r7_*/generated_images/` directory, using the fixed 64-image subset.

The matrix used one seed and one epoch. Spearman used 256 validation samples, while cosine and face rate used all 3,969 images. The KID/NIQE script selected a seeded 2,048-image subset from 3,969 generated images, so it did not use the same generated selection as lexical-first FID. The visual severity counts are manual judgments and do not have row-level annotations in the artifact tree. Privacy metric payloads were empty and no recognizer-based privacy audit was completed; privacy remains unverified.
