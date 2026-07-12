# MeanFlow Flow-Map Representation Guidance Design

**Status:** Approved for implementation and short feasibility experiments.

**Decision:** Use the frozen Stage 2 EMA MeanFlow SiT-B/4 checkpoint. Run a four-GPU semigroup preflight first. Compare the current official FMRG-J implementation as the main baseline against the paper's split-state algorithm, and keep constrained initial-noise optimization as a separate feasibility oracle. Do not rename diffusion-only methods as MeanFlow implementations.

## 1. Exact Starting Point

The experiment must use this checkpoint and no substitute:

```text
artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last_nopretrained.pt
```

The runner must verify these fields before allocating generation work:

```text
checkpoint weight source: ema_model_state_dict
stage: stage2
metrics.stage_epoch_1based: 1652
model_config.model_type: meanflow_sit
model_config.sit_patch_size: 4
model_config.sit_hidden_size: 768
model_config.sit_depth: 12
model_config.sit_num_heads: 12
model_config.sit_input_channels: 4
model_config.image_size: 32
model_config.embedding_dim: 512
model_config.learned_null_condition: true
model_config.sample_steps: 1
model_config.sit_data_space: latent
```

The frozen optimization stack is:

```text
E0: artifacts/checkpoints/e0_medium_v1/best.pt
E0 SHA256: d7d2c57a552155776b8c15a4e52e43ec5082fc046aa0aabb4e9709685f7e3d1a
VAE: artifacts/checkpoints/external/sd-vae-ft-ema
VAE scaling factor: 0.18215
validation index: data/index/val_face_mixed_e14.jsonl
validation features: artifacts/e0_features/val_face_mixed_e14_e0_medium_v1
```

Encoder evaluation is deliberately layered:

```text
optimization encoder E0:
  artifacts/checkpoints/e0_medium_v1/best.pt

calibration-only development encoder Edev (allowed before winner lock):
  artifacts/checkpoints/e0_resnet18/best.pt
  SHA256 373b331c917834467e854ddf3fe20f39000532f189ec73f76a1abc55d82e560e

prospective held-out E1 (forbidden before winner lock):
  artifacts/checkpoints/e0_dinov2_large_v2/best.pt
  SHA256 cce0de2f1eab097cb6091886f587a9f334dd84ced1ca4dd5e08c3a765718a14c

prospective held-out E2 (forbidden before winner lock):
  artifacts/checkpoints/e0_convnext_tiny/best.pt
  SHA256 09c88bd416057222abefeba52ebe88d710715ede791ec34198a23ae5e6e850a8
```

E1 and E2 must not be loaded, scored, or inspected during semigroup testing, calibration, candidate selection, or winner revision. Evaluate them once on the locked winner and matched native arm at 2048 samples. Each encoder uses its own coordinate system: compare `cos(Ek(generated), Ek(source))`. Never compute a cross-coordinate quantity such as `cos(E1(generated), Z0_E0)`.

The checkpoint says that Stage 2 was trained with `flow_condition: learned_null_condition`. Therefore all prior-transport calls in the new baselines use the learned null condition. The full dense target `Z0` enters the representation objective, not the transport condition. A direct target-conditioned sample may be recorded as a diagnostic, but it is not the quality baseline because that condition is out of the Stage 2 training contract.

The stored epoch-1650 EMA quality record contains FID 50.3347, but this is not the comparison baseline for the new experiment. The experiment must regenerate a matched native EMA baseline from epoch 1652 with the same sample IDs and initial noise.

## 2. What Transfers Across Generator Architectures

The high-level problem is the same for diffusion, MeanFlow, and StyleGAN:

```text
increase cos(E0(X), Z0) while keeping X on the generator's high-quality support
```

The mechanism that defines and follows that support is different.

| Generator | Native path | Where guidance can act | Main transfer risk |
| --- | --- | --- | --- |
| Diffusion | many noisy states with a known alpha/sigma schedule | every denoising transition or a Tweedie x0 estimate | guidance depends on score/noise parameterization and scheduler equations |
| MeanFlow | a learned two-time flow map, usually sampled in one `1 -> 0` call | an intermediate flow-map state or the initial noise | intermediate composition is useful only if the learned map is approximately semigroup-consistent |
| StyleGAN | one latent-to-image mapping | Z, W, W+, or style directions | latent constraints replace time-path constraints |

This difference is large enough that code and equations cannot be copied unchanged. It is not evidence that quality and representation are incompatible. It only changes the valid way to enforce the generator prior.

## 3. Transfer Boundary From the Reviewed Baselines

The audit root is `/home/hdd3/zhanghaonan/projects/safa-paper-code`. The study is pinned to these snapshots rather than to moving default branches:

| Method | Local directory | Commit | Top-level license state | Audited core entry |
| --- | --- | --- | --- | --- |
| MPGD | `mpgd_pytorch` | `9f94b386` | MIT | `nonlinear/Face-GD/functions/faceid_denoising.py:152`; `linear_inv/guided_diffusion/condition_methods.py:138` |
| DPS | `diffusion-posterior-sampling` | `effbde73` | no top-level license found | `guided_diffusion/posterior_mean_variance.py:120`; `guided_diffusion/condition_methods.py:28`; `guided_diffusion/gaussian_diffusion.py:170` |
| FreeDoM | `FreeDoM` | `1394b1dc` | no top-level license found | `Face-GD/functions/denoising.py:20,261`; `SD_style/ldm/models/diffusion/ddim.py:202` |
| Universal Guidance | `Universal-Guided-Diffusion` | `ff82f880` | no top-level license found | forward: `stable-diffusion-guided/ldm/models/diffusion/ddim_with_grad.py:129,216`; backward: `Guided_Diffusion_Imagenet/guided_diffusion/gaussian_diffusion.py:726` |
| Z+ inversion | `hypershpere-gan-inversion` | `abedeb1b` | AGPL/MIT split across components | `BDInvert_Release/BDInvert/invert_zp.py:211`; `BDInvert_Release/BDInvert/models/stylegan2_generator.py:246` |
| II2S | `II2S` | `6ce02da8` | no top-level license found | `models/Net.py:38,89`; related P-space audit: `hypershpere-gan-inversion/BDInvert_Release/BDInvert/pca_p_space.py:22` |
| StyleCLIP | `StyleCLIP` | `f87a47f7` | MIT | `optimization/run_optimization.py:38`; `mapper/training/coach.py:70,223` |
| FMRG | `fmrg` | `be485ba` | bundled license text is incomplete for all dependencies/components | `fluxfm_sampler_reward.py:388-461,1039-1115,1140-1167` |
| LGD | no author-official repository located | paper only | not applicable | no official code entry to port |

The license column is an audit fact, not permission to copy code. The implementation should reproduce equations through clean SAFA code and preserve the snapshot table for review.

The diffusion methods cannot be ported by name. MPGD uses diffusion latent projection inside a denoising recurrence. DPS requires score/Tweedie posterior estimates. FreeDoM and Universal Guidance require diffusion next-step equations, scheduler coefficients, and in some branches re-noising. LGD/LGD-MC require noisy conditional expectations and stochastic samples around a diffusion state. The current MeanFlow checkpoint has none of the alpha/sigma schedule, score, Tweedie identity, or DDIM recurrence required by those algorithms.

Z+, II2S, and StyleCLIP contribute the valid high-level idea of freezing the generator and constraining the optimized input or adapter. Their StyleGAN Z/W/W+/P interfaces do not exist in SiT-B/4, so their architecture-specific updates are not copied.

FMRG is the closest executable baseline because it targets a two-time flow map. However, the paper's Algorithms 5/6 and official HEAD `be485ba` do not apply the Jacobian guidance at the same state. The experiment must preserve and name both variants instead of merging them into one ambiguous `FMRG-J` label.

## 4. Existing MeanFlow Primitive

`src/safa/models/meanflow_sit.py` already accepts a start time `t` and an end time `r` in the vector field. The general map is:

```text
Phi_{t->r}(x, c) = x - (t - r) * u_theta(x, r, t, c)
```

The existing `sample` method hard-codes only:

```text
t = 1
r = 0
Phi_{1->0}(x_init, c)
```

The implementation should expose the general map without changing the native one-step result. This is a small model API addition, not a new model or training objective.

## 5. Route A: Semigroup Preflight

### Purpose

FMRG-J needs an intermediate state to produce a useful endpoint lookahead. MeanFlow training exposes two times, but it does not guarantee exact composition after finite training. Test this assumption before spending a 2048-sample FMRG run.

For split time `s`:

```text
x_direct(s) = Phi_{1->0}(x_init, c_null)
x_split(s)  = Phi_{s->0}(Phi_{1->s}(x_init, c_null), c_null)

e_semigroup(s) = 2 * ||x_direct - x_split||_2
                 / (||x_direct||_2 + ||x_split||_2 + 1e-8)
```

Test `s in {0.75, 0.50, 0.25}` on one deterministic set of 64 validation sample IDs. Partition the ordered IDs into four disjoint 16-sample shards and run them concurrently on physical GPUs 0-3. Aggregate by sample ID, reject missing or duplicate rows, and apply the gate only to the merged 64 rows. Report median and p90 latent residual, decoded pixel L1, PSNR, and cosine between `E0(x_direct)` and `E0(x_split)`. Save direct side-by-side images for every split.

### Operational Gate

This gate is an engineering validity test, not a theorem about MeanFlow:

- Every tensor and metric must be finite.
- At least one split must have median semigroup residual at most 0.10 and p90 at most 0.20.
- The same split must have median decoded endpoint E0 cosine at least 0.95.
- Direct inspection must not show systematic blank output, noise, tiling, color saturation, or broken structure in the composed endpoint.

If no split passes, do not run a full FMRG-J matrix. Keep the report, run Route C, and state that the current checkpoint does not support the required intermediate-map assumption. Do not repair the failure with extra smoothing or post-processing.

The passing split closest to `s=0.25` becomes `t_cut`, because a longer final unguided jump gives the frozen prior more opportunity to restore image quality. If only another split passes, use that split and record the reason.

## 6. Route B: Two Frozen-EMA FMRG-J Baselines

Both variants use a decreasing schedule `1=t_0 > t_1 > ... > t_N=0`, optimize only a temporary latent state, and leave the generator, VAE, and encoders frozen. Their state ordering is different and must remain visible in every config, artifact, and table.

### B1 Main Baseline: `official_head_current_xt`

This variant follows official HEAD `be485ba` in `fluxfm_sampler_reward.py:388-461,1039-1115,1140-1167`. At current state `x_t`, it differentiates the endpoint lookahead `Phi_{t->0}(x_t)` with respect to `x_t`, then adds that correction to the velocity used for the `t->s` advance.

For one guided interval `t->s`:

```text
u_endpoint = u_theta(x_t, 0, t, c_null)
x0_hat = x_t - t * u_endpoint
L_repr = mean(1 - cos(E0(VAE.decode(x0_hat)), Z0_E0))

current-x_t correction:
  direct/paper-normalized mode:
    g = d L_repr / d x_t
    g <- per-sample normalize(g) * ||u_step|| when normalization is enabled
    delta_xt <- eta * g

  official-Adam mode:
    run the official inner Adam update on x_t
    delta_xt <- -(x_t_after - x_t_before)

advance:
  x_s = x_t - (t-s) * (stop_gradient(u_step) + delta_xt)
```

Support both official transport modes:

```text
flow_map1:
  u_step = u_endpoint
  one vector-field call can serve endpoint lookahead and interval advance

flow_map2:
  u_endpoint = u_theta(x_t, 0, t, c_null)
  u_step = u_theta(x_t, s, t, c_null)
  endpoint and interval advance use separate vector-field calls
```

Support two optimization modes because both are part of the reviewed baseline surface:

```text
official_adam
paper_normalized_direct_autograd
```

Do not silently replace Adam with direct gradient descent. Report the mode, step size, number of inner optimization steps, and whether per-sample velocity-norm scaling is active.

The reference low-NFE contract must be reproduced in unit tests and smoke metadata. For nominal schedule `N=16`, guided prefix `L=4`, unguided tail `U=2`, and `nopt=1`:

```text
official_head_current_xt + flow_map1: NFE = 5
official_head_current_xt + flow_map2: NFE = 8
```

These counts include the official early-stop/tail ordering. They replace the incorrect blanket `2K+1` claim.

### B2 Paper Ablation: `paper_algorithm_split`

Algorithms 5/6 in the paper first transport from `x_t` to `x_s`, then differentiate an `s->0` lookahead with respect to `x_s`:

```text
x_bar = stop_gradient(Phi_{t->s}(x_t, c_null))
x_bar.requires_grad_(True)
x0_hat = Phi_{s->0}(x_bar, c_null)
L_repr = mean(1 - cos(E0(VAE.decode(x0_hat)), Z0_E0))
g_s = d L_repr / d x_bar
g_s <- per-sample normalize(g_s) * ||u_step||
x_s = stop_gradient(x_bar - (t-s) * eta * g_s)
```

Finish with an explicit unguided map to zero. This was the route previously called simply `FMRG-J`; it is not the current official HEAD ordering. Keep it as a paper-faithful comparison, not the main baseline.

Count B2 NFE from actual `CountedFlowMap` calls. Do not infer it from a generic formula, because schedule splitting, inner optimization count, and tail configuration change the total.

### Why These Routes May Protect Quality

Both routes optimize states produced or consumed by the frozen flow map and retain an unguided frozen-prior tail. This ties the representation update more closely to the learned image path than a full-model proxy quality loss. B1 tests the maintained official implementation; B2 isolates the paper's split-state claim. Neither guarantees distribution preservation, so FID, KID, Sharpness, held-out encoders, and images remain decisive.

## 7. Route C: Gaussian-Constrained Initial-Noise Oracle

### Purpose

This route asks a narrower feasibility question:

```text
Does the frozen one-step generator already contain a high-quality output with high Z0 cosine for a nearby admissible initial noise?
```

It is an oracle, not the recommended final sampler. It may require several forward/backward passes per image.

Let `eps_0 ~ N(0, I_d)` be the stable initial noise and optimize only `eps` through:

```text
x0(eps) = Phi_{1->0}(eps, c_null)
L_repr(eps) = mean(1 - cos(E0(VAE.decode(x0(eps))), Z0))
```

Use projected gradient descent with no Adam. Test two explicit projection sets:

```text
fixed initial radius:
  eps <- ||eps_0||_2 * eps / (||eps||_2 + 1e-8)

Gaussian radial typical shell:
  r_min = sqrt(d * (1 - delta))
  r_max = sqrt(d * (1 + delta))
  ||eps||_2 <- clamp(||eps||_2, r_min, r_max)
```

The fixed-radius form is the primary oracle because it preserves the exact initial per-sample norm. The shell is an ablation. Neither constraint preserves the full Gaussian distribution after target-dependent angular optimization. Therefore success proves reachability under a radial constraint, not distributional equivalence to random sampling.

Report initial and final norm, squared norm per dimension, cosine to the initial noise, update norm, and aggregate channel mean/std. Use the final re-evaluated projected point. With `T` updates, report `T+1` NFE.

## 8. Four-GPU Experiment

All arms use the same ordered sample IDs, `sampling_seed=1337`, exact initial noises, EMA weights, null transport condition, E0 target features, VAE, and real-image subset.

### Semigroup Phase

Run Route A first on all four GPUs. GPU `k` receives deterministic ordered positions `k, k+4, ..., k+60`. Aggregate the four 16-sample shards before applying the gate.

### Calibration Phase

Run four independent processes at the same time:

| Physical GPU | Arm | Calibration search |
| --- | --- | --- |
| 0 | native EMA control | null-conditioned native sample plus direct-target-condition diagnostic |
| 1 | B1 official HEAD `flow_map1` | `official_adam` and `paper_normalized_direct_autograd`, closed step-size search |
| 2 | B1 official HEAD `flow_map2` | the same two optimization modes and closed search |
| 3 | B2 paper split plus Route C | paper-split candidate sequence, then fixed-radius and typical-shell oracle candidates |

Every calibration candidate uses the same 64 samples. Save every image and make paired pages with columns `source`, `native`, and `candidate`. An agent must directly open every page before selecting the winner. Calibration FID is diagnostic only: 64 samples are too few for it to choose or reject a winner by itself. Selection uses E0, the allowed ResNet18 development encoder, Sharpness/KID/NIQE, exact costs, and direct images under a fixed rule.

If the semigroup gate fails, do not leave GPUs idle and do not run either FMRG family. Replace the calibration matrix with four explicit Route C configurations spanning fixed-radius versus typical-shell constraints and the pre-registered step sizes. Keep one matched native generation inside each shard for comparison.

### Full Phase

Lock one winner before loading E1 or E2. Then partition the same ordered 2048 sample IDs into four disjoint 512-sample shards and generate both native and winner on physical GPUs 0-3. Aggregate the shards by sample ID before computing metrics:

| Physical GPU | Arm |
| --- | --- |
| 0 | native plus locked winner, sample positions `0 mod 4` |
| 1 | native plus locked winner, sample positions `1 mod 4` |
| 2 | native plus locked winner, sample positions `2 mod 4` |
| 3 | native plus locked winner, sample positions `3 mod 4` |

No shard may silently reduce its sample count. A failed shard fails the 2048 result rather than producing a smaller comparison. Only after the 2048 images and winner identity are immutable, evaluate E1 and E2 once on the matched native/winner images.

## 9. Evaluation Contract

### Required Quantitative Metrics

- FID with 2048 generated and the same 2048 real validation images.
- KID mean and standard deviation on the same sets.
- NIQE as a secondary no-reference metric.
- Sharpness using the established grayscale Laplacian-variance definition. Report mean, standard deviation, median, p10, and p90.
- Optimization metric: per-sample `cos(E0(generated), Z0_E0)`. Report mean, standard deviation, median, p10, and p90.
- Calibration development metric: `cos(Edev(generated), Edev(source))` from the existing ResNet18. It may help select the winner but is not a prospective held-out result.
- Final prospective metrics, evaluated once after winner lock: `cos(E1(generated), E1(source))` and `cos(E2(generated), E2(source))` for native and winner.
- For E0, Edev, E1, and E2, report pairwise cosine-distance Spearman between generated and source embedding geometry and 8-class affect accuracy against the validation labels when that encoder is used in the corresponding phase.
- NFE, wall time, images per second, peak allocated VRAM, and peak reserved VRAM.

Face detection may be recorded as extra information, but it must not decide whether an image collapsed.

The final E1/E2 protocol is prospective. Their files and hashes may be validated before a run, but their model outputs must not be computed until the winner config, checkpoint hash, sample-ID digest, noise seed, and 2048 generated files are locked. Do not use E1/E2 to revise the winner afterward.

### Required Visual Review

Directly inspect the deterministic 64-sample pages. For each candidate, count these severe failures:

```text
blank or near-constant image
unstructured noise
repeated patch or tiled artifact
severe color clipping or saturation
broken global face/image structure
large non-image texture region
```

The review must record sample IDs and categories. It must not infer quality from face-detection rate.

### Decision Labels

Use the fresh native epoch-1652 arm as `B_native`.

`solved` requires all of:

```text
FID <= B_native.FID + 3
Sharpness mean >= 300
Sharpness mean >= 0.95 * B_native.Sharpness mean
KID mean <= B_native.KID mean + 0.005
E0 cosine mean >= 0.516
E1 winner mean cos(E1(generated), E1(source)) > matched native E1 mean cosine
E2 winner mean cos(E2(generated), E2(source)) > matched native E2 mean cosine
severe visual failures <= 5% of the 64 reviewed samples
all tensors and reported metrics finite
```

`directional evidence` means the method improves E0 cosine by at least 0.05 over the matched native arm and passes the visual/numerical safety checks, but misses one of the solved thresholds. A winner that improves E0 but falls or stays flat on E1 or E2 is encoder-specific evidence, not a solved representation result.

Anything else is not evidence that the quality-representation problem is solved.

## 10. Failure and Stop Rules

Stop or skip work under these conditions:

1. Abort all arms if the checkpoint, EMA, epoch, B/4, E0, VAE, index, or feature-cache contract does not match exactly.
2. If Route A fails, skip both FMRG variants. Use GPUs 0-3 for four pre-registered Route C constraint/step configurations. Do not hide the failed premise by changing the threshold after seeing results.
3. Stop a calibration candidate on the first non-finite loss, gradient, latent, decoded image, or metric.
4. Reject a calibration candidate if direct inspection finds severe failures in more than 10% of the 64 pages, Sharpness falls below 80% of native, or cosine improves by less than 0.02 at its strongest tested setting.
5. Do not extend an oracle beyond 16 updates in this first study. If it only improves after leaving the fixed-radius or typical-shell constraint, mark the reachable-set test failed.
6. Do not train model weights in this study. A positive frozen-generator result is required before adapter or full-model training resumes.
7. Do not load E1/E2 outputs before winner lock, and do not reopen candidate selection after seeing E1/E2. If a held-out encoder falls, report that prospective failure.

## 11. Interpretation

- B1 official-HEAD success means the maintained current-state Jacobian path can improve the frozen generator without weight updates.
- B2-only success means the paper's split-state ordering matters and the current official implementation is not the best match for this checkpoint.
- Initial-noise-oracle success with both FMRG variants failing means the target is reachable, but the current intermediate flow map or guidance schedules are inadequate.
- Failure of both routes is strong evidence that the target is difficult to reach inside this checkpoint's useful support. It is not a mathematical proof that low FID and high cosine are incompatible for SAFA or for other generator families.
- A result from MeanFlow does not replace later diffusion or StyleGAN comparisons. It answers the immediate question with the only mature checkpoint currently online.

## 12. Main Risks

1. Approximate semigroup consistency may be too weak at epoch 1652. This invalidates FMRG-J for this checkpoint even if the general method is sound.
2. E0 guidance may exploit encoder-specific directions. The locked-winner E1/E2 prospective test detects this risk without leaking held-out outputs into tuning.
3. Radial noise projection is not a full Gaussian-distribution constraint. Route C is only a reachability oracle.
4. Both FMRG variants and the oracle increase NFE and memory. The counter, not a generic formula, is authoritative for every reported arm.
5. Several audited repositories have absent, incomplete, or component-specific license text. Their code is research reference material, not source to copy into SAFA.
