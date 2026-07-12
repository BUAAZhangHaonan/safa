# MeanFlow Flow-Map Representation Guidance Design

**Status:** Approved for implementation and short feasibility experiments.

**Decision:** Use the frozen Stage 2 EMA MeanFlow SiT-B/4 checkpoint. Run a semigroup preflight first, use direct-autograd FMRG-J as the main baseline when that preflight passes, and use constrained initial-noise optimization as a separate feasibility oracle. Do not rename diffusion-only methods as MeanFlow implementations.

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

The frozen supporting models are:

```text
E0: artifacts/checkpoints/e0_medium_v1/best.pt
VAE: artifacts/checkpoints/external/sd-vae-ft-ema
VAE scaling factor: 0.18215
validation index: data/index/val_face_mixed_e14.jsonl
validation features: artifacts/e0_features/val_face_mixed_e14_e0_medium_v1
```

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

The following names must not be used for the MeanFlow implementation:

- MPGD-LDM uses a diffusion latent manifold projection inside a DDIM-like denoising recurrence.
- LGD and LGD-MC use noisy conditional expectations and, for the MC form, stochastic samples around a diffusion state.
- DPS uses a score/Tweedie posterior estimate tied to a diffusion noise schedule.
- FreeDoM and Universal Guidance apply gradients across a multi-step diffusion scheduler.

The current 1-NFE MeanFlow model has no alpha schedule, score output, Tweedie identity, or DDIM recurrence. Replacing those primitives with arbitrary flow-map calls would be a new method, not a faithful reproduction of any of these baselines.

FMRG is the closest reviewed implementation because it is written for a two-time flow map. The reviewed source is pinned at:

```text
/home/hdd3/zhanghaonan/projects/safa-paper-code/fmrg
commit be485ba40f1d1c163b95ad61112aca6baa13ed22
```

The useful branch is FMRG-J: differentiate a reward through a flow-map lookahead with respect to the current flow state. The SAFA port must use direct `torch.autograd.grad`. It must not copy the optional Adam branch from the reference code.

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

Test `s in {0.75, 0.50, 0.25}` on 64 fixed validation sample IDs. Report median and p90 latent residual, decoded pixel L1, PSNR, and cosine between `E0(x_direct)` and `E0(x_split)`. Save direct side-by-side images for every split.

### Operational Gate

This gate is an engineering validity test, not a theorem about MeanFlow:

- Every tensor and metric must be finite.
- At least one split must have median semigroup residual at most 0.10 and p90 at most 0.20.
- The same split must have median decoded endpoint E0 cosine at least 0.95.
- Direct inspection must not show systematic blank output, noise, tiling, color saturation, or broken structure in the composed endpoint.

If no split passes, do not run a full FMRG-J matrix. Keep the report, run Route C, and state that the current checkpoint does not support the required intermediate-map assumption. Do not repair the failure with extra smoothing or post-processing.

The passing split closest to `s=0.25` becomes `t_cut`, because a longer final unguided jump gives the frozen prior more opportunity to restore image quality. If only another split passes, use that split and record the reason.

## 6. Route B: Frozen-EMA FMRG-J

### Recommendation

This is the main baseline if Route A passes. It is the closest faithful match to the current model and to the reviewed FMRG code. It changes no model weight.

Use a decreasing guided schedule:

```text
1 = t_0 > t_1 > ... > t_K = t_cut > 0
```

For each guided interval `t -> s`:

```text
1. With no gradient:
   x_bar = Phi_{t->s}(x, c_null)

2. Re-enable only the input gradient:
   x_bar.requires_grad_(True)
   x0_hat = Phi_{s->0}(x_bar, c_null)

3. Decode and use the full dense representation target:
   L_repr = mean(1 - cos(E0(VAE.decode(x0_hat)), Z0))
   g = d L_repr / d x_bar

4. Normalize each sample and match the flow-step velocity scale:
   u_step = (x_before - x_bar) / (t - s)
   g_scaled_i = g_i / (||g_i||_2 + 1e-8) * ||u_step_i||_2

5. Update the state at time s:
   x = stop_gradient(x_bar - (t - s) * eta * g_scaled)
```

After the last guided interval, make one explicit unguided jump:

```text
x0 = Phi_{t_cut->0}(x, c_null)
```

There is no optimizer state, momentum, Adam, generator update, VAE update, or E0 update. The only optimized object is the temporary flow state.

### Cost

Each guided interval uses one no-grad transition and one differentiable lookahead. The tail uses one final transition:

```text
NFE = 2K + 1
```

Start with `K=1` and `K=2`. Do not describe them as 1-step inference. Report NFE, wall time, images per second, peak allocated VRAM, and peak reserved VRAM beside every quality result.

### Why It May Protect Quality

The representation gradient is applied to an intermediate state reached by the frozen prior. The last jump is always an unguided frozen-prior map. This is a stronger link to the learned image distribution than updating all network weights with a proxy quality loss. It is still not a proof of manifold membership, so FID, KID, Sharpness, and images remain decisive.

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

Run Route A first on GPU 0. This is small and sequential because its result gates Route B.

### Calibration Phase

Run four independent processes at the same time:

| Physical GPU | Arm | Calibration search |
| --- | --- | --- |
| 0 | native EMA control | null-conditioned native sample plus direct-target-condition diagnostic |
| 1 | FMRG-J K=1 | fixed passing `t_cut`, `eta in {0.01, 0.03, 0.10}` |
| 2 | FMRG-J K=2 | equal intervals from 1 to `t_cut`, `eta in {0.01, 0.03, 0.10}` |
| 3 | initial-noise oracle | fixed-radius PGD with `T in {4, 8, 16}` and fixed tested step-size list |

Use 128 samples for numerical and visual calibration. Save every image. Generate 64 paired comparison images in pages with columns `source`, `native`, and `candidate`. An agent must directly open every page before selecting the full-run candidate.

### Full Phase

After calibration, run four independent 2048-sample processes at the same time:

| Physical GPU | Arm |
| --- | --- |
| 0 | matched native EMA baseline |
| 1 | selected FMRG-J K=1 candidate |
| 2 | selected FMRG-J K=2 candidate |
| 3 | selected initial-noise oracle candidate |

No arm may silently reduce its sample count. A failed arm must be marked failed rather than compared with fewer samples.

## 9. Evaluation Contract

### Required Quantitative Metrics

- FID with 2048 generated and the same 2048 real validation images.
- KID mean and standard deviation on the same sets.
- NIQE as a secondary no-reference metric.
- Sharpness using the established grayscale Laplacian-variance definition. Report mean, standard deviation, median, p10, and p90.
- Per-sample `cos(E0(generated), Z0)`. Report mean, standard deviation, median, p10, and p90.
- NFE, wall time, images per second, peak allocated VRAM, and peak reserved VRAM.

Face detection may be recorded as extra information, but it must not decide whether an image collapsed.

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
severe visual failures <= 5% of the 64 reviewed samples
all tensors and reported metrics finite
```

`directional evidence` means the method improves cosine by at least 0.05 over the matched native arm and passes the visual/numerical safety checks, but misses one of the solved thresholds.

Anything else is not evidence that the quality-representation problem is solved.

## 10. Failure and Stop Rules

Stop or skip work under these conditions:

1. Abort all arms if the checkpoint, EMA, epoch, B/4, E0, VAE, index, or feature-cache contract does not match exactly.
2. If Route A fails, skip the full FMRG-J arms. Run Route C and the native arm only. Do not hide the failed premise by changing the threshold after seeing results.
3. Stop a calibration candidate on the first non-finite loss, gradient, latent, decoded image, or metric.
4. Reject a calibration candidate if direct inspection finds severe failures in more than 10% of the 64 pages, Sharpness falls below 80% of native, or cosine improves by less than 0.02 at its strongest tested setting.
5. Do not extend an oracle beyond 16 updates in this first study. If it only improves after leaving the fixed-radius or typical-shell constraint, mark the reachable-set test failed.
6. Do not train model weights in this study. A positive frozen-generator result is required before adapter or full-model training resumes.

## 11. Interpretation

- FMRG-J success means the earlier collapse was mainly an optimization/path problem. The frozen generator can retain quality while the endpoint representation changes.
- Initial-noise-oracle success with FMRG-J failure means the target is reachable, but the current intermediate flow map or guidance schedule is inadequate.
- Failure of both routes is strong evidence that the target is difficult to reach inside this checkpoint's useful support. It is not a mathematical proof that low FID and high cosine are incompatible for SAFA or for other generator families.
- A result from MeanFlow does not replace later diffusion or StyleGAN comparisons. It answers the immediate question with the only mature checkpoint currently online.

## 12. Main Risks

1. Approximate semigroup consistency may be too weak at epoch 1652. This invalidates FMRG-J for this checkpoint even if the general method is sound.
2. E0 guidance may exploit encoder-specific directions. Direct visual review and distribution metrics reduce this risk but do not replace a held-out encoder test.
3. Radial noise projection is not a full Gaussian-distribution constraint. Route C is only a reachability oracle.
4. FMRG-J and the oracle increase NFE and memory. Any quality gain must be reported with that cost rather than compared as if all arms were 1-NFE.
