# MEDIUM V2 Projected Update Failure Analysis

Date: 2026-06-04

This note summarizes the current medium-v2 results after the M3 point-projected run reached epoch 49. It is an interim failure analysis, not a final paper conclusion.

## Conclusion

The current projected-update experiment has not achieved efficient FM/CL decoupling.

M3 keeps generated images in the single-face regime, but it does not move representation preservation beyond the Stage 1 baseline. In practical terms, it has spent roughly two days training while leaving the main utility metrics almost unchanged.

The result should not be explained only as "the 5M FM is too small." The CL-only probes show that the same generator can be pushed to much higher cosine when the FM constraint is removed. The failure is more specific: the current projected update protects the FM side so strongly, or exposes such a narrow useful feasible direction, that the representation update does not translate into usable utility improvement.

## Current Experiment Snapshot

| Experiment | Checkpoint / epoch | Latent cosine | Source preserved | Single-face eq1 | FID | KID mean | NIQE | Main conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stage1 long200_v4 | `artifacts/checkpoints/g_medium_v1_stage1_long200_v4/last.pt`, epoch 200 | 0.6388 | 0.4922 | 0.9980 | 49.216 | 0.03555 | 6.109 | Best collected Stage 1 face prior so far. It is not a privacy/utility model. |
| M0 weighted-sum | epoch 100 snapshot | 0.9238 | 0.8594 | 1.0000 | 126.254 | 0.11807 | 7.167 | Utility improves, but image distribution quality is strongly damaged. |
| M2 Gram weighted-sum | epoch 88 last metric | 0.9065 | 0.8359 | 1.0000 | n/a | n/a | 6.913 | Gram weighted-sum did not jointly solve utility and quality. |
| Point-only CL-only | epoch 22 | 0.9536 | 0.8984 | 0.0000 | 371.632 at epoch 20 | 0.43597 at epoch 20 | 9.126 | CL alone can raise cosine, but generation collapses away from detectable faces. |
| Point+Gram CL-only | epoch 28 | 0.8949 | 0.7617 | 0.0020 | 335.702 at epoch 20 | 0.35693 at epoch 20 | 10.823 | Gram did not beat point-only in this probe. It also collapses generation. |
| Null-FM | epoch 120 | 0.1045 | 0.1406 | 1.0000 | 80.233 | 0.07296 | 5.944 | Null conditioning can generate faces, but it intentionally drops E0 condition information. |
| M3 point-projected | epoch 49 | 0.6411 raw / 0.6726 EMA | 0.5020 raw / 0.5137 EMA | 1.0000 | 72.870 at epoch 40 | 0.06233 at epoch 40 | 4.774 | Face quality is protected better than M0, but CL utility is almost unchanged from Stage 1. |

The most important comparison is Stage1 vs M3:

| Metric | Stage1 epoch 200 | M3 epoch 49 raw | M3 epoch 49 EMA |
| --- | ---: | ---: | ---: |
| Latent cosine | 0.6388 | 0.6411 | 0.6726 |
| Source prediction preserved | 0.4922 | 0.5020 | 0.5137 |
| Single-face eq1 | 0.9980 | 1.0000 | 1.0000 |

This is the clearest evidence for the "near standstill" diagnosis. M3 protects faces, but it does not substantially improve representation preservation.

## M3 Training Behavior

M3 point-projected uses pointwise cosine representation loss, not Gram relation loss. This was intentional after the Point+Gram probes indicated that Gram was not helping.

Latest M3 epoch-level values:

| Metric | Value |
| --- | ---: |
| stage2 epoch | 49 |
| total loss | 0.42990 |
| repr loss | 0.37185 |
| repr point loss | 0.37185 |
| repr relation loss | 0.0 |
| raw latent cosine mean | 0.64106 |
| EMA latent cosine mean | 0.67260 |
| raw source prediction preserved | 0.50195 |
| EMA source prediction preserved | 0.51367 |
| raw single-face eq1 | 1.0 |
| EMA single-face eq1 | 1.0 |
| projection applied fraction | 0.39936 |
| fm first-order effect mean | -0.03849 |
| repr descent inner product mean | 42.04735 |

The projection machinery is active. About 40% of monitored batches apply projection. The reported FM first-order effect is negative, so the projected representation step is not increasing FM loss to first order on average. But this local property has not produced useful global progress in latent cosine.

The fixed visual samples also show instability. At epoch 42, one sample had `latent_cosine = 0.0134` while the full 512-sample raw validation mean was much higher. This is not a global metric collapse, but it shows that individual conditional generations can move sharply across epochs.

## Why This Is Not Just a 5M Model Size Story

The 5M generator is probably too weak for final image quality, but the current evidence does not support using model size as the only explanation.

Evidence:

1. Stage1 and null-FM can keep single-face rates close to 1.0. The model has some face prior capacity.
2. Point-only CL-only reaches cosine 0.9536 and source preserved 0.8984. The representation target is reachable by parameter updates.
3. M0 reaches cosine 0.9238 and source preserved 0.8594 under weighted-sum training. The same small model can move toward E0 consistency, but it damages image quality.
4. M3 preserves face quality but stays near Stage1 cosine/source preserved. This points to the projected optimization path, not only to raw capacity.

A fairer statement is:

> The 5M model is too weak to give final-quality face generation, and the current projected-update rule is not strong enough to efficiently move this weak model along useful E0-preserving directions while keeping FM quality protected.

## What The Literature Does And Does Not Say About Model Scale

Primary diffusion/flow papers do not give a hard parameter-count threshold.

| Paper | Relevant point |
| --- | --- |
| DDPM, arXiv:2006.11239 | Reports models such as 35.7M on CIFAR-10 and larger U-Nets for LSUN/CelebA-HQ, but does not claim a minimum parameter count. |
| Flow Matching, arXiv:2210.02747 | Defines and tests the FM objective. It does not state a lower model-size bound. |
| Rectified Flow, arXiv:2209.03003 | Focuses on straighter flows and fewer sampling steps, not a parameter threshold. |
| Consistency Models, arXiv:2303.01469 | Focuses on few-step generation by learning consistency, not a minimum size rule. |
| InstaFlow, arXiv:2309.06380 | Uses large Stable-Diffusion-scale models and reports practical difficulty without strong initialization, but does not give a universal threshold. |
| Mean Flows, arXiv:2505.13447 | Uses much larger models than SAFA's 5M generator and directly targets one-step/few-step behavior. It does not claim a hard lower bound. |

So we should not write that DDPM, FM, or MeanFlow proves a model must exceed some size. The safe claim is practical:

> Mature diffusion/flow systems usually use far larger backbones and objectives designed for stable few-step sampling. SAFA's 5M FM is a useful prototype, but current results show it is not a strong enough final prior for publishable image quality.

## Current Problems

### 1. Projected update does not create effective CL progress

M3's raw cosine moved from Stage1's 0.6388 to 0.6411 after 49 epochs. EMA reaches 0.6726, but that is still far below the 0.95 guard.

This means the current projected update has not delivered the intended "FM and CL both make useful progress" behavior.

### 2. The local first-order guarantee is too weak for this global training problem

The projection guarantees that the representation step does not increase FM loss to first order at the current parameter point. It does not guarantee:

- large enough representation progress;
- monotonic validation cosine;
- stable per-sample E0 consistency;
- preservation of useful conditional controllability across many stochastic mini-batch updates.

This gap between the local guarantee and the global result is now visible in the metrics.

### 3. CL-only proves the target is reachable but unusable without a face constraint

Point-only CL-only reaches cosine 0.9536, but single-face eq1 is 0.0. This proves the representation objective can be optimized, but it also confirms that unconstrained CL leaves the face manifold.

M3 was meant to solve exactly this problem. It has not done so yet.

### 4. Gram relation loss is not justified by current experiments

Point+Gram CL-only is worse than Point-only CL-only on both cosine and source preserved in the latest snapshot:

| Probe | Cosine | Source preserved |
| --- | ---: | ---: |
| Point-only CL-only | 0.9536 | 0.8984 |
| Point+Gram CL-only | 0.8949 | 0.7617 |

This does not prove Gram is mathematically wrong, but it does show that the current batch Gram design is not useful enough to keep as the next main path.

### 5. M0 gives utility but destroys quality

M0 epoch100 has cosine 0.9238 and source preserved 0.8594, but FID is 126.254. It confirms that weighted-sum can force E0 consistency, but it also confirms that this route drags the model away from the face distribution.

### 6. The current prior is not publishable

Stage1 best collected FID is 49.216. Null-FM epoch120 FID is 80.233. M3 epoch40 FID is 72.870.

These are useful prototype signals, but they are not enough for a high-quality anonymized face dataset release.

## Interpretation

The current mathematical framework is not complete enough.

The projected two-step update is mathematically clean as a local first-order rule, but the experiment shows that local Euclidean projection alone is not enough to give efficient joint training. It protects FM better than weighted-sum, but it fails to move CL quickly. That means the feasible CL direction under the current FM guard is either too small, too noisy, or not aligned with validation utility.

This does not invalidate the whole SAFA idea. It shows that the current implementation of decoupled update is not yet the right optimizer for the problem.

## Recommended Next Direction

Recommend: move away from the 5M prototype as the main generator prior.

The next clean experiment should use a stronger and faster prior:

1. A larger mature FM/rectified-flow/MeanFlow-style model trained or adapted for face generation.
2. A frozen or mostly frozen pretrained prior with lightweight adapter tuning.
3. A null-condition or identity-agnostic prior path, so Stage1 does not learn direct `E0(X0) -> X0` reconstruction.

Alternative: continue theoretical work on a stronger constrained optimization method.

The current projected update should be treated as a failed or incomplete first implementation of the idea, not as proof that decoupled FM/CL optimization is impossible.

## What This Document Does Not Prove

This document does not prove that:

- 5M models cannot learn flow matching;
- DDPM/FM/MeanFlow require a fixed minimum parameter count;
- projected update is theoretically invalid;
- SAFA cannot work with a stronger prior;
- Point-only cosine is the final best representation loss.

It does show that the current M3 point-projected implementation has not achieved efficient FM/CL decoupling on the current 5M SAFA prototype.

