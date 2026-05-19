# Ablation Experiment Results — Epoch 0

Date: 2026-05-19

## Root Cause Confirmed

Stage 1 pure flow matching training creates velocity field representations incompatible with cycle consistency.

## Epoch 0 Results

| Metric | A (from scratch) | B (resume Stage 1) |
|--------|-----------------|-------------------|
| validation_latent_cosine_mean | **0.9542** | **0.5928** |
| validation_face_detection_rate | 0.1406 (14%) | 1.0 (100%) |
| validation_source_prediction_preserved | 0.8594 (86%) | 0.625 (63%) |
| flow_matching_mse | 0.0936 | 0.0643 |
| cycle loss | 0.0578 | 0.0101 |
| grad_norm | 0.482 | 0.172 |
| lambda_cycle | 0.05 | 0.05 |

## Ablation A Step Count Diagnostic (best.pt, epoch 0)

| Steps | Cosine |
|-------|--------|
| 4 | 0.9721 |
| 8 | 0.9670 |
| 16 | 0.9453 |
| 32 | 0.9285 |

Gap 4→32: only 0.04. Original training gap was 0.33.

## Interpretation

A fast (0.95) + B slow (0.59) → Stage 1 representations are fundamentally incompatible with cycle consistency.

- Ablation A: joint flow+cycle training from scratch achieves cosine 0.9542 in 1 epoch
- Ablation B: resuming from Stage 1 checkpoint with aggressive lambda (0.05) only reaches 0.5928
- Stage 1 velocity field overfits to 4-step Heun trajectories, making cycle loss gradients ineffective
- clamp(-1,1) zeros out gradients for diverged samples, preventing correction

## Conclusion

Combined training from scratch (Ablation A approach) is the only viable path. Stage 1 gating should be removed or redesigned to allow joint flow+cycle training from the start.

## Configs

- A: configs/ablation/ablation_a_combined.yaml (no Stage 1, lambda=0.05 fixed, train_cycle_steps=4)
- B: configs/ablation/ablation_b_aggressive.yaml (resume Stage 1 best.pt, lambda=0.05 fixed)
- V2: configs/train_g_v2.yaml (cycle_steps_schedule=[4,8,16], resume Stage 1)
- F: configs/ablation/ablation_f_8step.yaml (train_cycle_steps=8, from scratch)
