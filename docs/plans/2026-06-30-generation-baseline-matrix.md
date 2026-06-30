# Generation Baseline Matrix

Goal: complete the SAFA generation baseline matrix without touching the running E16 runtime config or starting a new formal training run.

## Matrix Shape

There are 12 report cells but only 8 unique checkpoints.

| Family | Size | Checkpoint config | Report cells |
| --- | --- | --- | --- |
| MeanFlow-SiT | B/2 | `configs/medium_v2/experiments/e19_meanflow_sit_b2_face_mixed_2400ep.yaml` | 1-step |
| MeanFlow-SiT | L/2 | `configs/medium_v2/experiments/e16_meanflow_sit_l2_face_mixed_2400ep.yaml` | 1-step |
| Rectified/FM-SiT | B/2 | `configs/medium_v2/experiments/e20_rectified_flow_sit_b2_face_mixed_2400ep.yaml` | 16-step Euler |
| Rectified/FM-SiT | L/2 | `configs/medium_v2/experiments/e21_rectified_flow_sit_l2_face_mixed_2400ep.yaml` | 16-step Euler |
| SiT-Diffusion | B/2 | `configs/medium_v2/experiments/e22_sit_diffusion_b2_face_mixed_2400ep.yaml` | DDIM-1, DDIM-16 |
| SiT-Diffusion | L/2 | `configs/medium_v2/experiments/e17_sit_diffusion_l2_face_mixed_2400ep.yaml` | DDIM-16 |
| Latent-Consistency | B/2 | `configs/medium_v2/experiments/e23_latent_consistency_b2_face_mixed_2400ep.yaml` | 1-step, 4-step |
| Latent-Consistency | L/2 | `configs/medium_v2/experiments/e18_latent_consistency_l2_face_mixed_2400ep.yaml` | 4-step |

DDIM-1 is an evaluation cell for a diffusion checkpoint. It is not a true one-step trained model.

## Model Notes

E16 stays the existing MeanFlow-SiT-L/2 1-step run. The untracked E16 runtime config is intentionally left out of git.

`rectified_flow_sit` is a real model type. It reuses the existing SiT backbone and trains the standard rectified-flow/FM target `x_noise - x_data`; it samples by integrating the learned velocity field with Euler or Heun steps.

E17 and E18 keep their existing L/2 diffusion and consistency settings. The added `eval_cells` entries are metadata for reporting and do not change training behavior.

## External Assets

Use `scripts/external/prepare_generation_baselines.sh` to clone reference repos and verify available MeanFlow-SiT weights. The script references:

- `https://github.com/Gsunshine/meanflow`
- `https://github.com/zhuyu-cs/MeanFlow`
- `https://github.com/facebookresearch/DiT`
- `https://github.com/openai/consistency_models`

Only MeanFlow-SiT B/2 and L/2 weights are auto-handled. The manifest at `docs/plans/2026-06-30-generation-baseline-weights-manifest.json` records the expected shapes. The runtime verifier writes `artifacts/manifests/generation_baseline_weights_manifest.json`.

Diffusion and consistency B/L do not have reliable public SAFA-compatible pretrained weights. They must not be marked as downloaded or present unless a real checkpoint is produced and verified.
