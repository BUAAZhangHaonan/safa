# Mature Generation Baselines

Date: 2026-07-02

## Goal

Separate paper main-table mature generation baselines from internal ablation runs, and provide restartable launch scripts for the currently executable mature MeanFlow-SiT family.

## Matrix Layers

## Baseline Taxonomy

The matrix must keep three levels separate:

1. **Paradigm or route**: GAN, diffusion/score-based latent diffusion, flow matching/interpolant flow, and diffusion acceleration/consistency distillation.
2. **Representative model family**: StyleGAN2/3, DDPM/DDIM or SDXL, Flow Matching/SiT, MeanFlow, LCM, SDXL-Lightning, Hyper-SD, DMD2.
3. **Concrete checkpoint and sampling budget**: for example `stylegan2-ffhq-1024x1024.pkl`, `stabilityai/stable-diffusion-xl-base-1.0` at 20/40 steps, `ByteDance/SDXL-Lightning` at 1/2/4/8 steps, or SAFA E16/E19 MeanFlow-SiT at 1-NFE.

Do not compare a backbone, a sampler, and a concrete model as if they were the same object. `DiT` and `SiT` describe transformer/interpolant model families or backbones. `DDIM` is a sampler or implicit sampling route for a diffusion checkpoint. `SDXL`, `StyleGAN`, and `MeanFlow` are representative model families, and the checkpoint path is one level lower.

### Main Experimental Axes

| Paradigm / route | Multi-step representative | One-step or few-step representative | Role in SAFA |
| --- | --- | --- | --- |
| GAN | StyleGAN2/3 face prior, one generator forward | Same generator forward, optionally W/W+ adapter | Frozen prior + adapter baseline |
| Diffusion / latent diffusion | DDPM/DDIM or SDXL with standard sampler steps | SDXL-Turbo, SDXL-Lightning, LCM, Hyper-SD, or DMD2 | Frozen prior + adapter baseline first; trainable version only if protocol is matched |
| Flow matching / interpolant flow | Flow Matching or SiT-style multi-step sampler | MeanFlow-SiT 1-NFE | Trainable SAFA prior baseline |
| Consistency | Standalone consistency training if implemented under the same face-data protocol | LCM-style or distilled consistency if using SDXL weights | Usually diffusion-acceleration subgroup, not a separate top-level route in the first batch |

The first paper-quality table should report two tracks instead of forcing all methods into one training protocol:

- **Trainable prior track**: DDPM/DDIM-style latent diffusion, Flow Matching/SiT, MeanFlow-SiT, and optionally standalone Consistency Training. These must use the same face data, resolution, VAE latent space, and Stage 1/Stage 2 budget.
- **Frozen prior + adapter track**: StyleGAN2/3 FFHQ, SDXL-base, SDXL-Turbo or LCM/Lightning/Hyper-SD/DMD2. These keep the mature generator mostly frozen and train only the adapter or condition path for SAFA Stage 2.

### First-Batch Main Table Recommendation

| Track | Paradigm | Representative model | Concrete candidate | Step budget |
| --- | --- | --- | --- | --- |
| Frozen + adapter | GAN | StyleGAN2-ADA or StyleGAN3 | FFHQ pretrained `.pkl` | 1 generator forward |
| Frozen + adapter | Latent diffusion | SDXL-base | `stabilityai/stable-diffusion-xl-base-1.0` | 20/40 steps |
| Frozen + adapter | Diffusion acceleration | SDXL-Lightning or LCM-SDXL | official 4-step checkpoint or LoRA | 2/4 steps first; 1-step optional |
| Trainable prior | Diffusion | DDPM/DDIM latent UNet or DiT | same face mixed dataset, same VAE | 16/32 steps and DDIM curve |
| Trainable prior | Flow matching | Flow Matching / SiT | same face mixed dataset, same VAE | 16/32 ODE steps |
| Trainable prior | One-step flow | MeanFlow-SiT | E16 L/2 and E19 B/2 | 1-NFE |

This keeps the experiment broad enough for a general vision venue while still making the one-step question testable: the key comparison is not only final FID, but Stage 2 time-to-quality, NFE/latency, stability, face pass rate, and representation preservation under matched quality constraints.

### Main-table mature baselines

These are baselines that should be reported as mature, named model families. They need stable configs, known weights, checkpoint resume, and launch scripts that do not mix in exploratory variants.

Current executable subset:

| Experiment | Family | Size | Status | Script coverage |
| --- | --- | --- | --- | --- |
| E16 | MeanFlow-SiT | L/2 | K100 current/resumable | K100 default, H100 opt-in |
| E19 | MeanFlow-SiT | B/2 | Mature baseline | K100 opt-in, H100 default |

Launchers:

```bash
bash scripts/k100/run_mature_generation_baselines_k100.sh --dry-run
bash scripts/k100/run_mature_generation_baselines_k100.sh --run
bash scripts/k100/run_mature_generation_baselines_k100.sh --include-e19 --run
bash scripts/h100/run_mature_generation_baselines_ddp_h100.sh --dry-run
bash scripts/h100/run_mature_generation_baselines_ddp_h100.sh --run
bash scripts/h100/run_mature_generation_baselines_ddp_h100.sh --include-e16 --run
```

Both launchers write runtime YAMLs and auto-resume from `out_dir/last.pt` by setting:

```yaml
resume_from: <out_dir>/last.pt
resume_mode: training_state
resume_optimizer_state: true
```

If `last.pt` is absent, they keep the source warm-start mode:

```yaml
resume_from: ''
resume_mode: model_weights_only
resume_optimizer_state: false
```

### Internal ablations

E17/E18/E20/E21/E22/E23 are internal ablations. They compare sampler or training-objective variants and are not the paper main-table mature baseline layer.

The old queue scripts now require explicit acknowledgement:

```bash
bash scripts/k100/run_generation_baseline_queue.sh --ablation-only --dry-run
bash scripts/h100/run_generation_baseline_ddp_h100.sh --ablation-only --dry-run
```

Without `--ablation-only`, these scripts refuse to dry-run or run.

## Layering Rule

Do not put MeanFlow, DiT/SiT diffusion, rectified flow, and latent consistency at the same reporting level just because the configs share a dataset or a SiT backbone.

MeanFlow-SiT E16/E19 is the first mature executable family because the official MeanFlow-SiT B/2 and L/2 weights are already wired into the repo and have resumable checkpoints. Other official mature models should be introduced in a later batch only after their weights, config contract, runtime resume path, and report cells are verified as stable.
