# Mature Generation Baselines

Date: 2026-07-02

## Goal

Separate paper main-table mature generation baselines from internal ablation runs, and provide restartable launch scripts for the currently executable mature MeanFlow-SiT family.

## Matrix Layers

### Main-table mature baselines

These are baselines that should be reported as mature, named model families. They need stable configs, known weights, checkpoint resume, and launch scripts that do not mix in exploratory variants.

Current executable subset:

| Experiment | Family | Size | Status | Script coverage |
| --- | --- | --- | --- | --- |
| E16 | MeanFlow-SiT | L/2 | K100 current/resumable | K100 default, H100 opt-in |
| E19 | MeanFlow-SiT | B/2 | Mature baseline | K100 default, H100 default |

Launchers:

```bash
bash scripts/k100/run_mature_generation_baselines_k100.sh --dry-run
bash scripts/k100/run_mature_generation_baselines_k100.sh --run
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
