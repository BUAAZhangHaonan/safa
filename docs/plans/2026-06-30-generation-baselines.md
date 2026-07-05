# E17/E18 Generation Baselines

E17 and E18 are candidate null-conditioned face-prior baselines for SAFA Stage 1. They are executable configs, but they are not launched automatically.

- E17 uses a latent SiT-L/2 diffusion baseline with deterministic DDIM sampling. The first config uses 16 sampling steps and batch 16 because multi-step sampling keeps much more activation state than MeanFlow 1-step training.
- E18 uses a latent SiT-L/2 consistency-style baseline with 4 sampling steps. This v1 objective is an analytic x0 surrogate, not a full teacher-distilled LCM or consistency pair loss.
- Both configs reuse the E16 mixed face dataset, SD VAE latent path, learned null condition, EMA, validation, and quality-eval layout. They keep separate checkpoint and eval output directories.
- The L/2 ImageNet MeanFlow checkpoint path is recorded as a partial warm-start source. If a downstream loader cannot reuse the compatible subset, the run should treat it as a source note rather than a required exact resume state.
