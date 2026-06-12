# MeanFlow-SiT Stage 1 Setup Record

Date: 2026-06-13

## Decision

- Training host: K100, host name `k100-X785-H30`.
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, 97887 MiB VRAM, idle at setup time.
- Driver / CUDA: NVIDIA driver `590.48.01`, system CUDA `13.1.2`.
- Disk: root filesystem 3.6T total, about 1.6T free.
- Conda root: `/home/k100/miniconda3`.
- Usable PyTorch environments:
  - `pt210_cu130_fa4`: Python 3.12.12, Torch 2.10.0+cu130, CUDA available.
  - `cu131`: Python 3.13.12, Torch 2.9.1+cu128, CUDA available.
  - `py311`: Python 3.11.14, Torch 2.9.0+cu128, CUDA available.
- Broken environment: `pt24` imports Python 3.10 but Torch fails with missing `libgalaxyhip.so.5`.

The Stage 1 MeanFlow run should use K100 first. The reason is simple: MeanFlow training uses JVP and a larger SiT/DiT backbone, so single-card memory is the main bottleneck. A single 96GB RTX PRO 6000 is safer than 4x3090 for the first integration pass.

## Repository Setup

- SAFA workspace: `/home/k100/projects/samplewise-affective-face-anonymization`.
- SAFA remote: `git@github.com:BUAAZhangHaonan/safa.git`.
- SAFA branch: `master`.
- Initial K100 clone HEAD for this setup: `794b39a6c2f68cc1f2842360441d46cb7b9c5426`.

Reference repositories are kept outside the SAFA repository. They must not be vendored or committed into SAFA.

| Purpose | Repository | Commit | Local path |
| --- | --- | --- | --- |
| Main PyTorch reference | `git@github.com:zhuyu-cs/MeanFlow.git` | `40a766689c8bccad61f2993ee6905630ac030ff3` | `/home/k100/projects/meanflow_refs/zhuyu-cs-MeanFlow` |
| Official formula reference | `git@github.com:Gsunshine/meanflow.git` | `d70cb55d298ee03c53bf6da67bec281082e4e2d9` | `/home/k100/projects/meanflow_refs/Gsunshine-meanflow` |
| PyTorch JVP/DDP reference | `git@github.com:Gsunshine/py-meanflow.git` | `1f6d72d94247c8fdeb89489acca1a8007a6baf6c` | `/home/k100/projects/meanflow_refs/Gsunshine-py-meanflow` |

## Implementation Direction

- Use a SAFA-native PyTorch implementation. Do not merge JAX dependencies into the SAFA conda environment.
- Use `zhuyu-cs/MeanFlow` as the main implementation reference for SiT/DiT MeanFlow training.
- Use `Gsunshine/meanflow` only to check the official math and sampling direction.
- First long run target: SiT-B based MeanFlow prior, fine-tuned on AffectNet for Stage 1.
- Use pretrained MeanFlow/SiT weights if the checkpoint can be downloaded and loaded cleanly.
- Stage 1 must be null-conditioned only. It must not use `Z0` as a condition and must not train paired `Z0 -> x0` reconstruction.
- `Z0` injection belongs to Stage 2 through an adapter or condition path.

## E9 Direction Fix Requirement

The previous E9 MeanFlow run used the wrong direction between training and one-step sampling. This is a fatal issue and must be fixed before any new MeanFlow training.

The implementation must follow the official MeanFlow direction:

```text
z_t = (1 - t) * x + t * eps
v   = eps - x
x   = eps - u(eps, r = 0, t = 1)
```

The next code task must add oracle tests for these equations before changing the training path.

## Next Tasks

1. Add MeanFlow oracle tests for path endpoints, target velocity, and one-step sampling sign.
2. Fix the current MeanFlow direction bug in SAFA.
3. Add `model_type: meanflow_sit` with SiT-B backbone and null-conditioned Stage 1 support.
4. Add checkpoint loading for the selected pretrained MeanFlow/SiT weights.
5. Prepare AffectNet latent/cache support if the selected reference path trains in latent space.
6. Run only small smoke tests before any 200 epoch training.
7. After smoke tests pass, start the K100 200 epoch Stage 1 run with full logs, metrics, and visualizations.

## Task 1 Boundary

This setup task did not start training, did not vendor external repositories into SAFA, and did not create any extra branch.
