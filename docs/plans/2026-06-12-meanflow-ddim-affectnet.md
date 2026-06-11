# MeanFlow and DDIM AffectNet Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace or compare the current small 5M Flow Matching generator with mature one-step candidates, first MeanFlow and then a DDIM 1-step baseline, trained on AffectNet for 200 epochs on physical GPU 6 only.

**Architecture:** Keep the current SAFA training and evaluation surface stable, then add new generator backends behind `build_generator`. MeanFlow should be ported as a native PyTorch core with a 1-NFE sampler. DDIM must be implemented as a separate diffusion denoiser, training loss, noise schedule, and DDIM sampler because DDIM is a sampler, not a standalone model.

**Tech Stack:** PyTorch, SAFA generator/training/evaluation code, AffectNet feature-aligned dataset, TensorBoard/JSONL logs, FID/KID/NIQE quality metrics, single-GPU CUDA execution through `CUDA_VISIBLE_DEVICES=6`.

---

## Scope and Completion Criteria

This plan covers the full experiment track. It must be executed as small commits on the existing remote `master` workspace. Do not create a new branch or worktree.

Completion criteria:
- MeanFlow and DDIM 1-step baselines can be built through the project generator factory.
- Both baselines support AffectNet labels and a null condition path.
- Both baselines have focused tests, smoke runs, 200 epoch launch commands, logs, checkpoints, samples, and metrics.
- All training, sample generation, quality evaluation, FID/KID subprocesses, and report jobs use physical GPU 6 only.
- Each small node below is committed and pushed separately.

## Current Project Facts

Core generator files:
- `src/safa/models/generator.py` defines `FlowGeneratorConfig`, `ConditionalFlowGenerator`, the current vector-field UNet, `flow_matching_loss`, `sample`, and `build_generator`.
- `src/safa/training/g_loop.py` owns the generator training loop, flow loss calls, representation/cycle losses, validation, quality hooks, logging, checkpoints, and metrics history.
- `src/safa/data/feature_dataset.py` defines `FeatureAlignedAffectNet`, which returns image tensors, feature `z`, label, and sample id for generator training.

Current FM capacity facts:
- The current small FM uses `base_channels: 32` and has about `5,004,291` parameters.
- Raising the same current FM family to `base_channels: 64` gives about `15,730,691` parameters.
- The current generator factory only has the existing conditional flow matching path, so MeanFlow and DDIM should be added as explicit model types rather than hidden changes to the old FM behavior.

Current risk points:
- Existing config files may already be dirty in the remote worktree. Do not stage or rewrite unrelated files.
- Existing quality evaluation configs may contain `quality_eval.distribution_cuda_visible_devices: "0"`. That is unsafe for this task because all subprocesses must map to physical GPU 6.

## Key Technical Judgments

MeanFlow:
- Port the PyTorch core directly into this repo instead of running the official JAX repository as an external training stack.
- Implement the needed pieces natively: `t/r` sampling, mean-velocity target, JVP-based loss, adaptive weighting if used, EMA support, and strict 1-NFE sampling.
- Keep the public generator interface compatible with the training loop where possible: `sample(...)`, loss method, config fields, checkpoint load through `build_generator`.

DDIM:
- Treat DDIM as the 1-step sampling baseline for a diffusion denoiser.
- Do not connect the existing FM vector field directly to a DDIM sampler. That would mix incompatible training targets.
- Add a diffusion denoiser, beta/alpha schedule, epsilon or v-prediction training loss, and DDIM sampling path. The scheduler prediction type must match the training target.

## Null Embedding Design

The condition path should support both real AffectNet labels and null conditioning.

Recommended implementation:
- Use an extra embedding row, for example 8 AffectNet classes plus `null_id = 8`, or a separate learned `nn.Parameter` null embedding if the model condition API needs a standalone tensor.
- During training, use class dropout such as `p_uncond = 0.1` and replace a random subset of labels with the learned null condition.
- During sampling, pass either the target label condition or the null condition through the normal model interface. Do not introduce a sampler-only shortcut that bypasses the model condition path.
- If CFG is later added for DDIM, train with null dropout first. Then run conditional and unconditional denoiser calls consistently and compute `eps = eps_uncond + scale * (eps_cond - eps_uncond)`.
- For strict MeanFlow 1-NFE reporting, prefer direct target-condition sampling and do not double the forward passes with CFG in the main metric table.

## Node Plan and Commit Boundaries

Each node must end with its own commit and push. Keep the commit scope exact. Do not batch unrelated code into one commit.

### Node 1: Condition and Null Embedding Interface

Files:
- Modify: `src/safa/models/generator.py`
- Add or modify focused tests under `tests/`
- Add config fields only where needed for condition dropout/null id

Work:
1. Add a shared condition embedding helper or generator-local helper that maps AffectNet labels to embeddings and supports a learned null condition.
2. Add config fields for `num_classes`, `null_condition`, and `class_dropout_prob`.
3. Add tests for real labels, null labels, dropout behavior, device placement, and checkpoint round trip.
4. Commit message: `feat(generator): add learned null condition support`.

Validation:
```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests -k "null or condition or generator" -v
```

### Node 2: MeanFlow Core

Files:
- Modify: `src/safa/models/generator.py` or create `src/safa/models/meanflow_generator.py`
- Modify: `src/safa/models/__init__.py` if needed
- Add focused tests under `tests/`

Work:
1. Add a PyTorch MeanFlow generator backend with a larger-capacity backbone than the current 5M FM.
2. Implement `sample_t_r`, `h = t - r`, the MeanFlow loss, JVP path, adaptive weighting, and EMA-compatible state dict behavior.
3. Add strict 1-step sampling through the generator `sample(...)` interface.
4. Register the backend in `build_generator` with an explicit model type such as `meanflow`.
5. Commit message: `feat(generator): add meanflow one step backend`.

Validation:
```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests -k "meanflow or generator" -v
```

### Node 3: MeanFlow Config and Tests

Files:
- Add: `configs/medium_v2/experiments/e9_meanflow_affectnet_200ep.yaml`
- Add or modify tests that load the config

Work:
1. Create an AffectNet 200 epoch MeanFlow config based on the current generator training config style.
2. Set `model_type: meanflow`, 1-step sampling, learned null condition, class dropout, EMA, and quality eval output paths.
3. Set every CUDA mapping in the config to physical GPU 6 or remove unsafe overrides that point at GPU 0.
4. Add a config smoke test that instantiates the model and dataloaders without starting a full run.
5. Commit message: `config(meanflow): add affectnet 200 epoch experiment`.

Validation:
```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests -k "meanflow and config" -v
```

### Node 4: MeanFlow Train Launch

Files:
- Add: `scripts/launch_meanflow_affectnet_gpu6.sh` or document an existing tmux command
- Add training log path under the experiment output convention

Work:
1. Launch the 200 epoch MeanFlow training on physical GPU 6 only.
2. Keep stdout/stderr in `artifacts/logs/`.
3. Confirm checkpoints, `metrics_history.jsonl`, `last_metrics.json`, samples, and quality outputs are being written.
4. Push the launch script or config/logging fix, not large checkpoints.
5. Commit message: `chore(meanflow): add gpu6 launch command`.

Launch template:
```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=src \
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
-m safa.cli.train_g \
--config configs/medium_v2/experiments/e9_meanflow_affectnet_200ep.yaml
```

Validation:
```bash
tail -n 80 artifacts/logs/<meanflow-log>.log
```

### Node 5: DDIM Schedule and Sampler

Files:
- Add: `src/safa/models/diffusion_schedule.py`
- Add: `src/safa/models/ddim_sampler.py`
- Add focused tests under `tests/`

Work:
1. Implement beta schedule, alpha products, timestep selection, and deterministic DDIM update with `eta = 0`.
2. Support `num_inference_steps = 1` as the main baseline path.
3. Make prediction type explicit: `epsilon` first, `v_prediction` only if fully wired.
4. Add shape, dtype, boundary timestep, and deterministic sampling tests.
5. Commit message: `feat(diffusion): add ddim schedule and sampler`.

Validation:
```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests -k "ddim or schedule" -v
```

### Node 6: DDIM Denoiser and Loss

Files:
- Add: `src/safa/models/ddim_generator.py` or equivalent
- Modify: `src/safa/models/generator.py` for `build_generator` registration
- Add focused tests under `tests/`

Work:
1. Add a conditional diffusion denoiser with higher capacity than the 5M FM baseline.
2. Train it with epsilon prediction loss and the same AffectNet condition/null interface.
3. Expose `diffusion_loss(...)` or adapt the training loop with an explicit objective branch that does not reuse FM velocity loss.
4. Implement `sample(...)` using the DDIM sampler and report the main result at 1 NFE.
5. Commit message: `feat(generator): add conditional ddim baseline`.

Validation:
```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests -k "ddim or diffusion or generator" -v
```

### Node 7: DDIM Config and Tests

Files:
- Add: `configs/medium_v2/experiments/e10_ddim_affectnet_200ep.yaml`
- Add or modify config tests

Work:
1. Create the 200 epoch DDIM AffectNet config.
2. Set `num_train_timesteps: 1000`, `prediction_type: epsilon`, `ddim_num_inference_steps: 1`, `eta: 0`, learned null condition, and class dropout.
3. Force all quality eval and metric subprocess device settings through physical GPU 6.
4. Add config instantiation and smoke tests.
5. Commit message: `config(ddim): add affectnet 200 epoch experiment`.

Validation:
```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests -k "ddim and config" -v
```

### Node 8: DDIM Train Launch

Files:
- Add: `scripts/launch_ddim_affectnet_gpu6.sh` or document an existing tmux command

Work:
1. Launch the 200 epoch DDIM training on physical GPU 6 only.
2. Keep logs in `artifacts/logs/`.
3. Confirm checkpoints, metrics history, sample grids, and quality eval outputs.
4. Push the launch script or config/logging fix, not large checkpoints.
5. Commit message: `chore(ddim): add gpu6 launch command`.

Launch template:
```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=src \
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
-m safa.cli.train_g \
--config configs/medium_v2/experiments/e10_ddim_affectnet_200ep.yaml
```

Validation:
```bash
tail -n 80 artifacts/logs/<ddim-log>.log
```

### Node 9: Metrics, Visualization, and Report

Files:
- Modify or add scripts under `scripts/` only as needed
- Add report under `docs/experiments/`

Work:
1. Generate fixed-seed sample grids with rows for real labels and null condition.
2. Compute FID, KID, and NIQE with the existing quality script.
3. Run the existing eval runner for representation/privacy metrics.
4. Add per-class metric summaries when available and include 1-NFE latency/images-per-second.
5. Compare current 5M FM, MeanFlow 1-step, and DDIM 1-step under the same AffectNet split and image size.
6. Commit message: `docs(eval): report meanflow and ddim affectnet results`.

Validation templates:
```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=src \
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
scripts/eval_generation_quality.py \
--config configs/medium_v2/experiments/e9_meanflow_affectnet_200ep.yaml
```

```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=src \
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
-m safa.cli.eval \
--config <eval-config-for-meanflow-or-ddim>
```

## GPU 6 Constraint

All commands must use physical GPU 6 only:
```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python <entrypoint>
```

Rules:
- In single-process training, project configs may keep `device: cuda:0` because `CUDA_VISIBLE_DEVICES=6` maps physical GPU 6 to logical `cuda:0`.
- Do not launch with `CUDA_VISIBLE_DEVICES=0` or multi-GPU defaults.
- Do not use `scripts/run_train_g_tmux.sh` defaults unless `SAFA_CUDA_VISIBLE_DEVICES=6` and `NPROC_PER_NODE=1` are set.
- Fix or override `quality_eval.distribution_cuda_visible_devices: "0"` because it can make FID/KID subprocesses use physical GPU 0 instead of physical GPU 6.
- Apply the same GPU rule to sample generation, FID/KID/NIQE, visualization watchers, and eval runner jobs.

Safe tmux template:
```bash
SAFA_CUDA_VISIBLE_DEVICES=6 \
NPROC_PER_NODE=1 \
scripts/run_train_g_tmux.sh \
--config <config-path>
```

## Verification Checklist

For each code node:
- Run the narrowest unit tests first.
- Run a smoke train with a tiny subset or very small step count before the full 200 epoch run.
- Confirm the model can save and reload through `build_generator` and checkpoint state dicts.
- Confirm `sample(..., steps=1)` produces tensors with valid shape and finite values.
- Confirm null-condition and real-condition sampling use the same public condition interface.

For each full training run:
- Check `artifacts/logs/<run>.log` for epoch progress, loss values, GPU id, memory, validation, and no NaN.
- Check checkpoint outputs: `last.pt`, `best.pt` or `best_stage2.pt`, `manifest.json`, `last_metrics.json`, and `metrics_history.jsonl`.
- Check visual outputs: fixed-seed sample grid, per-class grid, null row, failure samples, and nearest-real or memorization check if available.
- Check metrics: FID, KID, NIQE, existing eval runner result, representation metrics, and 1-NFE throughput.

## Command Templates

Train MeanFlow:
```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=src \
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
-m safa.cli.train_g \
--config configs/medium_v2/experiments/e9_meanflow_affectnet_200ep.yaml
```

Train DDIM:
```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=src \
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
-m safa.cli.train_g \
--config configs/medium_v2/experiments/e10_ddim_affectnet_200ep.yaml
```

Run focused tests:
```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests -k "<keyword>" -v
```

Run quality metrics:
```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=src \
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
scripts/eval_generation_quality.py \
--config <config-path>
```

Run eval runner:
```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=src \
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
-m safa.cli.eval \
--config <eval-config-path>
```

## Pre-Commit Guardrails

Before every commit:
```bash
git status -sb
git diff --cached --name-only
```

Required result:
- `git diff --cached --name-only` contains only the files for the current node.
- Existing user-modified config files stay unstaged unless that exact node intentionally owns them.
- No generated checkpoints, image dumps, metric caches, or logs are committed unless they are small report artifacts explicitly intended for docs.
