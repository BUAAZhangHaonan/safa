# H100 DDP Generation Baseline Queue Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dry-run-first 4xH100 torchrun queue for generation baseline configs, with runtime YAML copies and pytest coverage.

**Architecture:** The queue is a Bash entrypoint that keeps the source experiment YAML files immutable. For each selected config, it creates a sibling `*_h100_ddp_runtime.yaml` file with H100 DDP runtime overrides, prints or runs a `torchrun --standalone` command, and records log, pid, and status files under `artifacts/`.

**Tech Stack:** Bash, `torchrun`, Python/PyYAML for YAML copy/update, and pytest subprocess tests.

---

### Task 1: Add failing tests for the H100 queue

**Files:**
- Create: `tests/test_h100_generation_baseline_ddp_queue.py`

**Step 1: Write failing tests**

Cover these behaviors:
- Default invocation is dry-run and prints each selected runtime config, batch values, torchrun command, log file, pid file, and status file.
- Runtime YAML is written without modifying the source YAML.
- B/2 configs E19/E20/E22/E23 use `per_device_batch_size: 32` and `global_batch_size: 128`.
- L/2 configs E17/E18/E21 use `per_device_batch_size: 16` and `global_batch_size: 64`.
- Runtime overrides set `device: cuda:0`, `distributed.backend: nccl`, `num_workers: 8`, `validation.batch_size: 16`, quality eval `output_dir` with `_h100_ddp`, `distribution_cuda_visible_devices: "0"`, and `distribution_device: cuda:0`.
- E16 process guard exits 2 unless `--skip-e16-check` is set.
- `--one` selects one config by experiment name or config path.
- Dry-run command includes `torchrun`, the configured nproc value, and the runtime YAML path.

**Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_h100_generation_baseline_ddp_queue.py -q
```

Expected: FAIL because `scripts/h100/run_generation_baseline_ddp_h100.sh` does not exist yet.

### Task 2: Implement the queue script

**Files:**
- Create: `scripts/h100/run_generation_baseline_ddp_h100.sh`

**Step 1: Add argument parsing**

Support:
- `--run`
- `--dry-run`
- `--repo-root PATH`
- `--python PATH`
- `--nproc-per-node N`
- `--timestamp VALUE`
- `--skip-e16-check`
- `--one NAME_OR_CONFIG`
- `-h` and `--help`

Defaults:
- dry-run mode
- repo root inferred from the script path
- Python from `PYTHON`, then the known H100 conda path
- `nproc_per_node=4`
- current timestamp
- E16 guard enabled

**Step 2: Add config metadata**

Queue order:
- `configs/medium_v2/experiments/e22_sit_diffusion_b2_face_mixed_2400ep.yaml`
- `configs/medium_v2/experiments/e23_latent_consistency_b2_face_mixed_2400ep.yaml`
- `configs/medium_v2/experiments/e19_meanflow_sit_b2_face_mixed_2400ep.yaml`
- `configs/medium_v2/experiments/e20_rectified_flow_sit_b2_face_mixed_2400ep.yaml`
- `configs/medium_v2/experiments/e17_sit_diffusion_l2_face_mixed_2400ep.yaml`
- `configs/medium_v2/experiments/e18_latent_consistency_l2_face_mixed_2400ep.yaml`
- `configs/medium_v2/experiments/e21_rectified_flow_sit_l2_face_mixed_2400ep.yaml`

Use B/2 batch `32/128` for E19/E20/E22/E23. Use L/2 batch `16/64` for E17/E18/E21.

**Step 3: Generate runtime YAML**

Use a small Python/PyYAML helper from the Bash script. It must:
- Read the source YAML.
- Write `configs/medium_v2/experiments/<name>_h100_ddp_runtime.yaml`.
- Preserve all paths unless the requested runtime fields need changes.
- Set the exact runtime override fields from Task 1.
- Add `_h100_ddp` to `stages.stage2.quality_eval.output_dir`.

**Step 4: Implement dry-run output**

For every selected config, print:
- `runtime_config`
- `batch: per_device=... global=...`
- `command: torchrun --standalone --nproc_per_node=... -m safa.cli.train_g --config ...`
- `log`
- `pid`
- `status`

Dry-run still writes runtime YAML so the user can inspect it.

**Step 5: Implement run mode**

In `--run` mode:
- Create log and run artifact directories.
- Run selected configs sequentially.
- For each config, write the process pid to the pid file.
- Wait for completion.
- Write the exit code to the status file.
- Stop at the first non-zero exit code.

### Task 3: Verify and adjust

**Step 1: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_h100_generation_baseline_ddp_queue.py -q
```

Expected: PASS.

**Step 2: Run related queue tests**

Run:

```bash
python3 -m pytest tests/test_generation_baseline_queue.py tests/test_gpu6_generator_queue.py tests/test_h100_generation_baseline_ddp_queue.py -q
```

Expected: PASS.

**Step 3: Final audit**

Check:
- Only allowed files are changed.
- No real training was started.
- Source config YAML files were not changed.
- Runtime YAML files are generated only by script/test execution and are not part of the requested tracked edits.
