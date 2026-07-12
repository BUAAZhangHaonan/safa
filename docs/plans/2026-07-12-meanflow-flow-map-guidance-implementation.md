# MeanFlow Flow-Map Representation Guidance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the frozen Stage 2 epoch-1652 MeanFlow SiT-B/4 EMA can improve full-Z0 cosine without losing FID, Sharpness, KID, or visible image quality.

**Architecture:** Expose the trained two-time MeanFlow map as a differentiable model API. Build three frozen-model routes on top of it: a semigroup diagnostic, direct-autograd FMRG-J, and projected initial-noise optimization. Generate matched images with stable sample noise, evaluate every arm with the same 2048-image protocol, and launch independent arms concurrently on physical GPUs 0-3.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, diffusers AutoencoderKL, existing SAFA E0 and feature cache, torchmetrics FID/KID, pyIQA NIQE, OpenCV Sharpness, pytest, Ruff, YAML, nvidia-smi.

---

## Fixed Contracts

Use these paths in every config and test command:

```text
repo=/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
python=/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python
checkpoint=artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last_nopretrained.pt
e0=artifacts/checkpoints/e0_medium_v1/best.pt
vae=artifacts/checkpoints/external/sd-vae-ft-ema
index=data/index/val_face_mixed_e14.jsonl
features=artifacts/e0_features/val_face_mixed_e14_e0_medium_v1
```

Use `ema_model_state_dict`, learned null condition for every flow-map call, `sampling_seed=1337`, and full cached `Z0` only as the representation target. No task in this plan updates model weights.

The implementation must not use `find /` or scan outside the repository and the explicitly reviewed FMRG repository. Use `rg`, `rg --files`, and explicit paths.

## Completion Criteria

1. Native `sample` remains numerically identical after the flow-map API refactor.
2. The checkpoint loader rejects any checkpoint other than Stage 2 epoch 1652 MeanFlow SiT-B/4 with EMA weights.
3. Semigroup, FMRG-J, and noise-oracle unit tests pass, including exact NFE assertions and frozen-weight assertions.
4. FID, KID, NIQE, Sharpness, E0 cosine, NFE, speed, and peak VRAM are recorded for every completed full arm.
5. The 64 fixed visual comparisons are opened and reviewed directly; face detection is not used as the collapse decision.
6. Calibration and full processes use physical GPUs 0, 1, 2, and 3 concurrently.
7. Every code/config/test unit is committed and pushed as a fine-grained commit. Generated result docs are committed separately.

## Task 1: Add the General MeanFlow Map API

**Files:**
- Modify: `tests/test_meanflow_sit_generator.py`
- Modify: `src/safa/models/meanflow_sit.py:324-369`

### Step 1: Write failing direct-equivalence and validation tests

Add these tests to `tests/test_meanflow_sit_generator.py`:

```python
def test_meanflow_flow_map_1_to_0_matches_native_sample() -> None:
    generator = build_generator(_tiny_meanflow_sit_config())
    z = torch.randn(2, 16)
    x_init = torch.randn(2, 3, 16, 16)

    expected = generator.sample(z, x_init=x_init, clamp_output=False)
    actual = generator.flow_map(x_init, z, t=1.0, r=0.0)

    assert torch.equal(actual, expected)


def test_meanflow_flow_map_accepts_per_sample_times_and_input_gradient() -> None:
    generator = build_generator(_tiny_meanflow_sit_config())
    z = torch.randn(2, 16)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    t = torch.tensor([1.0, 0.75])
    r = torch.tensor([0.5, 0.25])

    output = generator.flow_map(x, z, t=t, r=r)
    output.square().mean().backward()

    assert tuple(output.shape) == tuple(x.shape)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    ("t", "r", "message"),
    [
        (-0.1, 0.0, "within [0,1]"),
        (1.1, 0.0, "within [0,1]"),
        (0.25, 0.5, "r <= t"),
    ],
)
def test_meanflow_flow_map_rejects_invalid_interval(t, r, message) -> None:
    generator = build_generator(_tiny_meanflow_sit_config())
    with pytest.raises(ValueError, match=re.escape(message)):
        generator.flow_map(torch.randn(2, 3, 16, 16), torch.randn(2, 16), t=t, r=r)
```

Add one test that a time tensor with the wrong batch length fails clearly.

### Step 2: Run the focused tests and verify failure

```bash
cd /home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_sit_generator.py -q
```

Expected: the new tests fail because `_MeanFlowSiTGenerator.flow_map` does not exist.

### Step 3: Implement the minimal general map

Add a public method with this contract:

```python
def flow_map(self, x, z, *, t, r):
    self._validate_z(z)
    self._validate_x_init(x, z)
    t_batch = self._expand_flow_time("t", t, z)
    r_batch = self._expand_flow_time("r", r, z)
    if torch.any(r_batch > t_batch):
        raise ValueError("MeanFlow flow_map requires r <= t for every sample")
    horizon = (t_batch - r_batch).view(-1, 1, 1, 1)
    velocity = self.vector_field(x, r_batch, t_batch, z)
    return x - horizon * velocity
```

`_expand_flow_time` must accept a Python number, a scalar tensor, or a `[B]` tensor; move it to `z.device/z.dtype`; require finite values in `[0,1]`; and reject any other shape. Do not clamp invalid inputs.

Refactor `sample` to call:

```python
x = self.flow_map(x, z, t=1.0, r=0.0)
```

Keep the existing null-condition DDP graph term and `_model_to_data_space` behavior unchanged. `flow_map` returns a model-space state and does not clamp or decode it.

### Step 4: Run focused and adjacent tests

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_sit_generator.py \
  tests/test_meanflow_sit_latent_training.py \
  tests/test_meanflow_sit_attention.py -q
```

Expected: all pass, including exact native-sample equality.

### Step 5: Commit and push

```bash
git add src/safa/models/meanflow_sit.py tests/test_meanflow_sit_generator.py
git commit -m "feat(meanflow): expose two-time flow map"
git push origin master
```

## Task 2: Implement Frozen-Stack and Semigroup Primitives

**Files:**
- Create: `src/safa/guidance/__init__.py`
- Create: `src/safa/guidance/meanflow_flow_map.py`
- Create: `tests/test_meanflow_flow_map_guidance.py`

### Step 1: Write failing tests for freezing, NFE, and semigroup metrics

Create a tiny deterministic fake flow map and identity codec/E0 in the test file. Add tests with these names:

```text
test_freeze_guidance_stack_disables_parameter_gradients
test_counted_flow_map_counts_one_nfe_per_call
test_semigroup_probe_returns_zero_for_exact_semigroup
test_semigroup_probe_reports_each_requested_split
test_semigroup_probe_rejects_unsorted_or_boundary_split
test_semigroup_relative_residual_is_finite_for_zero_endpoints
```

The exact-semigroup fake should implement `Phi_{t->r}(x) = exp(-(t-r)) * x`, so direct and composed endpoints are equal within floating-point tolerance.

### Step 2: Run and verify failure

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_flow_map_guidance.py -q
```

Expected: import failure because `safa.guidance.meanflow_flow_map` does not exist.

### Step 3: Add explicit result types and counter

Implement these public types and functions:

```python
@dataclass(frozen=True)
class GuidanceResult:
    latent: torch.Tensor
    nfe: int
    diagnostics: dict[str, torch.Tensor | float | int | list[float]]


class CountedFlowMap:
    def __init__(self, generator): ...
    def __call__(self, x, z, *, t, r): ...


def freeze_guidance_stack(generator, codec, e0) -> None: ...
def assert_guidance_stack_frozen(generator, codec, e0) -> None: ...
def symmetric_relative_l2(left, right, eps=1.0e-8) -> torch.Tensor: ...
def semigroup_probe(flow_map, x_init, condition, split_times) -> dict: ...
```

`CountedFlowMap` increments once for each vector-field evaluation, not once per image. `semigroup_probe` must return the direct endpoint, each composed endpoint, per-sample residuals, and total NFE. It must not decode or compute FID.

### Step 4: Run focused tests

Use the command from Step 2. Expected: all pass.

### Step 5: Commit and push

```bash
git add src/safa/guidance tests/test_meanflow_flow_map_guidance.py
git commit -m "feat(guidance): add MeanFlow semigroup probe"
git push origin master
```

## Task 3: Implement Direct-Autograd FMRG-J

**Files:**
- Modify: `src/safa/guidance/meanflow_flow_map.py`
- Modify: `tests/test_meanflow_flow_map_guidance.py`

### Step 1: Write failing algorithm tests

Add tests with these exact requirements:

```text
test_fmrg_j_uses_two_nfe_per_guided_interval_plus_tail
test_fmrg_j_finishes_with_explicit_unguided_jump_to_zero
test_fmrg_j_reduces_tiny_representation_loss
test_fmrg_j_normalizes_gradient_per_sample
test_fmrg_j_leaves_generator_codec_and_e0_weights_unchanged
test_fmrg_j_rejects_non_decreasing_schedule
test_fmrg_j_rejects_non_positive_eta
test_fmrg_j_fails_on_non_finite_gradient
```

For `K=2`, assert `result.nfe == 5`. Record fake flow-map `(t,r)` calls and assert the final call has `r=0` and starts at `t_cut`.

### Step 2: Verify the tests fail

Run the focused guidance test command. Expected: failure because `sample_fmrg_j` does not exist.

### Step 3: Implement the reviewed FMRG-J equation

Add:

```python
def sample_fmrg_j(
    *,
    flow_map: CountedFlowMap,
    codec,
    e0,
    x_init: torch.Tensor,
    transport_condition: torch.Tensor,
    target_z0: torch.Tensor,
    guided_times: Sequence[float],
    eta: float,
) -> GuidanceResult:
    ...
```

The `guided_times` sequence contains `[1.0, ..., t_cut]`; zero is not included. For each adjacent `(t,s)` pair:

```python
with torch.no_grad():
    x_bar = flow_map(x, transport_condition, t=t, r=s)
    u_step = (x - x_bar) / (t - s)

x_bar = x_bar.detach().requires_grad_(True)
x0_hat = flow_map(x_bar, transport_condition, t=s, r=0.0)
image = codec.decode(x0_hat)
pred_z0 = e0(normalize_for_e0(image))["embedding"]
loss = (1.0 - F.cosine_similarity(pred_z0, target_z0, dim=1)).mean()
gradient = torch.autograd.grad(loss, x_bar, only_inputs=True)[0]

gradient_norm = gradient.flatten(1).norm(dim=1).clamp_min(1.0e-8)
velocity_norm = u_step.flatten(1).norm(dim=1)
scale = (velocity_norm / gradient_norm).view(-1, 1, 1, 1)
x = (x_bar - (t - s) * eta * gradient * scale).detach()
```

After all guided intervals:

```python
with torch.no_grad():
    x0 = flow_map(x, transport_condition, t=guided_times[-1], r=0.0)
```

Do not use `.backward()`, Adam, momentum, generator gradients, a quality loss, clipping, smoothing, or image post-processing. Validate every loss, gradient, scale, state, and endpoint with finite checks. Save per-interval representation loss and gradient/velocity norms in diagnostics.

### Step 4: Run focused and model tests

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_flow_map_guidance.py \
  tests/test_meanflow_sit_generator.py -q
```

Expected: all pass.

### Step 5: Commit and push

```bash
git add src/safa/guidance/meanflow_flow_map.py tests/test_meanflow_flow_map_guidance.py
git commit -m "feat(guidance): add frozen FMRG-J sampler"
git push origin master
```

## Task 4: Implement the Constrained Initial-Noise Oracle

**Files:**
- Modify: `src/safa/guidance/meanflow_flow_map.py`
- Modify: `tests/test_meanflow_flow_map_guidance.py`

### Step 1: Write failing projection and oracle tests

Add:

```text
test_project_fixed_radius_restores_each_initial_norm
test_project_typical_shell_clamps_only_outside_radii
test_project_typical_shell_rejects_invalid_delta
test_noise_oracle_reduces_tiny_representation_loss
test_noise_oracle_re_evaluates_projected_final_point
test_noise_oracle_reports_updates_plus_one_nfe
test_noise_oracle_does_not_change_frozen_weights
test_noise_oracle_rejects_non_finite_state
```

For eight updates, assert `result.nfe == 9`. Test norms per sample rather than only a batch average.

### Step 2: Verify failure

Run the focused guidance tests. Expected: the projection and oracle symbols are missing.

### Step 3: Implement exact projections

Add:

```python
def project_fixed_radius(candidate, initial, eps=1.0e-8): ...
def project_gaussian_typical_shell(candidate, *, delta, eps=1.0e-8): ...
```

For shape `[B,C,H,W]`, use `d=C*H*W`, `r_min=sqrt(d*(1-delta))`, and `r_max=sqrt(d*(1+delta))`. Require `0 < delta < 1`. A zero candidate must raise instead of inventing a direction.

### Step 4: Implement normalized projected gradient descent

Add:

```python
def optimize_initial_noise(
    *,
    flow_map: CountedFlowMap,
    codec,
    e0,
    x_init: torch.Tensor,
    transport_condition: torch.Tensor,
    target_z0: torch.Tensor,
    num_updates: int,
    step_fraction: float,
    projection: Literal["fixed_radius", "typical_shell"],
    typical_delta: float = 0.05,
) -> GuidanceResult:
    ...
```

At each update, evaluate `Phi_{1->0}`, decode, calculate full-Z0 cosine loss, take `grad = autograd.grad(loss, noise)`, normalize each sample, and update by:

```text
step_norm = step_fraction * sqrt(d)
noise <- noise - step_norm * grad / (||grad||_2 + 1e-8)
noise <- project(noise)
```

Detach and re-enable only the input gradient between updates. Re-evaluate the projected final point after the last update. Record the initial/final norm, norm squared per dimension, initial-final cosine, update norm, channel mean/std, loss history, and NFE.

### Step 5: Run tests and commit

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_flow_map_guidance.py -q
git add src/safa/guidance/meanflow_flow_map.py tests/test_meanflow_flow_map_guidance.py
git commit -m "feat(guidance): add constrained noise oracle"
git push origin master
```

## Task 5: Add Sharpness to the Shared Quality Evaluator

**Files:**
- Modify: `tests/test_phase_a_scripts.py`
- Modify: `scripts/eval_generation_quality.py`

### Step 1: Write failing Sharpness tests

Add tests that:

1. A flat grayscale image has Laplacian variance zero.
2. A checkerboard image has positive Sharpness.
3. `metrics=["sharpness"]` does not require a real index or create FID/KID/NIQE models.
4. The JSON contains mean, std, median, p10, and p90.
5. Existing default metrics remain exactly `fid`, `kid`, and `niqe` for backward compatibility.

Use the installed OpenCV definition:

```python
float(cv2.Laplacian(gray, cv2.CV_64F).var())
```

### Step 2: Run and verify failure

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_phase_a_scripts.py -q
```

Expected: `sharpness` is rejected as unsupported.

### Step 3: Implement without changing defaults

Add `sharpness` to `SUPPORTED_METRICS`, not to `DEFAULT_METRICS`. Compute it for the selected generated paths only. Write:

```json
{
  "sharpness": {
    "definition": "grayscale_laplacian_variance",
    "mean": 0.0,
    "std": 0.0,
    "median": 0.0,
    "p10": 0.0,
    "p90": 0.0
  }
}
```

Do not mix source/real image Sharpness into the generated summary.

### Step 4: Run tests and commit

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_phase_a_scripts.py -q
git add scripts/eval_generation_quality.py tests/test_phase_a_scripts.py
git commit -m "feat(metrics): report generated-image sharpness"
git push origin master
```

## Task 6: Build the Guidance Generation Runner

**Files:**
- Create: `src/safa/evaluation/meanflow_guidance_runner.py`
- Create: `scripts/run_meanflow_flow_map_guidance.py`
- Create: `tests/test_meanflow_guidance_runner.py`

### Step 1: Write failing config and checkpoint-contract tests

Add tests with synthetic checkpoint dictionaries for:

```text
test_checkpoint_contract_requires_ema_state
test_checkpoint_contract_requires_stage2_epoch1652
test_checkpoint_contract_requires_meanflow_sit_b4
test_checkpoint_contract_requires_learned_null_condition
test_checkpoint_contract_accepts_exact_target_metadata
test_guidance_config_rejects_target_condition_as_transport
test_guidance_config_requires_unique_output_directory
test_result_metadata_records_checkpoint_hash_seed_nfe_and_weight_source
```

Do not load the 2 GB real checkpoint in unit tests.

### Step 2: Verify failure

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_guidance_runner.py -q
```

Expected: import failure.

### Step 3: Implement strict loading

The runner module must:

1. Load YAML and validate all required fields before CUDA work.
2. Load the checkpoint with `map_location="cpu"` and `weights_only=True`.
3. Check `stage`, `metrics.stage_epoch_1based`, and every fixed model-config field.
4. Build the generator from `model_config` and load only `ema_model_state_dict` with strict state loading.
5. Load/freeze E0 and VAE, then call `assert_guidance_stack_frozen`.
6. Use `FeatureAlignedAffectNet` and `make_x_init_for_sample_ids` so sample IDs and noise match existing evaluation.
7. Construct transport condition only with `generator.make_null_condition(...)`.

Reject mismatches; do not fall back to raw weights, another epoch, B/2, random initialization, a downloaded model, or target-conditioned transport.

### Step 4: Implement route execution and artifacts

Support these route names:

```text
native
semigroup
fmrg_j
initial_noise
```

For each generation arm:

- Save one PNG per sample with a stable ordinal and sanitized sample ID.
- Save `per_sample.jsonl` with target cosine and route diagnostics.
- Save `generation_result.json` with summary cosine statistics, NFE, wall time, images/sec, peak allocated VRAM, peak reserved VRAM, checkpoint SHA256, exact config, and sample count.
- Save 64 deterministic visual pages. Each row contains source, matched native, and candidate. Label columns in a separate JSON manifest, not by drawing text over images.
- Save all raw candidate images. Do not save only face-detected images or a selected subset.

Measure CUDA cost as follows:

```python
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats(device)
start = time.perf_counter()
# generate
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
peak_allocated = torch.cuda.max_memory_allocated(device)
peak_reserved = torch.cuda.max_memory_reserved(device)
```

NFE is the counted vector-field calls in the candidate algorithm. Native comparison images generated only for montage construction must be reported separately and excluded from candidate NFE.

### Step 5: Add a thin CLI

`scripts/run_meanflow_flow_map_guidance.py` should accept:

```text
--config PATH
--max-samples N
--output-dir PATH
--eta FLOAT
--num-updates N
--step-fraction FLOAT
--projection fixed_radius|typical_shell
--semigroup-report PATH
```

Config values are the default; explicit CLI values are recorded overrides. `--semigroup-report` is mandatory for FMRG-J and must name a passing report with the same checkpoint SHA256.

### Step 6: Test, compile, commit, and push

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_guidance_runner.py \
  tests/test_meanflow_flow_map_guidance.py -q
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m compileall -q \
  src/safa/guidance src/safa/evaluation/meanflow_guidance_runner.py \
  scripts/run_meanflow_flow_map_guidance.py
git add src/safa/evaluation/meanflow_guidance_runner.py \
  scripts/run_meanflow_flow_map_guidance.py tests/test_meanflow_guidance_runner.py
git commit -m "feat(evaluation): add MeanFlow guidance runner"
git push origin master
```

## Task 7: Add Reproducible Experiment Configs

**Files:**
- Create: `configs/medium_v2/experiments/r8_meanflow_semigroup_preflight.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_native_ema_gpu0.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_fmrg_j_k1_gpu1.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_fmrg_j_k2_gpu2.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_noise_oracle_gpu3.yaml`
- Create: `tests/test_r8_meanflow_guidance_configs.py`

### Step 1: Write config-contract tests first

The tests must load all five YAML files and assert:

- Exact checkpoint, E0, VAE, index, and feature paths.
- `checkpoint_model: ema`.
- Expected Stage 2 epoch 1652 and patch size 4.
- `transport_condition: learned_null_condition`.
- `sampling_seed: 1337`.
- Unique experiment and output names.
- Semigroup split times exactly `[0.75, 0.5, 0.25]`.
- FMRG K=1 times `[1.0, 0.25]` and K=2 times `[1.0, 0.625, 0.25]`; the runner may replace `0.25` only with the passing preflight split and must record it.
- Noise oracle projection defaults to `fixed_radius`, `num_updates: 8`, and `step_fraction: 0.003`.
- Required metrics are `fid`, `kid`, `niqe`, and `sharpness`.
- Full sample count is 2048 and visual review count is 64.

### Step 2: Verify tests fail

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_r8_meanflow_guidance_configs.py -q
```

Expected: missing files.

### Step 3: Create the five standalone configs

Every config must include this common block:

```yaml
seed: 1337
sampling_seed: 1337
device: cuda:0
checkpoint: artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last_nopretrained.pt
checkpoint_model: ema
expected_stage: stage2
expected_stage_epoch_1based: 1652
expected_model_type: meanflow_sit
expected_sit_patch_size: 4
transport_condition: learned_null_condition
e0_checkpoint: artifacts/checkpoints/e0_medium_v1/best.pt
vae_path: artifacts/checkpoints/external/sd-vae-ft-ema
vae_scaling_factor: 0.18215
index: data/index/val_face_mixed_e14.jsonl
features: artifacts/e0_features/val_face_mixed_e14_e0_medium_v1
pixel_image_size: 256
batch_size: 2
num_workers: 4
max_samples: 2048
visual_review_samples: 64
quality_metrics: [fid, kid, niqe, sharpness]
```

Add the route-specific values asserted above. Keep calibration candidates in config:

```text
FMRG eta candidates: 0.01, 0.03, 0.10
noise update candidates: 4, 8, 16
noise step-fraction candidates: 0.001, 0.003, 0.010
```

These are a closed first search. Do not add candidates after looking at results without a new config and commit.

### Step 4: Test and commit

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_r8_meanflow_guidance_configs.py -q
git add configs/medium_v2/experiments/r8_meanflow_*.yaml \
  tests/test_r8_meanflow_guidance_configs.py
git commit -m "experiment: add R8 MeanFlow guidance configs"
git push origin master
```

## Task 8: Add the Four-GPU Matrix Launcher

**Files:**
- Create: `scripts/run_r8_meanflow_guidance_matrix.py`
- Create: `tests/test_run_r8_meanflow_guidance_matrix.py`

### Step 1: Write launcher tests first

Follow the tested structure in `scripts/run_r7_independent_prior_matrix.py`. Add tests for:

```text
test_matrix_pins_exact_physical_gpus_zero_through_three
test_matrix_uses_gpu_uuid_as_cuda_visible_devices
test_matrix_dry_run_is_default_and_has_no_writes
test_matrix_requires_explicit_execute
test_matrix_rejects_missing_checkpoint_e0_vae_index_or_features
test_matrix_rejects_existing_output_directory
test_matrix_requires_passing_semigroup_report_for_fmrg
test_matrix_calibration_launches_four_processes_concurrently
test_matrix_full_requires_2048_samples_and_visual_review
test_matrix_records_exit_codes_peak_memory_and_external_processes
test_matrix_terminates_started_children_after_partial_launch_failure
```

### Step 2: Verify failure

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_run_r8_meanflow_guidance_matrix.py -q
```

Expected: missing script.

### Step 3: Implement explicit phases

Support:

```text
--phase semigroup|calibrate|full|all
--dry-run
--execute
--allow-busy-gpus
--python PATH
--repo-root PATH
```

Dry-run is the default. Query `nvidia-smi` once, bind physical GPUs 0-3 by UUID, and expose each as logical `cuda:0`. Do not kill or modify external processes. `--allow-busy-gpus` records authorized sharing but does not bypass a hard free-memory minimum required by the config batch size.

Phase behavior:

```text
semigroup:
  GPU0 only, 64 samples, writes the gate report

calibrate:
  GPU0 native control
  GPU1 sequential K=1 eta candidates
  GPU2 sequential K=2 eta candidates
  GPU3 sequential noise-oracle candidates
  all four physical GPU processes start before waiting for any one of them

full:
  GPU0 native 2048
  GPU1 selected K=1 2048
  GPU2 selected K=2 2048
  GPU3 selected noise oracle 2048
```

For each generated candidate, chain the quality command only after generation succeeds:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/eval_generation_quality.py \
  --real-index data/index/val_face_mixed_e14.jsonl \
  --generated-dir ARM_OUTPUT/generated_images \
  --output ARM_OUTPUT/quality.json \
  --max-generated 2048 \
  --max-real 2048 \
  --seed 1337 \
  --device cuda:0 \
  --metrics fid kid niqe sharpness
```

The calibration form changes both limits to 128. Write a lock and `matrix_status.json` with exact commands, GPU UUIDs, external process records, start/end times, exit codes, and output paths. Refuse to overwrite any output.

### Step 4: Test, dry-run, commit, and push

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_run_r8_meanflow_guidance_matrix.py -q
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/run_r8_meanflow_guidance_matrix.py --dry-run --phase all
git add scripts/run_r8_meanflow_guidance_matrix.py \
  tests/test_run_r8_meanflow_guidance_matrix.py
git commit -m "experiment: add four-GPU MeanFlow guidance runner"
git push origin master
```

## Task 9: Add Selection and Result Aggregation

**Files:**
- Create: `scripts/summarize_r8_meanflow_guidance.py`
- Create: `tests/test_summarize_r8_meanflow_guidance.py`

### Step 1: Write failing summary tests

Test that the script:

- Joins `generation_result.json`, `quality.json`, and visual review by exact arm ID.
- Refuses mismatched sample counts, checkpoint hashes, seeds, or sample-ID digests.
- Refuses non-finite metrics.
- Does not treat face-detection rate as a quality gate.
- Calculates `solved` and `directional_evidence` exactly as the design specifies.
- Writes a CSV and Markdown table with FID, KID, NIQE, Sharpness, cosine, NFE, speed, peak VRAM, and severe visual count.
- Requires a direct-review `visual_review.json` before choosing full-run candidates.

### Step 2: Verify failure

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_summarize_r8_meanflow_guidance.py -q
```

Expected: missing script.

### Step 3: Implement strict aggregation

Use this calibration eligibility filter before ranking within K=1, K=2, or oracle families:

```text
all values finite
cosine improvement over matched native >= 0.02
Sharpness mean >= 0.80 * native Sharpness mean
severe visual failures <= 10% of the 64 reviewed samples
```

Among eligible candidates in one family, select the highest cosine, then lower FID, then lower NFE as deterministic tie breakers. If no candidate is eligible, mark the family failed and do not launch its full arm. Do not weaken the filter automatically.

Write:

```text
artifacts/r8_meanflow_flow_map_guidance/selection.json
artifacts/r8_meanflow_flow_map_guidance/summary.csv
artifacts/r8_meanflow_flow_map_guidance/summary.md
```

### Step 4: Test and commit

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_summarize_r8_meanflow_guidance.py -q
git add scripts/summarize_r8_meanflow_guidance.py \
  tests/test_summarize_r8_meanflow_guidance.py
git commit -m "experiment: summarize R8 guidance tradeoffs"
git push origin master
```

## Task 10: Run the Semigroup Gate

**Files:**
- Generated only under: `artifacts/r8_meanflow_flow_map_guidance/semigroup/`

### Step 1: Verify current repository and GPU state

```bash
git status --short
git rev-parse HEAD
git rev-parse origin/master
nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits
```

Require a clean tree and `HEAD == origin/master`. Existing GPU processes may be shared only because the user authorized it; record them with `--allow-busy-gpus`.

### Step 2: Run the gate

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/run_r8_meanflow_guidance_matrix.py \
  --phase semigroup --execute --allow-busy-gpus
```

### Step 3: Validate artifacts and inspect images

Check the report against the fixed numerical gate. Open every direct/composed comparison page. Record the selected `t_cut`, or record that no split passed.

If the gate fails, skip FMRG calibration/full tasks below and continue with native plus initial-noise oracle. Do not wait for new input.

## Task 11: Run Four-GPU Calibration and Direct Visual Review

**Files:**
- Generated under: `artifacts/r8_meanflow_flow_map_guidance/calibration/`
- Create generated artifact: `artifacts/r8_meanflow_flow_map_guidance/visual_review.json`

### Step 1: Launch all four physical GPUs

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/run_r8_meanflow_guidance_matrix.py \
  --phase calibrate --execute --allow-busy-gpus
```

Do not return while child sessions are running. Check `matrix_status.json` and every log after all processes exit.

### Step 2: Inspect every visual page directly

Open the deterministic pages for native and every candidate. For each severe failure, record sample ID and one of the fixed failure categories from the design. Do not use face count as a substitute.

Write `visual_review.json` with:

```json
{
  "reviewed_sample_count": 64,
  "arms": {
    "arm_id": {
      "severe_failure_count": 0,
      "failures": [
        {"sample_id": "...", "category": "unstructured_noise"}
      ]
    }
  }
}
```

### Step 3: Select candidates

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/summarize_r8_meanflow_guidance.py \
  --root artifacts/r8_meanflow_flow_map_guidance \
  --phase calibration
```

Verify `selection.json` contains only candidates that pass numerical and visual filters.

## Task 12: Run the Four-GPU Full Evaluation

**Files:**
- Generated under: `artifacts/r8_meanflow_flow_map_guidance/full/`

### Step 1: Launch full arms concurrently

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/run_r8_meanflow_guidance_matrix.py \
  --phase full --execute --allow-busy-gpus
```

The launcher must start physical GPUs 0-3 before waiting. If semigroup or calibration disqualified a family, use that GPU for an exact native repeat with a distinct seed only if the config explicitly records it; do not fabricate an FMRG result.

### Step 2: Verify full counts and metrics

For every completed primary arm, require:

```text
generated PNG count: 2048
real metric count: 2048
per-sample JSONL rows: 2048
visual-review manifest sample IDs: 64
all required metric fields finite
checkpoint hashes and sample-ID digests identical across arms
```

Open all full visual pages and update `visual_review.json` with the full-arm IDs.

### Step 3: Produce the final table

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/summarize_r8_meanflow_guidance.py \
  --root artifacts/r8_meanflow_flow_map_guidance \
  --phase full
```

The table must compare every arm directly with fresh native epoch-1652 EMA, not with an older approximate FID value.

## Task 13: Final Validation and Result Commit

**Files:**
- Create after experiments: `docs/results/safa-r8-meanflow-flow-map-guidance-2026-07-12.md`

### Step 1: Run focused tests

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_sit_generator.py \
  tests/test_meanflow_flow_map_guidance.py \
  tests/test_meanflow_guidance_runner.py \
  tests/test_r8_meanflow_guidance_configs.py \
  tests/test_run_r8_meanflow_guidance_matrix.py \
  tests/test_summarize_r8_meanflow_guidance.py \
  tests/test_phase_a_scripts.py -q
```

### Step 2: Run the full existing suite and static checks

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest -q
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m ruff check src scripts tests
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m compileall -q src scripts tests
git diff --check
```

Expected: all tests pass, Ruff passes, compileall passes, and `git diff --check` prints nothing.

### Step 3: Write the result report

The report must contain:

- Exact checkpoint contract and SHA256.
- Semigroup residual table and pass/fail decision.
- Full metric table with FID, KID, NIQE, Sharpness, cosine, NFE, speed, and VRAM.
- Direct visual-failure counts with linked artifact paths.
- `solved`, `directional evidence`, or `failed` label using the fixed rules.
- A plain statement that MPGD, LGD, DPS, FreeDoM, and Universal Guidance were not reproduced on this 1-step model.
- A plain statement that Route C is only a radial-constraint feasibility oracle.
- The next action implied by the observed branch, without adding a new untested fix.

### Step 4: Commit and push the result separately

```bash
git add docs/results/safa-r8-meanflow-flow-map-guidance-2026-07-12.md
git commit -m "docs(results): record R8 MeanFlow guidance study"
git push origin master
git status --short
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)"
```

Expected: clean worktree and local HEAD equal to `origin/master`.
