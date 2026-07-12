# MeanFlow Flow-Map Representation Guidance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the frozen Stage 2 epoch-1652 MeanFlow SiT-B/4 EMA can improve full-Z0 cosine without losing FID, Sharpness, KID, or visible image quality.

**Architecture:** Expose the trained two-time MeanFlow map as a differentiable model API. Build a semigroup diagnostic, the official current-x_t FMRG-J baseline, the paper split-state FMRG-J ablation, and projected initial-noise optimization. Calibrate with E0 and a ResNet18 development encoder, lock one winner, then evaluate frozen DINOv2 and ConvNeXt held-out encoders once on the 2048-sample native/winner comparison.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, diffusers AutoencoderKL, existing SAFA E0 and feature cache, torchmetrics FID/KID, pyIQA NIQE, OpenCV Sharpness, pytest, Ruff, YAML, nvidia-smi.

---

## Fixed Contracts

Use these paths in every config and test command:

```text
repo=/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
python=/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python
checkpoint=artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last_nopretrained.pt
e0=artifacts/checkpoints/e0_medium_v1/best.pt
edev=artifacts/checkpoints/e0_resnet18/best.pt
e1=artifacts/checkpoints/e0_dinov2_large_v2/best.pt
e2=artifacts/checkpoints/e0_convnext_tiny/best.pt
vae=artifacts/checkpoints/external/sd-vae-ft-ema
index=data/index/val_face_mixed_e14.jsonl
features=artifacts/e0_features/val_face_mixed_e14_e0_medium_v1
```

Use `ema_model_state_dict`, learned null condition for every flow-map call, `sampling_seed=1337`, and full cached `Z0` only as the representation target. No task in this plan updates model weights.

Use these fixed encoder hashes:

```text
E0   d7d2c57a552155776b8c15a4e52e43ec5082fc046aa0aabb4e9709685f7e3d1a
Edev 373b331c917834467e854ddf3fe20f39000532f189ec73f76a1abc55d82e560e
E1   cce0de2f1eab097cb6091886f587a9f334dd84ced1ca4dd5e08c3a765718a14c
E2   09c88bd416057222abefeba52ebe88d710715ede791ec34198a23ae5e6e850a8
```

E1/E2 are prospective held-out encoders. The runner may verify their file hashes at preflight, but no E1/E2 forward pass is allowed until the winner and 2048 image manifests are locked.

The implementation must not use `find /` or scan outside the SAFA repository and the explicitly listed directories under `/home/hdd3/zhanghaonan/projects/safa-paper-code`. Use `rg`, `rg --files`, and explicit paths.

The clean-room references are fixed at: MPGD `9f94b386` (MIT), DPS `effbde73` (no top-level license), FreeDoM `1394b1dc` (no top-level license), Universal Guidance `ff82f880` (no top-level license), Z+ `abedeb1b` (AGPL/MIT component split), II2S `6ce02da8` (no top-level license), StyleCLIP `f87a47f7` (MIT), and FMRG `be485ba` (license text incomplete across components). No author-official LGD code was located. Use the exact audited entries recorded in the companion design document; do not copy code from a repository whose license does not allow it.

## Completion Criteria

1. Native `sample` remains numerically identical after the flow-map API refactor.
2. The checkpoint loader rejects any checkpoint other than Stage 2 epoch 1652 MeanFlow SiT-B/4 with EMA weights.
3. Semigroup, both named FMRG-J variants, and noise-oracle tests pass, including official reference NFE and frozen-weight assertions.
4. FID, KID, NIQE, Sharpness, E0/Edev metrics, NFE, speed, and peak VRAM are recorded for calibration/full as applicable.
5. The 64 fixed visual comparisons are opened and reviewed directly; face detection is not used as the collapse decision.
6. Semigroup, calibration, and full processes use physical GPUs 0, 1, 2, and 3 concurrently.
7. E1/E2 run once on the locked native/winner 2048 manifests and report within-encoder cosine, pairwise-distance Spearman, and 8-class accuracy. No cross-encoder cosine is computed.
8. Every code/config/test unit is committed and pushed as a fine-grained commit. Generated result docs are committed separately.

## Task 1: Add the General MeanFlow Map API

**Files:**
- Modify: `tests/test_meanflow_sit_generator.py`
- Modify: `src/safa/models/meanflow_sit.py:324-369`

### Step 1: Write failing direct-equivalence and validation tests

Add these tests to `tests/test_meanflow_sit_generator.py`:

```python
def test_meanflow_flow_map_1_to_0_matches_native_sample() -> None:
    config = _tiny_meanflow_sit_config()
    config["sit_data_space"] = "latent"
    generator = build_generator(config)
    z = torch.randn(2, 16)
    x_init = torch.randn(2, 3, 16, 16)

    expected = generator.sample(z, x_init=x_init, clamp_output=False)
    actual = generator.flow_map(x_init, z, t=1.0, r=0.0)

    assert torch.equal(actual, expected)


def test_meanflow_pixel_flow_map_matches_sample_after_data_space_conversion() -> None:
    config = _tiny_meanflow_sit_config()
    config["sit_data_space"] = "pixel"
    generator = build_generator(config)
    z = torch.randn(2, 16)
    x_init = torch.randn(2, 3, 16, 16)

    expected = generator.sample(z, x_init=x_init, clamp_output=False)
    model_space = generator.flow_map(x_init, z, t=1.0, r=0.0)
    actual = generator._model_to_data_space(model_space, clamp_output=False)

    assert torch.equal(actual, expected)


def test_meanflow_flow_map_accepts_per_sample_times_and_input_gradient() -> None:
    config = _tiny_meanflow_sit_config()
    config["sit_data_space"] = "latent"
    generator = build_generator(config)
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
test_latent_codec_wrapper_freezes_vae_but_keeps_decode_input_gradient
test_assert_guidance_stack_checks_codec_vae_not_codec_parameters
```

The exact-semigroup fake should implement `Phi_{t->r}(x) = exp(-(t-r)) * x`, so direct and composed endpoints are equal within floating-point tolerance.

Do not rely only on the identity codec fake. Construct the real `safa.training.latent_codec.LatentCodec` around a small differentiable fake VAE that exposes the actual `.decode(...).sample` interface. Assert that `codec` is not treated as an `nn.Module`, `codec.vae.training` is false, every VAE parameter has `requires_grad=False`, `decoded.sum().backward()` populates the latent input gradient, and no VAE parameter gradient appears.

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


def freeze_guidance_stack(generator, codec, e0) -> None:
    generator.eval().requires_grad_(False)
    codec.vae.eval()
    codec.vae.requires_grad_(False)
    e0.eval().requires_grad_(False)


def assert_guidance_stack_frozen(generator, codec, e0) -> None:
    # Inspect generator.parameters(), codec.vae.parameters(), and e0.parameters().
    ...
def symmetric_relative_l2(left, right, eps=1.0e-8) -> torch.Tensor: ...
def semigroup_probe(flow_map, x_init, condition, split_times) -> dict: ...
```

`CountedFlowMap` increments once for each vector-field evaluation, not once per image. `semigroup_probe` must return the direct endpoint, each composed endpoint, per-sample residuals, and total NFE. It must not decode or compute FID.

Never wrap `codec.decode(...)` or the following E0 forward in `torch.no_grad()` inside guidance. Parameter freezing prevents weight updates while autograd must still connect the decoded image and loss to the latent input.

### Step 4: Run focused tests

Use the command from Step 2. Expected: all pass.

### Step 5: Commit and push

```bash
git add src/safa/guidance tests/test_meanflow_flow_map_guidance.py
git commit -m "feat(guidance): add MeanFlow semigroup probe"
git push origin master
```

## Task 3: Implement Both Audited FMRG-J Orderings

**Files:**
- Modify: `src/safa/guidance/meanflow_flow_map.py`
- Modify: `tests/test_meanflow_flow_map_guidance.py`

### Step 1: Write failing official-HEAD tests

Add tests with these exact requirements:

```text
test_official_current_xt_takes_endpoint_gradient_at_xt_before_advance
test_official_current_xt_flow_map1_reuses_endpoint_velocity
test_official_current_xt_flow_map2_uses_distinct_endpoint_and_step_maps
test_official_current_xt_supports_adam_and_normalized_direct_modes
test_safa_uniform_schedule_flow_map1_nopt1_is_five_nfe
test_safa_uniform_schedule_flow_map2_nopt1_is_eight_nfe
test_official_adam_uses_interval_decay_one_minus_i_over_four
test_official_adam_nopt_gt_one_refreshes_endpoint_at_updated_xt
test_normalized_mode_has_no_adam_state_or_lr_decay
test_official_current_xt_finishes_with_official_unguided_tail_order
test_official_current_xt_leaves_generator_codec_and_e0_unchanged
test_official_current_xt_fails_on_non_finite_gradient
```

Record every fake vector-field `(t,r)` call. With `guided_times=linspace(1,t_cut,4)`, `unguided_times=linspace(t_cut,0,3)`, and `nopt=1`, assert five NFE for `flow_map1` and eight for `flow_map2`. Assert Adam learning rates are exactly `step_size*(1-i/4)` for guided interval indices `i=0,1,2`. With `nopt=2`, record two distinct endpoint calls and prove the second consumes the updated `x_t`, not the interval's initial tensor.

### Step 2: Write failing paper-split tests

Add:

```text
test_paper_split_transports_to_xs_before_endpoint_gradient
test_paper_split_finishes_with_explicit_unguided_map
test_paper_split_nfe_matches_counted_calls
test_paper_split_normalizes_gradient_per_sample
test_paper_split_reduces_tiny_representation_loss
test_fmrg_variants_reject_non_decreasing_schedule
test_fmrg_variants_reject_non_positive_step_size
```

The fake call trace must prove that `official_head_current_xt` differentiates `Phi_{t->0}` at `x_t`, while `paper_algorithm_split` first calls `Phi_{t->s}` and differentiates `Phi_{s->0}` at `x_s`.

### Step 3: Verify the tests fail

Run the focused guidance test command. Expected: failures because neither named variant exists.

### Step 4: Implement B1 `official_head_current_xt`

Add:

```python
def sample_official_head_current_xt(
    *,
    flow_map: CountedFlowMap,
    codec,
    e0,
    x_init: torch.Tensor,
    transport_condition: torch.Tensor,
    target_z0: torch.Tensor,
    guided_times: Sequence[float],
    unguided_times: Sequence[float],
    sample_mode: Literal["flow_map1", "flow_map2"],
    optimization_mode: Literal["official_adam", "paper_normalized_direct_autograd"],
    num_optim_iters: int,
    step_size: float,
) -> GuidanceResult:
    ...
```

Follow the current-x_t update behavior in official `fmrg/fluxfm_sampler_reward.py:388-461,1039-1115,1140-1167`, but use SAFA's explicit uniform time arrays rather than FLUX dynamic-shift timesteps. At current `x_t`, build the endpoint lookahead before advancing:

```python
x_t = x_t.detach().requires_grad_(True)
x0_hat = flow_map(x_t, transport_condition, t=t, r=0.0)
u_endpoint = (x_t - x0_hat) / t
image = codec.decode(x0_hat)
pred_z0 = e0(normalize_for_e0(image))["embedding"]
loss = (1.0 - F.cosine_similarity(pred_z0, target_z0, dim=1)).mean()
```

For `flow_map1`, reuse `u_endpoint` as `u_step`. For `flow_map2`, make a separate counted `x_step=flow_map(x_t, ..., t=t, r=s)` call and derive `u_step=(x_t-x_step)/(t-s)`. Implement the two audited correction modes without silently substituting one for the other:

```python
if optimization_mode == "official_adam":
    lr_i = step_size * (1.0 - interval_index / 4.0)
    # Match the official inner x_t Adam update and derive
    # delta_xt = -(x_t_after - x_t_before).
    # For every inner iteration, recompute x0_hat from the updated x_t.
    ...
else:
    gradient = torch.autograd.grad(loss, x_t, only_inputs=True)[0]
    gradient = normalize_per_sample_to_velocity_norm(gradient, u_step)
    delta_xt = step_size * gradient

x_s = (x_t_before - (t - s) * (u_step.detach() + delta_xt)).detach()
```

The official Adam mode may use `.backward()` only on the temporary `x_t` leaf; clear and assert all frozen parameter gradients remain `None`. The normalized mode uses `torch.autograd.grad`. Validate every loss, gradient, state, and endpoint. Record exact vector-field calls, not a derived NFE estimate.

### Step 5: Implement B2 `paper_algorithm_split`

Add a separate function:

```python
def sample_paper_algorithm_split(
    *,
    flow_map: CountedFlowMap,
    codec,
    e0,
    x_init: torch.Tensor,
    transport_condition: torch.Tensor,
    target_z0: torch.Tensor,
    guided_times: Sequence[float],
    step_size: float,
) -> GuidanceResult:
    ...
```

This function first computes `x_bar=Phi_{t->s}(x_t)`, then differentiates `Phi_{s->0}(x_bar)` with respect to `x_bar`, applies the per-sample normalized correction at time `s`, and finishes with an unguided map. It must not call the official-current-x_t function internally. `CountedFlowMap` is the only source of its NFE result.

### Step 6: Run focused and model tests

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_flow_map_guidance.py \
  tests/test_meanflow_sit_generator.py -q
```

Expected: all pass.

### Step 7: Commit and push

```bash
git add src/safa/guidance/meanflow_flow_map.py tests/test_meanflow_flow_map_guidance.py
git commit -m "feat(guidance): add audited FMRG-J variants"
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
    eta: float,
    projection: Literal["fixed_radius", "typical_shell"],
    typical_delta: float = 0.05,
) -> GuidanceResult:
    ...
```

At each update, evaluate `Phi_{1->0}`, decode, calculate full-Z0 cosine loss, take `grad = autograd.grad(loss, noise)`, normalize each sample, and update by:

```text
noise <- noise - eta * grad / (||grad||_2 + 1e-8)
noise <- project(noise)
```

Use only `eta in {0.25,0.5,1.0,2.0}`. Detach and re-enable only the input gradient between updates. Re-evaluate the projected final point after the last update. Record the initial/final norm, norm squared per dimension, initial-final cosine, update norm, channel mean/std, loss history, and NFE.

### Step 5: Run tests and commit

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_meanflow_flow_map_guidance.py -q
git add src/safa/guidance/meanflow_flow_map.py tests/test_meanflow_flow_map_guidance.py
git commit -m "feat(guidance): add constrained noise oracle"
git push origin master
```

## Task 5: Add Sharpness and Exact Sample-ID Quality Joining

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
6. `--sample-id-manifest` joins the real index and `--per-sample-jsonl` by exact sample ID.
7. Missing, duplicate, or extra IDs in any input fail before metric creation.
8. Manifest mode rejects `--max-real` and `--max-generated` and never calls path-hash subset selection.
9. Two generated output directories with different filenames and path order still select the same ordered IDs when their per-sample JSONL files map the same manifest.
10. Native and candidate payloads record the same ordered sample-ID digest.

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

### Step 3: Implement manifest joining without changing legacy defaults

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

Add CLI arguments:

```text
--sample-id-manifest PATH
--per-sample-jsonl PATH
```

The manifest is ordered JSONL with one unique `sample_id` per row. Build exact maps from `real_index.sample_id -> image_path` and `per_sample.sample_id -> generated_image_path`, then materialize both path lists in manifest order. Reject any missing, duplicate, or extra sample ID. Record `sample_id_manifest`, count, and SHA256 digest in `quality.json`.

R8 must use manifest mode. In this mode reject `--max-real`, `--max-generated`, path-hash selection, directory-order truncation, and any generated file not selected through `per_sample.jsonl`. Legacy non-R8 callers may retain the old flags for backward compatibility, but the R8 matrix and result validator must reject an R8 payload that used them.

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
test_guidance_config_locks_uniform_times_from_manifest_t_cut
test_guidance_config_rejects_t_cut_mismatch_across_manifest_cli_and_yaml
test_result_metadata_records_checkpoint_hash_seed_nfe_and_weight_source
test_calibration_loads_e0_and_edev_but_not_e1_or_e2
test_final_heldout_eval_requires_locked_winner_manifest
test_encoder_metric_rejects_cross_coordinate_target
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
5. Load/freeze E0 and VAE. Load/freeze Edev only for calibration. Do not load E1/E2 in this runner before winner lock.
6. Use `FeatureAlignedAffectNet` and `make_x_init_for_sample_ids` so sample IDs and noise match existing evaluation.
7. Construct transport condition only with `generator.make_null_condition(...)`.
8. Resolve `guided_times=linspace(1,t_cut,4)` and `unguided_times=linspace(t_cut,0,3)` from the locked schedule manifest; reject any CLI/config disagreement.

Reject mismatches; do not fall back to raw weights, another epoch, B/2, random initialization, a downloaded model, or target-conditioned transport.

### Step 4: Implement route execution and artifacts

Support these route names:

```text
native
semigroup
official_head_current_xt
paper_algorithm_split
initial_noise
```

For each generation arm:

- Save one PNG per sample with a stable ordinal and sanitized sample ID.
- Save `per_sample.jsonl` with unique `sample_id`, exact `generated_image_path`, target cosine, and route diagnostics so quality evaluation can join by ID.
- Save `generation_result.json` with E0 and phase-allowed Edev statistics, NFE, exact flow-map call trace, wall time, images/sec, peak allocated VRAM, peak reserved VRAM, checkpoint SHA256, exact config, and sample count.
- Save 64 deterministic visual pages. Each row contains source, matched native, and candidate. Label columns in a separate JSON manifest, not by drawing text over images.
- Save all raw candidate images. Do not save only face-detected images or a selected subset.
- Copy the locked ordered sample-ID manifest and its digest into every arm result. Record the resolved `t_cut`, guided time array, and unguided time array for every FMRG candidate.

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
--projection fixed_radius|typical_shell
--semigroup-report PATH
--schedule-manifest PATH --t-cut FLOAT
--fmrg-variant official_head_current_xt|paper_algorithm_split
--sample-mode flow_map1|flow_map2
--optimization-mode official_adam|paper_normalized_direct_autograd
--num-optim-iters N
```

Config values are the default; explicit CLI values are recorded overrides. `--semigroup-report` and `--schedule-manifest` are mandatory for either FMRG variant. The report, manifest, resolved config, and explicit `--t-cut` must agree on the passing split and checkpoint SHA256 or the run aborts.

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
- Create: `configs/medium_v2/experiments/r8_meanflow_native_ema.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_official_xt_flow_map1_gpu0.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_official_xt_flow_map2_gpu1.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_paper_split_gpu2.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_noise_fixed_eta025.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_noise_fixed_eta05.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_noise_shell_eta1.yaml`
- Create: `configs/medium_v2/experiments/r8_meanflow_noise_shell_eta2.yaml`
- Create: `tests/test_r8_meanflow_guidance_configs.py`

### Step 1: Write config-contract tests first

The tests must load all nine YAML files and assert:

- Exact checkpoint, E0, Edev, E1, E2, VAE, index, and feature paths plus all fixed encoder hashes.
- `checkpoint_model: ema`.
- Expected Stage 2 epoch 1652 and patch size 4.
- `transport_condition: learned_null_condition`.
- `sampling_seed: 1337`.
- Unique experiment and output names.
- Semigroup split times exactly `[0.75, 0.5, 0.25]`.
- Every guided config points to one locked schedule manifest. It resolves `guided_steps=3`, `guided_times=linspace(1,t_cut,4)`, and `unguided_times=linspace(t_cut,0,3)` and rejects a CLI/config `t_cut` mismatch.
- Official variants use `nopt=1`, one config each for `flow_map1` and `flow_map2`, Adam step sizes `[1.0,3.0]`, and normalized eta `[0.25,0.5,1.0,2.0]`.
- The paper variant is named `paper_algorithm_split` and never aliases the official implementation.
- Paper split and noise oracle use the same closed normalized eta `[0.25,0.5,1.0,2.0]`; the four fallback noise configs bind those four eta values to their registered fixed-radius/shell constraints.
- Required metrics are `fid`, `kid`, `niqe`, and `sharpness`.
- Calibration and visual review counts are 64. Full native/winner count is 2048.
- E1/E2 evaluation has `prospective_after_winner_lock: true` and cannot appear in a calibration metric list.

### Step 2: Verify tests fail

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_r8_meanflow_guidance_configs.py -q
```

Expected: missing files.

### Step 3: Create the nine standalone configs

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
schedule_manifest: artifacts/r8_meanflow_flow_map_guidance/semigroup/locked_schedule_manifest.json
e0_checkpoint: artifacts/checkpoints/e0_medium_v1/best.pt
edev_checkpoint: artifacts/checkpoints/e0_resnet18/best.pt
heldout_e1_checkpoint: artifacts/checkpoints/e0_dinov2_large_v2/best.pt
heldout_e2_checkpoint: artifacts/checkpoints/e0_convnext_tiny/best.pt
heldout_eval: prospective_after_winner_lock
vae_path: artifacts/checkpoints/external/sd-vae-ft-ema
vae_scaling_factor: 0.18215
index: data/index/val_face_mixed_e14.jsonl
features: artifacts/e0_features/val_face_mixed_e14_e0_medium_v1
pixel_image_size: 256
batch_size: 2
num_workers: 4
calibration_samples: 64
full_samples: 2048
visual_review_samples: 64
quality_metrics: [fid, kid, niqe, sharpness]
```

Add the route-specific values asserted above. Keep calibration candidates in config:

```text
official Adam step_size candidates: 1.0, 3.0
official normalized eta candidates: 0.25, 0.5, 1.0, 2.0
official sample modes: flow_map1 or flow_map2 as named by the config
SAFA schedule: guided_steps=3, two unguided tail intervals, nopt=1
paper-split eta candidates: 0.25, 0.5, 1.0, 2.0
noise-oracle eta candidates: 0.25, 0.5, 1.0, 2.0
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
test_matrix_semigroup_shards_64_ids_across_all_four_gpus
test_matrix_semigroup_merge_rejects_missing_or_duplicate_ids
test_matrix_calibration_launches_four_processes_concurrently
test_matrix_failed_semigroup_replaces_all_four_arms_with_noise_configs
test_matrix_full_requires_2048_samples_and_visual_review
test_matrix_full_shards_locked_native_and_winner_across_four_gpus
test_matrix_quality_commands_require_manifest_and_per_sample_join
test_matrix_quality_commands_never_use_max_count_flags
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
  GPU0-3 each receive a disjoint deterministic 16-sample shard
  merge all 64 rows by sample ID, then write one gate report

calibrate:
  GPU0 official_head_current_xt flow_map1 closed candidates
  GPU1 official_head_current_xt flow_map2 closed candidates
  GPU2 paper_algorithm_split closed candidates
  GPU3 initial-noise oracle closed candidates
  all four physical GPU processes start before waiting for any one of them
  if semigroup failed, replace GPUs 0-3 with the four registered noise configs

full:
  lock one winner before any E1/E2 forward pass
  GPU0-3 each generate native and winner for one 512-sample modulo shard
  merge to exact 2048 native and 2048 winner manifests
```

For each generated candidate, chain the quality command only after generation succeeds:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/eval_generation_quality.py \
  --real-index data/index/val_face_mixed_e14.jsonl \
  --generated-dir ARM_OUTPUT/generated_images \
  --per-sample-jsonl ARM_OUTPUT/per_sample.jsonl \
  --sample-id-manifest LOCKED_SAMPLE_ID_MANIFEST.jsonl \
  --output ARM_OUTPUT/quality.json \
  --seed 1337 \
  --device cuda:0 \
  --metrics fid kid niqe sharpness
```

The calibration command uses a locked 64-ID manifest; the full command uses the locked 2048-ID manifest. Neither passes max-count flags. Calibration FID is diagnostic and is never the sole selection/ranking key. Write a lock and `matrix_status.json` with exact commands, GPU UUIDs, external process records, start/end times, exit codes, schedule manifest, sample-ID manifest digest, and output paths. Refuse to overwrite any output.

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

## Task 9: Add Selection, Held-Out Evaluation, and Result Aggregation

**Files:**
- Create: `scripts/summarize_r8_meanflow_guidance.py`
- Create: `src/safa/evaluation/encoder_generalization.py`
- Create: `scripts/eval_r8_heldout_encoders.py`
- Create: `tests/test_summarize_r8_meanflow_guidance.py`
- Create: `tests/test_encoder_generalization.py`

### Step 1: Write failing summary tests

Test that the script:

- Joins `generation_result.json`, `quality.json`, and visual review by exact arm ID.
- Refuses mismatched sample counts, checkpoint hashes, seeds, or sample-ID digests.
- Refuses non-finite metrics.
- Does not treat face-detection rate as a quality gate.
- Treats 64-sample calibration FID as diagnostic, never as the sole winner decision.
- Uses only E0 and Edev during calibration and rejects E1/E2 fields before winner lock.
- Calculates `solved` and `directional_evidence` exactly as the design specifies.
- Writes a CSV and Markdown table with FID, KID, NIQE, Sharpness, cosine, NFE, speed, peak VRAM, and severe visual count.
- Requires a direct-review `visual_review.json` before choosing full-run candidates.
- Refuses held-out evaluation without a locked winner and exact 2048 native/winner manifests.
- Computes within-encoder source/generated cosine, pairwise cosine-distance Spearman, and 8-class accuracy.
- Rejects cross-coordinate cosine and a second E1/E2 evaluation under the same protocol marker.

### Step 2: Verify failure

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_summarize_r8_meanflow_guidance.py -q
```

Expected: missing script.

### Step 3: Implement strict aggregation

Use this calibration eligibility filter before ranking official-flow-map1, official-flow-map2, paper-split, or oracle candidates:

```text
all values finite
cosine improvement over matched native >= 0.02
Edev cosine does not move in the opposite direction
Sharpness mean >= 0.80 * native Sharpness mean
severe visual failures <= 10% of the 64 reviewed samples
```

Among eligible candidates, use the pre-registered joint order: higher E0 cosine, then higher Edev cosine, then fewer severe visual failures, then higher Sharpness retention, then lower NFE. Use 64-sample FID only as a reported diagnostic, not as a sole filter or tie breaker. If no candidate is eligible, mark the study branch failed. Do not weaken the filter automatically.

Write:

```text
artifacts/r8_meanflow_flow_map_guidance/selection.json
artifacts/r8_meanflow_flow_map_guidance/summary.csv
artifacts/r8_meanflow_flow_map_guidance/summary.md
```

Implement `eval_r8_heldout_encoders.py` as a separate prospective command. It must load only the fixed E1/E2 checkpoint hashes, validate the locked winner/config/image manifests, run both encoders once, and write `heldout_e1_e2.json` plus an immutable protocol marker. It must never expose E1/E2 to the calibration selection function.

### Step 4: Test and commit

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest \
  tests/test_summarize_r8_meanflow_guidance.py tests/test_encoder_generalization.py -q
git add scripts/summarize_r8_meanflow_guidance.py scripts/eval_r8_heldout_encoders.py \
  src/safa/evaluation/encoder_generalization.py \
  tests/test_summarize_r8_meanflow_guidance.py tests/test_encoder_generalization.py
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

Verify four 16-sample shard manifests merge to the exact deterministic 64 IDs with no missing or duplicate row. Check the merged report against the fixed numerical gate. Open every direct/composed comparison page. Record the selected `t_cut`, or record that no split passed. On pass, write `locked_schedule_manifest.json` with checkpoint hash, gate-report hash, `t_cut`, `guided_steps=3`, the four guided time points, the three unguided time points, sample-ID digest, and selection rule. The launcher must pass both `--schedule-manifest` and the same explicit `--t-cut`.

If the gate fails, skip both FMRG variants and run the four registered noise-oracle constraint/step configurations on GPUs 0-3. Do not leave a GPU idle and do not wait for new input.

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

The launcher must start physical GPUs 0-3 before waiting. Each GPU generates a disjoint 512-sample modulo shard for both native and the locked winner. Do not generate or rank runner-up candidates in this phase.

### Step 2: Verify full counts and metrics

For every completed primary arm, require:

```text
native generated PNG count after merge: 2048
winner generated PNG count after merge: 2048
real metric count: 2048
per-sample JSONL rows: 2048
visual-review manifest sample IDs: 64
all required metric fields finite
checkpoint hashes and sample-ID digests identical across arms
real/native/winner IDs exactly equal the locked 2048 manifest in order
quality payload confirms manifest join and contains no max-count selection
```

Open all full visual pages and update `visual_review.json` with the full-arm IDs.

### Step 3: Run the one-time prospective held-out evaluation

Only after the winner manifest is locked, load E1 and E2 and evaluate matched source/native/winner images. For each encoder `Ek`, calculate only within-encoder quantities:

```text
cos(Ek(generated), Ek(source))
Spearman between off-diagonal pairwise cosine-distance vectors
8-class affect accuracy against validation labels
```

Reject any request for `cos(E1(generated), Z0_E0)` or another cross-coordinate cosine. Persist a marker containing winner config hash, image-manifest hash, E1/E2 checkpoint hashes, and evaluation timestamp. The command must refuse a second prospective run unless artifacts are deleted under an explicit new protocol commit.

### Step 4: Produce the final table

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
  tests/test_encoder_generalization.py \
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
- Locked `t_cut`, explicit SAFA guided/unguided time arrays, and schedule-manifest hash.
- Locked 2048 sample-ID manifest digest proving identical real/native/winner membership.
- Semigroup residual table and pass/fail decision.
- Full metric table with FID, KID, NIQE, Sharpness, cosine, NFE, speed, and VRAM.
- Separate E0, Edev, E1, and E2 columns; E1/E2 include within-encoder cosine, pairwise-distance Spearman, and 8-class accuracy.
- The locked-winner/prospective marker proving E1/E2 were evaluated only after selection.
- Direct visual-failure counts with linked artifact paths.
- `solved`, `directional evidence`, or `failed` label using the fixed rules.
- The pinned official-code commit/license/entry table and a plain statement that MPGD, LGD, DPS, FreeDoM, and Universal Guidance were not reproduced on this 1-step model.
- Separate rows for `official_head_current_xt` and `paper_algorithm_split`; never collapse them into one FMRG-J name.
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
