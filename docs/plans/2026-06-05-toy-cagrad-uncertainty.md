# Toy CAGrad And Uncertainty Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and run a toy-layer CAGrad experiment against a clearly named uncertainty scalar weighting baseline.

**Architecture:** Keep all changes inside the existing toy FM/CL script and its tests/configs. CAGrad uses only the two task gradients, FM and CL, and solves the two-task simplex problem directly. The uncertainty baseline uses homoscedastic scalar weights and is not named GradNorm.

**Tech Stack:** Python, PyTorch, pytest, tmux.

---

### Task 1: Add Failing Tests

**Files:**
- Modify: `tests/test_toy_fm_cl_projected.py`

**Steps:**
1. Add a direct CAGrad aggregation test using the TorchJD two-row example.
2. Add a smoke test that runs `cagrad` and checks `cagrad_fm_weight`, `cagrad_cl_weight`, `gradient_cosine_mean`, `combined_grad_norm`, `valid_fm_loss`, and `repr_cosine_mean`.
3. Add a smoke test that runs `uncertainty_weighted` and checks the logged formula fields.
4. Run the new tests and confirm they fail because the methods are not implemented.

### Task 2: Implement Minimal Toy Methods

**Files:**
- Modify: `scripts/run_toy_fm_cl_projected.py`

**Steps:**
1. Add `cagrad` and `uncertainty_weighted` to `SUPPORTED_METHODS`.
2. Add `cagrad_c` and uncertainty log-variance fields to `ToyConfig`.
3. Implement two-task CAGrad aggregation with a deterministic one-dimensional convex solve over the FM/CL simplex.
4. Implement uncertainty scalar weighting as `0.5 * exp(-s_fm) * L_fm + 0.5 * s_fm + 0.5 * exp(-s_cl) * L_cl + 0.5 * s_cl`.
5. Extend metric windows and empty stats with CAGrad weights, gradient cosine, combined norm, and uncertainty fields.

### Task 3: Add Run Configs

**Files:**
- Create: `configs/toy_fm_cl_projected_delta45_cagrad_gpu5_bs8192_20260605.json`
- Create: `configs/toy_fm_cl_projected_delta45_uncertainty_gpu6_bs8192_20260605.json`

**Steps:**
1. Use batch size 8192, delta 45, and unique run names.
2. Set devices to `cuda:5` and `cuda:6` only in config files.
3. Keep output paths under new run names so old artifacts are not overwritten.

### Task 4: Validate, Commit, And Launch

**Steps:**
1. Run the targeted pytest file.
2. Commit only the toy implementation, tests, configs, and this plan.
3. Start tmux jobs only if physical/logical GPU5 and GPU6 are visible.
4. If experiment scripts only write metrics at the end, start CPU-only watchers against the two new run dirs.
