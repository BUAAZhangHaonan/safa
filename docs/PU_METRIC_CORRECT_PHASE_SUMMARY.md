# PU Metric-Correct Implementation: Phase Summary

**Date**: 2026-06-10
**Branch**: `master`

---

## What Was Done

### Problem

The original PU (Projected Update) implementation used AdamW optimizer but projected gradients in Euclidean space. This metric mismatch breaks the first-order guarantee that "projected step does not harm FM." Three additional issues compound this:

1. FM target computed multiple times per batch with independent randomness (guard gradient does not match FM step target)
2. FM/repr gradient norms differ by ~18x with no normalization
3. No post-step verification that FM loss did not increase

### Solution: Q-Weighted Projection (AdamW Metric)

Instead of switching to SGD (which would match Euclidean projection geometry), we made the projection geometry match AdamW's optimizer geometry.

**Key insight**: AdamW's effective update is `param -= lr * w * g` where `w = 1/(sqrt(v_hat) + eps)` from the second moment. The natural inner product in Adam's space is:

```
<a, b>_Q = sum_i( w_i * a_i * b_i )
```

We project onto the FM-feasible cone using this Q-weighted inner product instead of Euclidean. This makes the projection guarantee "no FM harm" in the same metric that AdamW actually moves parameters.

### Code Changes

| File | Lines Added | Description |
|------|-------------|-------------|
| `src/safa/training/projected_update.py` | +94 | Core math: `project_gradient_onto_fm_feasible_cone_adam()`, `_dot_weighted()`, `_squared_norm_weighted()` |
| `src/safa/training/g_loop.py` | +252 | Config parsing, AdamW preconditioner extraction, Q-norm normalization, preconditioned parameter step, backtracking line search |
| `scripts/run_toy_fm_cl_projected.py` | +209 | Toy experiment mirrors all g_loop.py changes |

New config fields:
- `optimizer_type: "adamw"` (also supports `"sgd"`)
- `pu_gradient_normalization: true` (normalize repr grad to match FM Q-norm before projection)
- `pu_backtrack_max_retries: 3` (post-step guard with backtracking)
- `pu_fm_increase_budget: 0.0` (tolerance for FM loss increase)

---

## Experimental Results

### AffectNet Stage 1 (Baseline)

Pure FM trained 200 epochs with null-embedding conditioning (no identity information).

| Metric | Value |
|--------|-------|
| cosine_similarity | 0.639 |
| source_prediction | 0.492 (random) |
| NIQE | ~5.0 |

### AffectNet Stage 2 — AdamW Metric-Correct PU (5 epochs)

Loaded Stage 1 checkpoint, running PU with repr conditioning.

| Metric | Value |
|--------|-------|
| cosine_similarity | 0.952 |
| NIQE | 5.85 |
| flow_loss | 0.061 |
| pu_norm_ratio (avg) | 18.1 |
| pu_backtrack_count (avg) | 1.49/step |
| pu_effective_repr_lr | 57% of configured |

### Projection Diagnostics

- 79% of steps have zero conflict (projection removes nothing)
- When conflict exists, projection removes only ~0.5% of repr gradient norm
- FM loss remains stable (projection is working as intended)

### Toy Experiment (2D Synthetic)

SGD-PU with gradient normalization converges correctly. AdamW metric-correct path shows same qualitative behavior. Full sweep results in `configs/adamw_pu_metric_toy_sweep.json` and `configs/sgd_pu_toy_sweep.json`.

---

## Current Problems

### 1. Backtracking Over-Triggers

**Symptom**: Average 1.49 backtracks per step, effective repr lr only 57% of configured.

**Root cause**: Post-step FM loss check uses a fresh random `t` (timestep), different from the `t` used for the FM step and guard computation. Flow matching loss is noisy across different `t` values, so the check frequently sees a "loss increase" that is just noise.

**Impact**: Repr learning rate is effectively cut by ~43%. The optimizer is much more conservative than configured.

**Fix**: Reuse the same `x_0, t` for all three evaluations (FM step, guard, post-step check). This was planned but not yet implemented in the AffectNet training code.

### 2. pu_fm_increase_budget = 0.0 Is Too Strict

**Symptom**: Combined with the random-t issue above, budget=0.0 means any positive FM loss fluctuation triggers backtracking.

**Fix**: Either fix the t-consistency issue (which makes budget=0.0 valid), or relax to a small positive budget (e.g., 1e-4) that absorbs noise.

### 3. Repr Cosine Similarity Plateau

**Symptom**: Repr cosine is 0.952 after 5 epochs. Repr-only training reaches ~1.0.

**Why**: The 43% effective lr reduction from over-triggering backtracking is the primary bottleneck. The projection itself is nearly lossless (only 0.5% norm removed on conflicting steps). The gap between 0.95 and 1.0 is almost entirely caused by the backtracking issue.

**Expected after fix**: With t-consistent backtracking, effective lr should be close to 100% of configured. Cosine should approach 1.0 at a rate proportional to the configured repr learning rate.

### 4. pu_norm_ratio = 18.1 (Repr is 18x FM in Q-norm)

**What it means**: The repr gradient has 18x the Q-norm of the FM gradient. Without normalization, the projection would be dominated by the repr gradient's scale.

**Current handling**: `pu_gradient_normalization: true` scales repr gradient to match FM Q-norm before projection, then scales back after.

**Potential improvement**: The 18x ratio itself is worth investigating. It may indicate that the repr learning rate is too high relative to FM learning rate, or that the repr loss landscape is much steeper. A ratio closer to 1.0 would reduce the need for normalization.

---

## Improvement Roadmap

### Priority 1: Fix FM Target Consistency (t-reuse)

In `_run_projected_stage2_batch()`, sample `x_0` and `t` once and reuse for:
- FM step (compute g_fm)
- Guard check (compute g_guard, compare with g_fm)
- Post-step check (verify FM loss after repr step)

This is the single highest-impact fix. It should eliminate most false backtracks and bring effective repr lr close to 100%.

### Priority 2: Tune Repr Learning Rate

After fixing t-consistency, the effective lr will jump from ~57% to ~100% of configured. The current `repr_learning_rate: 0.00003` may become too aggressive. Monitor for instability and reduce if needed.

Alternatively, if cosine converges faster than expected, keep the lr and benefit from faster convergence.

### Priority 3: Relax pu_fm_increase_budget

With t-consistent checks, budget=0.0 should work correctly. But if there is still some noise (e.g., from dropout or data augmentation), a small positive budget (1e-4 to 1e-3) would be safer.

### Priority 4: Investigate Norm Ratio

The 18x Q-norm ratio between repr and FM gradients is large. Options:
- Adjust relative learning rates to bring ratio closer to 1.0
- Use per-gradient-type preconditioning
- Accept the ratio and rely on normalization (current approach)

---

## Config Files

| File | Description |
|------|-------------|
| `configs/medium_v2/train_g_medium_v2_stage2_m3_point_projected_adamw_metric.yaml` | AffectNet Stage 2 AdamW metric-correct config |
| `configs/adamw_pu_metric_toy_sweep.json` | Toy sweep, AdamW metric, gradient normalization, backtracking |
| `configs/sgd_pu_toy_sweep.json` | Toy sweep, SGD Euclidean, gradient normalization, backtracking |
| `configs/adamw_pu_toy_sweep.json` | Old toy sweep, AdamW, no normalization, no backtracking |

---

## Training Status

- **AffectNet Stage 2**: Running on server 4029, GPU 5. Currently epoch 0/120, ~6s/iteration. Expected completion: ~2 days.
- **Monitor**: `pu_backtrack_count` should decrease significantly after t-consistency fix. `cosine_similarity` should approach 1.0.
