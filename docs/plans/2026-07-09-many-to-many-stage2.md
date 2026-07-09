# SAFA Many-to-Many Stage2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Train Stage2 with source features and target images decoupled, so the generator no longer learns `E0(x0) -> x0` reconstruction.

**Architecture:** Keep the current feature-cache contract for source `z`. Add an opt-in dataset path that pairs each source record with a different target record from the same affect label bucket. The training loop still receives `image`, `z`, `label`, and `sample_id`, so existing objectives stay intact; extra metadata records the source/target relation.

**Tech Stack:** Python, PyTorch Dataset/DataLoader, existing SAFA YAML configs, existing `FeatureAlignedAffectNet`, existing MeanFlow/SiT Stage2 runner.

---

## Design

Recommend same-label identity-disjoint many-to-many pairing. Alternative is many-to-one, but that makes the target identity a fixed attractor and encourages the model to ignore `z`. Another alternative is face-ID/VL matching before training, but that adds a new moving part before we know whether the simple data relation fixes collapse.

The first implementation uses a deterministic cyclic pairing inside each label bucket:

```text
source: train_index + train_features -> z_source, source_label, source_sample_id
target: target_index -> image_target, target_label, target_sample_id
constraint: target_label == source_label and target_sample_id != source_sample_id
output image: image_target
output z: z_source
```

This is many-to-many because every source gets a target, and target selection is offset by `pairing_seed` and optional `pairs_per_source`. It is not formal identity protection. It only prevents exact same-record reconstruction in training. The real identity check remains post-hoc validation, as requested.

Validation and quality eval stay on the original aligned validation loader first. That keeps metrics comparable to previous runs. The training-only switch isolates the experiment to the suspected failure point.

## Stop Rule

Run short controlled experiments first. Stop this line if early validation shows the same collapse pattern: face rate falls hard or FID/NIQE explodes while latent cosine improves. Do not add VL, ArcFace filtering, or post-processing bandages until the simple many-to-many data relation has been tested.

## Tasks

### Task 1: Plan artifact

**Files:**
- Create: `docs/plans/2026-07-09-many-to-many-stage2.md`

**Steps:**
1. Save this design and implementation plan.
2. Run `git status --short` and verify only this plan is staged.
3. Commit and push on `master`.

### Task 2: Dataset tests first

**Files:**
- Modify: `tests/test_feature_dataset.py`

**Steps:**
1. Add a helper that writes several source and target images plus a source feature cache.
2. Add a failing test for same-label different-sample pairing.
3. Add a failing test that a singleton target bucket raises a clear error.
4. Run: `PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests/test_feature_dataset.py -q`
5. Expected: new tests fail because the many-to-many dataset does not exist yet.
6. Commit and push only the tests.

### Task 3: Dataset implementation

**Files:**
- Modify: `src/safa/data/feature_dataset.py`

**Steps:**
1. Add `ManyToManyFeatureAlignedAffectNet`.
2. Reuse `load_feature_cache` for source features.
3. Read target records with `read_index`.
4. Build target buckets by label.
5. Pair deterministically with `target_position = (source_index + pairing_seed) % bucket_size`, then advance if the target sample id equals source sample id.
6. Return existing keys plus `source_sample_id`, `target_sample_id`, `target_label`, and `pair_id`.
7. Run the same focused pytest command.
8. Commit and push only this implementation.

### Task 4: Training config integration

**Files:**
- Modify: `src/safa/training/g_loop.py`
- Modify or add tests in the most focused existing training test file.

**Steps:**
1. Add a helper such as `_build_train_feature_dataset(config, transform)`.
2. If `config["many_to_many"]["enabled"]` is true, instantiate `ManyToManyFeatureAlignedAffectNet` with `target_index`, `pairing_seed`, and `pairs_per_source`.
3. Otherwise keep `FeatureAlignedAffectNet` behavior unchanged.
4. Keep validation and quality eval unchanged.
5. Add tests that the default path still uses `FeatureAlignedAffectNet` and the opt-in path uses many-to-many.
6. Run focused tests.
7. Commit and push.

### Task 5: Experiment configs

**Files:**
- Create configs under `configs/medium_v2/experiments/`
- Create helper launcher under `scripts/` only if existing launch style needs it.

**Steps:**
1. Use full-parameter point-projected configs, not LoRA/PEFT.
2. Set source cache to `artifacts/e0_features/train_face_mixed_e14_e0_medium_v1`.
3. Set target index to `data/index/train_face_mixed_e14_4029avail.jsonl`.
4. Create four short configs for GPUs 0-3:
   - many-to-many lambda 0.25
   - many-to-many lambda 0.5
   - many-to-many lower lr 5e-5 lambda 0.5
   - aligned control or lambda 0.0 sanity check
5. Keep output dirs unique.
6. Run config smoke tests if available, otherwise run YAML parse/import checks.
7. Commit and push.

### Task 6: Launch and monitor

**Files:**
- No required repo file edits unless a small launcher is committed in Task 5.

**Steps:**
1. Verify GPUs 0-3 are free.
2. Launch four short jobs using `CUDA_VISIBLE_DEVICES=0`, `1`, `2`, `3`.
3. Monitor logs and first validation artifacts.
4. Stop failed/collapsing jobs instead of extending the search.
5. Report the actual metrics and whether many-to-many improved the quality/repr tradeoff.
