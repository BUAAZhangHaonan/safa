# SAFA Many-to-Many Stage2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Train Stage2 with source features and target images decoupled, so the generator no longer learns  reconstruction.

**Architecture:** Keep the current feature-cache contract for source . Add an opt-in dataset path that pairs each source record with a different target record from the same affect label bucket. The training loop still receives , , , and , so existing objectives stay intact; extra metadata records the source/target relation.

**Tech Stack:** Python, PyTorch Dataset/DataLoader, existing SAFA YAML configs, existing , existing MeanFlow/SiT Stage2 runner.

---

## Design

Recommend same-label identity-disjoint many-to-many pairing. Alternative is many-to-one, but that makes the target identity a fixed attractor and encourages the model to ignore . Another alternative is face-ID/VL matching before training, but that adds a new moving part before we know whether the simple data relation fixes collapse.

The first implementation uses a deterministic cyclic pairing inside each label bucket:



This is many-to-many because every source gets a target, and target selection is offset by  and optional . It is not formal identity protection. It only prevents exact same-record reconstruction in training. The real identity check remains post-hoc validation, as requested.

Validation and quality eval stay on the original aligned validation loader first. That keeps metrics comparable to previous runs. The training-only switch isolates the experiment to the suspected failure point.

## Stop Rule

Run short controlled experiments first. Stop this line if early validation shows the same collapse pattern: face rate falls hard or FID/NIQE explodes while latent cosine improves. Do not add VL, ArcFace filtering, or post-processing bandages until the simple many-to-many data relation has been tested.

## Tasks

### Task 1: Plan artifact

**Files:**
- Create: 

**Steps:**
1. Save this design and implementation plan.
2. Run  and verify only this plan is staged.
3. Commit and push on .

### Task 2: Dataset tests first

**Files:**
- Modify: 

**Steps:**
1. Add a helper that writes several source and target images plus a source feature cache.
2. Add a failing test for same-label different-sample pairing.
3. Add a failing test that a singleton target bucket raises a clear error.
4. Run:
   
5. Expected: new tests fail because the many-to-many dataset does not exist yet.
6. Commit and push only the tests.

### Task 3: Dataset implementation

**Files:**
- Modify: 

**Steps:**
1. Add .
2. Reuse  for source features.
3. Read target records with .
4. Build target buckets by label.
5. Pair deterministically with , then advance if the target sample id equals source sample id.
6. Return existing keys plus , , , and .
7. Run the same focused pytest command.
8. Commit and push only this implementation.

### Task 4: Training config integration

**Files:**
- Modify: 
- Modify or add tests in the most focused existing training test file.

**Steps:**
1. Add a helper such as .
2. If  is true, instantiate  with , , and .
3. Otherwise keep  behavior unchanged.
4. Keep validation and quality eval unchanged.
5. Add tests that the default path still uses  and the opt-in path uses many-to-many.
6. Run focused tests.
7. Commit and push.

### Task 5: Experiment configs

**Files:**
- Create configs under 
- Create helper launcher under  only if existing launch style needs it.

**Steps:**
1. Use full-parameter point-projected configs, not LoRA/PEFT.
2. Set source cache to .
3. Set target index to .
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
2. Launch four short jobs using , , , .
3. Monitor logs and first validation artifacts.
4. Stop failed/collapsing jobs instead of extending the search.
5. Report the actual metrics and whether many-to-many improved the quality/repr tradeoff.
