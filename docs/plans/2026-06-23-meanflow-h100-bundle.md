# MeanFlow H100 Bundle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a portable E15 MeanFlow resume bundle under 100GB for H100 1/2/4 GPU training.

**Architecture:** The bundle carries code, 256 JPEG q95 images, bundle-local indexes, E0 feature shards, required checkpoints, and H100 scripts. A prepare script rewrites runtime absolute paths and cache manifests after unpacking.

**Tech Stack:** Python, PyYAML, PIL, PyTorch checkpoint loading, tar+zstd, torchrun/DDP with NCCL.

---

### Task 1: H100 Runtime Scripts

Create `scripts/h100/setup_meanflow_env.sh`, `scripts/h100/prepare_h100_bundle.py`, `scripts/h100/train_meanflow_h100.sh`, and `scripts/h100/verify_bundle.py`. Verify shell syntax and Python compile.

### Task 2: Bundle Builder

Create `scripts/h100/build_meanflow_h100_bundle.py` to resize indexed images to 256 JPEG q95, copy required weights only, freeze a readable E15 snapshot, and create a tar.zst archive plus SHA256.

### Task 3: Documentation

Create `docs/meanflow_h100_resume_bundle.md` with unpack, setup, verify, single/multi GPU train, epoch recovery, OOM, eval failure, and NCCL troubleshooting steps.

### Task 4: Verification

Run targeted tests, build the bundle, verify file counts and checkpoint readability, check archive size under 100GB, list key archive entries, and commit only scripts/docs/tests.
