# MEDIUM V2 CL-Only Ablation Plan

Date: 2026-06-01

## Shared Inputs

- E0 checkpoint: `artifacts/checkpoints/e0_medium_v1/best.pt`
- Train E0 cache: `artifacts/e0_features/train_balanced_medium_e0_medium_v1`
- Validation E0 cache: `artifacts/e0_features/val_single_face_e0_medium_v1`
- Stage1 resume checkpoint: `artifacts/checkpoints/g_medium_v1_stage1_long200_v4/best_stage1.pt`

## Runs

| Run | Config | GPUs | NPROC | Checkpoint dir | Quality dir |
| --- | --- | --- | ---: | --- | --- |
| null-conditioned FM | `configs/medium_v2/train_g_medium_v2_stage2_null_fm.yaml` | `1` | 1 | `artifacts/checkpoints/g_medium_v2_stage2_null_fm` | `artifacts/eval/g_medium_v2_stage2_null_fm/quality` |
| point-only CL-only | `configs/medium_v2/train_g_medium_v2_stage2_point_only_cl_only.yaml` | `3,4` | 2 | `artifacts/checkpoints/g_medium_v2_stage2_point_only_cl_only` | `artifacts/eval/g_medium_v2_stage2_point_only_cl_only/quality` |
| point+Gram CL-only | `configs/medium_v2/train_g_medium_v2_stage2_point_gram_cl_only.yaml` | `5,6` | 2 | `artifacts/checkpoints/g_medium_v2_stage2_point_gram_cl_only` | `artifacts/eval/g_medium_v2_stage2_point_gram_cl_only/quality` |

The null-FM config has `stage2_objective.type: fm_only_probe`, `flow_condition: fixed_null_condition`, and no repr weights.

## Launch Commands

Run from repo root:

```bash
SAFA_CUDA_VISIBLE_DEVICES=1 scripts/run_train_g_tmux.sh \
  --config configs/medium_v2/train_g_medium_v2_stage2_null_fm.yaml \
  --session medium_v2_null_fm \
  --log artifacts/logs/medium_v2_null_fm.log \
  --nproc-per-node 1

SAFA_CUDA_VISIBLE_DEVICES=3,4 scripts/run_train_g_tmux.sh \
  --config configs/medium_v2/train_g_medium_v2_stage2_point_only_cl_only.yaml \
  --session medium_v2_point_only_cl_only \
  --log artifacts/logs/medium_v2_point_only_cl_only.log \
  --nproc-per-node 2

SAFA_CUDA_VISIBLE_DEVICES=5,6 scripts/run_train_g_tmux.sh \
  --config configs/medium_v2/train_g_medium_v2_stage2_point_gram_cl_only.yaml \
  --session medium_v2_point_gram_cl_only \
  --log artifacts/logs/medium_v2_point_gram_cl_only.log \
  --nproc-per-node 2
```

## Checkpoint Visual Watchers

Use GPU0 for rendering. `--backfill-every 1` asks the watcher to render every completed epoch.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/watch_medium_v2_checkpoint_visuals.py \
  --interval 300 \
  --backfill-every 1 \
  --checkpoint-dir artifacts/checkpoints/g_medium_v2_stage2_null_fm \
  --config configs/medium_v2/train_g_medium_v2_stage2_null_fm.yaml \
  --index data/index/val_single_face.jsonl \
  --features artifacts/e0_features/val_single_face_e0_medium_v1 \
  --out-dir artifacts/plots/medium_v2_cl_ablation/null_fm \
  --run-name null_fm_checkpoint_visuals \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/watch_medium_v2_checkpoint_visuals.py \
  --interval 300 \
  --backfill-every 1 \
  --checkpoint-dir artifacts/checkpoints/g_medium_v2_stage2_point_only_cl_only \
  --config configs/medium_v2/train_g_medium_v2_stage2_point_only_cl_only.yaml \
  --index data/index/val_single_face.jsonl \
  --features artifacts/e0_features/val_single_face_e0_medium_v1 \
  --out-dir artifacts/plots/medium_v2_cl_ablation/point_only_cl_only \
  --run-name point_only_cl_only_checkpoint_visuals \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/watch_medium_v2_checkpoint_visuals.py \
  --interval 300 \
  --backfill-every 1 \
  --checkpoint-dir artifacts/checkpoints/g_medium_v2_stage2_point_gram_cl_only \
  --config configs/medium_v2/train_g_medium_v2_stage2_point_gram_cl_only.yaml \
  --index data/index/val_single_face.jsonl \
  --features artifacts/e0_features/val_single_face_e0_medium_v1 \
  --out-dir artifacts/plots/medium_v2_cl_ablation/point_gram_cl_only \
  --run-name point_gram_cl_only_checkpoint_visuals \
  --device cuda:0
```

## Curve Comparison

After both CL-only runs have history and quality JSON:

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python \
  scripts/plot_medium_v2_cl_ablation_curves.py \
  --out-dir artifacts/plots/medium_v2_cl_ablation
```
