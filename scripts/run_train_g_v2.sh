#!/bin/bash
cd /home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
export CUDA_VISIBLE_DEVICES=4,5
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH=src
mkdir -p artifacts/logs
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m torch.distributed.run     --standalone --nproc_per_node=2     -m safa.cli.train_g     --config configs/train_g_v2.yaml 2>&1 | tee artifacts/logs/train_g_v2.log
