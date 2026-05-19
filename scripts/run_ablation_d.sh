#!/usr/bin/env bash
set -euo pipefail
cd /home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
export CUDA_VISIBLE_DEVICES=${SAFA_GPU:-7}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH=src
/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m safa.cli.train_g     --config configs/ablation/ablation_d_lambda_002.yaml
