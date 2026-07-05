# MeanFlow E15 H100 Resume Bundle

这份文档从“已经把压缩包下载到新服务器”开始。目标是在 H100 80GB 的 1/2/4 卡单机环境上继续 E15 MeanFlow 训练到总 `2400 epoch`。

## 1. 解压

```bash
mkdir -p ~/meanflow_resume
cd ~/meanflow_resume
# 文件名以实际下载为准
tar --use-compress-program=unzstd -xf meanflow_e15_h100_bundle_YYYYMMDD.tar.zst
cd meanflow_e15_h100_bundle_YYYYMMDD/meanflow_e15_h100_bundle
```

如果系统没有 `unzstd`：

```bash
sudo apt-get update
sudo apt-get install -y zstd
```

先检查包内容：

```bash
ls
cat BUNDLE_MANIFEST.json
```

## 2. 配置 conda 环境

环境名固定为 `meanflow`：

```bash
bash scripts/h100/setup_meanflow_env.sh
conda activate meanflow
```

服务器驱动版本按 `585` 设计，驱动 `585` 最高支持 CUDA 13.0；环境脚本会优先尝试安装 `cu130` PyTorch wheel。若该 wheel 不可用，会自动回退到官方 `cu128` wheel。驱动 `585` 也可以运行 CUDA 12.8 wheel。

如果 `conda` 命令不存在，先安装 Miniconda 或 Mambaforge，然后重新运行脚本。

如果 `insightface` 或 `onnxruntime-gpu` 安装失败，通常是构建工具或系统库缺失。先安装常见构建依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libgl1 libglib2.0-0
bash scripts/h100/setup_meanflow_env.sh
```

Flash Attention 不是必须项。默认不安装。当前代码的 `attention_backend: auto` 会优先探测可用后端，不能用 FA 时会走 SDPA/native。

## 3. 验证 bundle

单卡默认验证：

```bash
conda activate meanflow
PYTHONPATH=src python scripts/h100/verify_bundle.py --bundle-root . --gpu-count 1
```

2 卡验证：

```bash
PYTHONPATH=src python scripts/h100/verify_bundle.py --bundle-root . --gpu-count 2 --cuda-visible-devices 0,1
```

4 卡验证：

```bash
PYTHONPATH=src python scripts/h100/verify_bundle.py --bundle-root . --gpu-count 4 --cuda-visible-devices 0,1,2,3
```

验证会做这些事：

- 生成当前机器绝对路径的 runtime index；
- 生成与 runtime index hash 匹配的 E0 cache manifest；
- 生成 runtime config；
- 检查 E15 snapshot checkpoint 可读；
- 检查 train/val index 数量。

## 4. 继续训练

默认脚本会自动检测 GPU 数，并生成 runtime config。

单卡 H100 80GB，默认 `per_device_batch=384, global_batch=384`：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/h100/train_meanflow_h100.sh
```

2 卡，默认 `per_device_batch=384, global_batch=768`：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/h100/train_meanflow_h100.sh
```

4 卡，默认 `per_device_batch=256, global_batch=1024`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/h100/train_meanflow_h100.sh
```

也可以手动覆盖 batch：

```bash
CUDA_VISIBLE_DEVICES=0 PER_DEVICE_BATCH=320 GLOBAL_BATCH=320 bash scripts/h100/train_meanflow_h100.sh
CUDA_VISIBLE_DEVICES=0,1 PER_DEVICE_BATCH=256 GLOBAL_BATCH=512 bash scripts/h100/train_meanflow_h100.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PER_DEVICE_BATCH=256 GLOBAL_BATCH=1024 bash scripts/h100/train_meanflow_h100.sh
```

硬性约束是：

```text
global_batch_size == per_device_batch_size * GPU 数
```

否则 prepare 脚本会拒绝启动。

## 5. 确认恢复 epoch

训练启动后看日志：

```bash
tail -f artifacts/logs/e15_meanflow_h100_resume_*.log
```

也可以检查 checkpoint：

```bash
python - <<'PY'
import torch
ckpt = torch.load('artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last.pt', map_location='cpu', weights_only=False)
print(ckpt.get('metrics', {}).get('stage_epoch_1based'))
PY
```

`stage2.epochs: 2400` 表示总目标 epoch，不是追加 2400 epoch。训练会从 snapshot 里的 `stage_epoch + 1` 继续。

## 6. 常见问题

### OOM

先降低 batch：

```bash
CUDA_VISIBLE_DEVICES=0 PER_DEVICE_BATCH=320 GLOBAL_BATCH=320 bash scripts/h100/train_meanflow_h100.sh
CUDA_VISIBLE_DEVICES=0,1 PER_DEVICE_BATCH=256 GLOBAL_BATCH=512 bash scripts/h100/train_meanflow_h100.sh
```

脚本默认设置了：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTORCH_ALLOC_CONF=expandable_segments:True
```

如果仍然在 VAE encode OOM，下一步应预提取 VAE latent，去掉在线 VAE encode 的显存和 IO 压力。

### FID/KID eval 失败

优先切到 CPU 分布评估：

```bash
MEANFLOW_DISTRIBUTION_DEVICE=cpu CUDA_VISIBLE_DEVICES=0 bash scripts/h100/train_meanflow_h100.sh
```

或者在 runtime config 中降低：

```yaml
distribution_max_samples: 1024
```

### NCCL 问题

多卡使用 `torchrun` 和 `nccl`。如果 NCCL 报网卡错误，先限制本机通信：

```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
CUDA_VISIBLE_DEVICES=0,1 bash scripts/h100/train_meanflow_h100.sh
```

### cu130 wheel 不存在

这是正常情况。服务器驱动 585 最高支持 CUDA 13.0；安装脚本会优先尝试 cu130 wheel，不可用时自动回退 cu128。驱动 585 可以运行 cu128 wheel。

## 7. 关键路径

- 训练入口：`scripts/h100/train_meanflow_h100.sh`
- 环境脚本：`scripts/h100/setup_meanflow_env.sh`
- prepare 脚本：`scripts/h100/prepare_h100_bundle.py`
- verify 脚本：`scripts/h100/verify_bundle.py`
- runtime config：`configs/medium_v2/experiments/e15_meanflow_sit_b_face_mixed_resume_h100_runtime.yaml`
- E15 snapshot：`artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last.pt`
- 256 JPEG q95 数据：`data/face_256_q95/`
- bundle-local index：`data/index_bundle/`
- runtime absolute index：`data/index_runtime/`

## 8. 不要改的点

- 不要把 `resume_from` 改成 E14 checkpoint。
- 不要把 `stage2.epochs` 理解为追加 epoch。
- 不要把原始 1024 FFHQ/CelebA-HQ 数据重新放进包里。
- 不要在多卡下使用 `gloo` 后端；H100 多卡使用 `nccl`。
