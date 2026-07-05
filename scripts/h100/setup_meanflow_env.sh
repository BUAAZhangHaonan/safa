#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="meanflow"
PYTHON_VERSION="3.12"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Mambaforge first, then rerun this script." >&2
  exit 1
fi

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"

python -m pip install -U pip setuptools wheel

set +e
python -m pip install \
  torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
TORCH_STATUS=$?
set -e

if [[ "$TORCH_STATUS" -ne 0 ]]; then
  echo "cu130 PyTorch wheel install failed; falling back to official cu128 wheels. Driver 585 can run cu128 wheels." >&2
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

python -m pip install -e ".[quality,privacy]"
python -m pip install \
  diffusers==0.38.0 \
  transformers==5.12.0 \
  accelerate==1.14.0 \
  timm==1.0.27 \
  einops==0.8.2 \
  safetensors==0.8.0 \
  torch-fidelity==0.4.0

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  python -m pip install flash-attn --no-build-isolation || {
    echo "flash-attn install failed. This is optional; MeanFlow will use SDPA/native attention through attention_backend=auto." >&2
  }
fi

python - <<'PY'
import torch
print('python ok')
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device_count', torch.cuda.device_count())
    print('device0', torch.cuda.get_device_name(0))
PY
