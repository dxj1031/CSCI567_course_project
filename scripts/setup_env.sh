#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/setup_env.sh /project2/<PI>_<project_id>/envs/cs567-baseline"
  exit 1
fi

ENV_PREFIX="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y --prefix "$ENV_PREFIX" python=3.11 pip

export PYTHONNOUSERSITE=1

"$ENV_PREFIX/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_PREFIX/bin/python" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
"$ENV_PREFIX/bin/python" -m pip install --no-cache-dir -r "$PROJECT_ROOT/requirements.txt"

"$ENV_PREFIX/bin/python" - <<'PY'
import torch
print("torch_version =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_name =", torch.cuda.get_device_name(0))
PY

echo "Environment ready at $ENV_PREFIX"
