#!/bin/bash
# Run ON the GPU server (inside tmux). User-level Python environment for the Decision 3.0 runs; no sudo.
# Uses a uv-managed standalone CPython because Triton needs Python.h, which the system Python may lack.
# Usage: REMOTE_ROOT=/home/<USER>/teaching-one-decision bash setup_env.sh
set -euo pipefail
ROOT=${REMOTE_ROOT:?set REMOTE_ROOT}
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
VENV=$ROOT/venv
LOG=$ROOT/logs/setup-env-$(date +%F_%H%M%S).log
mkdir -p "$ROOT/logs"; exec > >(tee -a "$LOG") 2>&1
test ! -e "$VENV" || { echo "ABORT: $VENV exists"; exit 3; }
"$UV" python install 3.12
"$UV" venv --managed-python --python 3.12 "$VENV"
PY=$VENV/bin/python
# aarch64 or x86_64 CUDA 13.0 wheels; change the index for another CUDA version.
"$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 torchvision==0.26.0
"$UV" pip install --python "$PY" transformers==5.17.0 accelerate==1.15.0 "safetensors>=0.6" "huggingface_hub>=1.0" \
  flash-linear-attention==0.5.2 einops "pillow>=10" opencv-python-headless peft==0.21.0
cat > "$ROOT/logs/triton_check.py" <<'PYEOF'
import torch, triton, triton.language as tl
@triton.jit
def add(x, y, out, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(out + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
x = torch.randn(4096, device="cuda"); y = torch.randn(4096, device="cuda"); o = torch.empty_like(x)
add[(4,)](x, y, o, 4096, B=1024); assert torch.allclose(o, x + y)
print("TRITON_KERNEL_PASS", triton.__version__, torch.__version__, torch.cuda.get_device_name(0))
PYEOF
"$PY" "$ROOT/logs/triton_check.py"
"$UV" pip freeze --python "$PY" > "$ROOT/logs/pip-freeze.txt"
echo "SETUP_ENV_EXIT=0"
